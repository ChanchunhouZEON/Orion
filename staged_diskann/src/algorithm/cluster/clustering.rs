/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::cluster::ClusterPointManager;
use diskann::common::ANNResult;
use diskann::model::{CsrGraph, InmemDataset};
use ndarray::{ArcArray1, Array1};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use vector::{FullPrecisionDistance, Metric};

#[cfg(feature = "indicatif")]
use indicatif::ProgressFinish::AndLeave;
#[cfg(feature = "indicatif")]
use indicatif::ProgressIterator;

pub const INVALID_CLUSTER_AFFILIATION: i32 = -1;
const MAX_DIST: f32 = f32::MAX;
/// Components larger than this threshold are sub-partitioned
/// to avoid O(n^2) batch construction cost.
const LARGE_COMPONENT_THRESHOLD: usize = 200;

// ─── Union-Find ───────────────────────────────────────────────────────────────
#[cfg(not(feature = "indicatif"))]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

#[cfg(not(feature = "indicatif"))]
impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

// ─── CohesiveClusterManager ──────────────────────────────────────────────────

/// Algorithm 6: ConstructCohesiveClusters (parallel version).
///
/// Phase 1: Parallel bidirectional edge detection (rayon)
/// Phase 2: Union-Find clustering with candidate set filter
/// Phase 3: Parallel cluster construction & refinement per-component (rayon)
/// Phase 4: Parallel-compute + sequential-apply consolidation
///
/// Generic over `N` (vector dimension) to directly reference `InmemDataset<f32, N>`
/// for distance computation — same code path as DiskANN.
pub struct CohesiveClusterManager<'a, const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    dataset: &'a InmemDataset<f32, N>,
    graph: Arc<CsrGraph>,
    candidate_sets: Arc<Vec<HashSet<u32>>>,
    cluster_index: u32,
    pub cohesive_clusters: RefCell<HashMap<u32, ClusterPointManager>>,
    max_cluster_points_size: usize,
    pub point_affiliation: ArcArray1<i32>,
    critical_minimum_rate: f32,
}

impl<'a, const N: usize> CohesiveClusterManager<'a, N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    pub fn new(
        size: u32,
        dataset: &'a InmemDataset<f32, N>,
        graph: Arc<CsrGraph>,
        candidate_sets: Arc<Vec<HashSet<u32>>>,
        max_cluster_points_size: usize,
        critical_minimum_rate: f32,
    ) -> Self {
        Self {
            dataset,
            graph,
            candidate_sets,
            cluster_index: 0,
            cohesive_clusters: RefCell::new(HashMap::new()),
            max_cluster_points_size,
            point_affiliation: Array1::from_elem(size as usize, -1).to_shared(),
            critical_minimum_rate,
        }
    }

    pub fn point_affiliation(&self, point: u32) -> Option<u32> {
        assert!(
            (point as usize) < self.point_affiliation.len(),
            "Invalid point idx: {point}"
        );
        if self.point_affiliation[point as usize] == INVALID_CLUSTER_AFFILIATION {
            None
        } else {
            Some(self.point_affiliation[point as usize] as u32)
        }
    }

    /// Parallel clustering with pre-computed bidirectional neighbors (skips Phase 1).
    #[cfg(not(feature = "indicatif"))]
    pub fn construct_cohesive_clusters_with_bidir(
        &mut self,
        bidir_neighbors: Arc<Vec<Vec<u32>>>,
    ) -> ANNResult<()> {
        log::info!("  Phase 1 (bidir edges): skipped (pre-computed)");
        self.construct_cohesive_clusters_inner(&bidir_neighbors)
    }

    /// Parallel clustering: 4 phases.
    #[cfg(not(feature = "indicatif"))]
    pub fn construct_cohesive_clusters(&mut self) -> ANNResult<()> {
        let num_nodes = self.graph.num_nodes();

        // ── Phase 1: Parallel bidirectional edge detection ──
        let t0 = std::time::Instant::now();
        let graph_ref = &self.graph;
        let bidir_neighbors: Vec<Vec<u32>> = (0..num_nodes)
            .into_par_iter()
            .map(|node| {
                graph_ref.neighbors(node)
                    .iter()
                    .filter(|&&nbr| graph_ref.contains_edge(nbr, node as u32))
                    .copied()
                    .collect()
            })
            .collect();
        log::info!(
            "  Phase 1 (bidir edges): {:.3}s",
            t0.elapsed().as_secs_f32()
        );
        self.construct_cohesive_clusters_inner(&bidir_neighbors)
    }

    #[cfg(not(feature = "indicatif"))]
    fn construct_cohesive_clusters_inner(
        &mut self,
        bidir_neighbors: &[Vec<u32>],
    ) -> ANNResult<()> {
        let num_nodes = self.graph.num_nodes();

        // ── Phase 2: Union-Find with candidate set filter ──
        let t1 = std::time::Instant::now();
        let mut uf = UnionFind::new(num_nodes);
        for (node, nbrs) in bidir_neighbors.iter().enumerate() {
            let node_u32 = node as u32;
            for &nbr in nbrs {
                // Only union if there's a candidate set relationship
                let has_cs_relationship = self.candidate_sets[node].contains(&nbr)
                    || self.candidate_sets[nbr as usize].contains(&node_u32);
                if has_cs_relationship {
                    uf.union(node, nbr as usize);
                }
            }
        }

        // Extract connected components
        let mut components: HashMap<usize, Vec<u32>> = HashMap::new();
        for i in 0..num_nodes {
            let root = uf.find(i);
            components.entry(root).or_default().push(i as u32);
        }

        // Separate small vs large components
        let mut small_components: Vec<Vec<u32>> = Vec::new();
        let mut large_components: Vec<Vec<u32>> = Vec::new();
        let mut singleton_count = 0usize;

        for (_, members) in components {
            if members.len() <= 1 {
                singleton_count += 1;
                // Single-node components → handled in consolidation
            } else if members.len() <= LARGE_COMPONENT_THRESHOLD {
                small_components.push(members);
            } else {
                large_components.push(members);
            }
        }

        log::info!(
            "  Phase 2 (union-find): {:.3}s — {} small components, {} large, {} singletons",
            t1.elapsed().as_secs_f32(),
            small_components.len(),
            large_components.len(),
            singleton_count,
        );

        // ── Phase 3: Parallel cluster construction & refinement ──
        let t2 = std::time::Instant::now();
        let graph = self.graph.clone();
        let candidate_sets = self.candidate_sets.clone();
        let max_size = self.max_cluster_points_size;
        let critical_rate = self.critical_minimum_rate;

        // Phase 3a: Small components → parallel batch construction
        let small_results: Vec<Vec<ClusterPointManager>> = small_components
            .into_par_iter()
            .map(|members| {
                build_and_refine_component(
                    &graph,
                    &candidate_sets,
                    &members,
                    max_size,
                    critical_rate,
                )
            })
            .collect();

        // Phase 3b: Large components → parallel sub-partition + build
        let large_results: Vec<Vec<ClusterPointManager>> = large_components
            .into_par_iter()
            .map(|members| {
                build_large_component(
                    &graph,
                    &candidate_sets,
                    &bidir_neighbors,
                    &members,
                    max_size,
                    critical_rate,
                )
            })
            .collect();

        // Assign cluster IDs and update point_affiliation
        let mut cluster_map: HashMap<u32, ClusterPointManager> = HashMap::new();
        let mut cluster_id: u32 = 0;

        for cluster_group in small_results.into_iter().chain(large_results.into_iter()) {
            for mut cpm in cluster_group {
                if cpm.size() == 0 {
                    continue;
                }
                cpm.set_id(cluster_id);
                for &pid in cpm.cluster_point().keys() {
                    self.point_affiliation[pid as usize] = cluster_id as i32;
                }
                cluster_map.insert(cluster_id, cpm);
                cluster_id += 1;
            }
        }

        self.cohesive_clusters = RefCell::new(cluster_map);
        self.cluster_index = cluster_id;
        log::info!(
            "  Phase 3 (build clusters): {:.3}s — {} clusters formed",
            t2.elapsed().as_secs_f32(),
            cluster_id,
        );

        // ── Phase 4: Parallel consolidation of unaffiliated points ──
        let t3 = std::time::Instant::now();
        self.parallel_consolidate_unaffiliated();
        log::info!(
            "  Phase 4 (consolidation): {:.3}s — {} total clusters",
            t3.elapsed().as_secs_f32(),
            self.cluster_index,
        );

        Ok(())
    }

    /// Compute best cluster assignments in parallel, then apply sequentially.
    #[cfg(not(feature = "indicatif"))]
    fn parallel_consolidate_unaffiliated(&mut self) {
        let n = self.point_affiliation.len();

        // Snapshot cluster centroids and sizes for parallel read access
        let cluster_snapshot: HashMap<u32, (u32, usize)> = {
            let clusters = self.cohesive_clusters.get_mut();
            clusters
                .iter()
                .filter_map(|(&k, v)| v.centroid().map(|c| (k, (c, v.size()))))
                .collect()
        };

        let graph_ref = &self.graph;
        let dataset_ref = self.dataset;
        let pa_ref = &self.point_affiliation;
        let max_size = self.max_cluster_points_size;

        // Collect unaffiliated point indices
        let unaffiliated: Vec<usize> = (0..n)
            .filter(|&idx| pa_ref[idx] == INVALID_CLUSTER_AFFILIATION)
            .collect();

        // Parallel: compute best cluster for each unaffiliated point
        let assignments: Vec<(usize, Option<u32>)> = unaffiliated
            .par_iter()
            .map(|&idx| {
                let mut min_dist = MAX_DIST;
                let mut best_cluster: Option<u32> = None;

                let neighbors = graph_ref.neighbors(idx);
                for &neighbor in neighbors {
                    let aff = pa_ref[neighbor as usize];
                    if aff == INVALID_CLUSTER_AFFILIATION {
                        continue;
                    }
                    let neighbor_aff = aff as u32;

                    if let Some(&(centroid, size)) = cluster_snapshot.get(&neighbor_aff) {
                        if size >= max_size {
                            continue;
                        }
                        let dist = dataset_ref.get_distance(centroid, idx as u32, Metric::L2).unwrap_or(MAX_DIST);
                        if dist < min_dist {
                            min_dist = dist;
                            best_cluster = Some(neighbor_aff);
                        }
                    }
                }

                (idx, best_cluster)
            })
            .collect();

        // Sequential: apply assignments (must be sequential due to cluster mutation)
        for (idx, assignment) in assignments {
            match assignment {
                Some(cluster_id) => {
                    let clusters = self.cohesive_clusters.get_mut();
                    // Re-check size constraint (may have filled up from earlier assignments)
                    if let Some(cpm) = clusters.get(&cluster_id) {
                        if cpm.cluster_point().len() >= self.max_cluster_points_size {
                            // Cluster is full, create a new singleton
                            self.create_singleton_cluster(idx as u32);
                            continue;
                        }
                    }
                    clusters
                        .get_mut(&cluster_id)
                        .expect("cluster disappeared")
                        .append(idx as u32, false);
                    self.point_affiliation[idx] = cluster_id as i32;
                }
                None => {
                    self.create_singleton_cluster(idx as u32);
                }
            }
        }
    }

    #[cfg(not(feature = "indicatif"))]
    fn create_singleton_cluster(&mut self, point: u32) {
        let cluster_id = self.cluster_index;
        let mut cpm = ClusterPointManager::new(
            cluster_id as i32,
            self.graph.clone(),
            self.candidate_sets.clone(),
            self.max_cluster_points_size,
            self.critical_minimum_rate,
        );
        cpm.append(point, true);
        self.cohesive_clusters.get_mut().insert(cluster_id, cpm);
        self.point_affiliation[point as usize] = cluster_id as i32;
        self.cluster_index += 1;
    }

    /// Main clustering algorithm: iterate bidirectional edges and build clusters.
    #[cfg(feature = "indicatif")]
    pub fn construct_cohesive_clusters(&mut self) -> ANNResult<()> {
        let num_nodes = self.graph.num_nodes();
        let style = crate::utils::NODES_PROGRESS_STYLE.clone();

        for origin in (0..num_nodes as u32)
            .progress_with_style(style)
            .with_finish(AndLeave)
        {
            let neighbors = self.graph.neighbors(origin as usize);

            for &neighbor in neighbors {
                let is_bidirectional = self.graph.contains_edge(neighbor, origin);

                if !is_bidirectional {
                    continue;
                }

                let origin_affiliation = self.point_affiliation(origin);
                let neighbor_affiliation = self.point_affiliation(neighbor);

                match (origin_affiliation, neighbor_affiliation) {
                    (Some(o_aff), Some(n_aff)) => {
                        if o_aff == n_aff {
                            continue;
                        }

                        let mut origin_cpm = self.get_cluster_point_manager(o_aff);
                        let neighbor_cpm = self.get_cluster_point_manager(n_aff);

                        if !origin_cpm.should_clusters_be_merged(&neighbor_cpm) {
                            self.cohesive_clusters.get_mut().insert(o_aff, origin_cpm);
                            self.cohesive_clusters.get_mut().insert(n_aff, neighbor_cpm);
                            continue;
                        }

                        for &neighbor_point_id in neighbor_cpm.cluster_point().keys() {
                            self.point_affiliation[neighbor_point_id as usize] = o_aff as i32;
                        }

                        if let Some((popped_list, sub_cpms)) =
                            origin_cpm.append_new_cluster(neighbor_cpm, true)
                        {
                            self.handle_overflow(o_aff, origin_cpm, popped_list, sub_cpms);
                        } else {
                            self.cohesive_clusters.get_mut().insert(o_aff, origin_cpm);
                        }
                    }
                    (Some(o_aff), None) => {
                        let origin_cpm = self.get_cluster_point_manager(o_aff);
                        if !origin_cpm.should_affiliated_into_cluster(neighbor) {
                            self.cohesive_clusters.get_mut().insert(o_aff, origin_cpm);
                            continue;
                        }
                        self.cluster_append_new_point(o_aff, origin_cpm, neighbor);
                    }
                    (None, Some(n_aff)) => {
                        let neighbor_cpm = self.get_cluster_point_manager(n_aff);
                        if !neighbor_cpm.should_affiliated_into_cluster(origin) {
                            self.cohesive_clusters.get_mut().insert(n_aff, neighbor_cpm);
                            continue;
                        }
                        self.cluster_append_new_point(n_aff, neighbor_cpm, origin);
                    }
                    (None, None) => {
                        let mut cpm = ClusterPointManager::new(
                            self.cluster_index as i32,
                            self.graph.clone(),
                            self.candidate_sets.clone(),
                            self.max_cluster_points_size,
                            self.critical_minimum_rate,
                        );
                        cpm.append(origin, true);
                        cpm.append(neighbor, true);
                        self.cohesive_clusters
                            .get_mut()
                            .insert(self.cluster_index, cpm);
                        self.point_affiliation[origin as usize] = self.cluster_index as i32;
                        self.point_affiliation[neighbor as usize] = self.cluster_index as i32;
                        self.cluster_index += 1;
                    }
                }
            }
        }

        self.consolidate_unaffiliated_points();
        Ok(())
    }

    #[cfg(feature = "indicatif")]
    fn get_cluster_point_manager(&mut self, affiliation: u32) -> ClusterPointManager {
        self.cohesive_clusters
            .get_mut()
            .remove(&affiliation)
            .unwrap_or_else(|| panic!("Cluster {affiliation} not found."))
    }

    #[cfg(feature = "indicatif")]
    fn cluster_append_new_point(
        &mut self,
        affiliation: u32,
        mut cpm: ClusterPointManager,
        point: u32,
    ) {
        self.point_affiliation[point as usize] = affiliation as i32;
        if let Some((popped_list, sub_cpms)) = cpm.append(point, true) {
            self.handle_overflow(affiliation, cpm, popped_list, sub_cpms);
        } else {
            self.cohesive_clusters.get_mut().insert(affiliation, cpm);
        }
    }

    #[cfg(feature = "indicatif")]
    fn handle_overflow(
        &mut self,
        cur_affiliation: u32,
        origin_cpm: ClusterPointManager,
        popped_list: Vec<u32>,
        sub_cpms: Vec<ClusterPointManager>,
    ) {
        for popped_point in popped_list {
            self.point_affiliation[popped_point as usize] = INVALID_CLUSTER_AFFILIATION;
        }

        if !sub_cpms.is_empty() {
            for mut cpm in sub_cpms {
                cpm.set_id(self.cluster_index);
                for &point_id in cpm.cluster_point().keys() {
                    self.point_affiliation[point_id as usize] = self.cluster_index as i32;
                }
                self.cohesive_clusters
                    .get_mut()
                    .insert(self.cluster_index, cpm);
                self.cluster_index += 1;
            }
        } else if !origin_cpm.cluster_point().is_empty() {
            self.cohesive_clusters
                .get_mut()
                .insert(cur_affiliation, origin_cpm);
        }
    }

    /// Assign remaining unaffiliated points to nearest cluster centroid.
    #[cfg(feature = "indicatif")]
    fn consolidate_unaffiliated_points(&mut self) {
        let size = self.point_affiliation.len();

        for idx in 0..size {
            if self.point_affiliation[idx] != INVALID_CLUSTER_AFFILIATION {
                continue;
            }

            let mut min_dist = MAX_DIST;
            let mut best_cluster: Option<u32> = None;

            let neighbors = self.graph.neighbors(idx);

            for &neighbor in neighbors {
                if let Some(neighbor_affiliation) = self.point_affiliation(neighbor) {
                    let cluster = self
                        .cohesive_clusters
                        .get_mut()
                        .get(&neighbor_affiliation)
                        .unwrap_or_else(|| panic!("Invalid cluster index {neighbor_affiliation}."));

                    if cluster.cluster_point().len() >= self.max_cluster_points_size {
                        continue;
                    }

                    let centroid = cluster.centroid().unwrap_or_else(|| {
                        panic!("Cluster {neighbor_affiliation}'s centroid is none.")
                    });

                    let dist = self.dataset.get_distance(centroid, idx as u32, Metric::L2).unwrap_or(MAX_DIST);

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = Some(neighbor_affiliation);
                    }
                }
            }

            match best_cluster {
                Some(cluster) => {
                    self.cohesive_clusters
                        .get_mut()
                        .get_mut(&cluster)
                        .expect("cluster is null")
                        .append(idx as u32, false);
                    self.point_affiliation[idx] = cluster as i32;
                }
                None => {
                    let cluster_id = self.cluster_index;
                    let mut cpm = ClusterPointManager::new(
                        cluster_id as i32,
                        self.graph.clone(),
                        self.candidate_sets.clone(),
                        self.max_cluster_points_size,
                        self.critical_minimum_rate,
                    );
                    cpm.append(idx as u32, true);
                    self.cohesive_clusters.get_mut().insert(cluster_id, cpm);
                    self.point_affiliation[idx] = cluster_id as i32;
                    self.cluster_index += 1;
                }
            }
        }
    }
}

// ─── Component processing helpers ────────────────────────────────────────────

/// Build and refine a small component using batch ClusterPointManager construction.
#[cfg(not(feature = "indicatif"))]
fn build_and_refine_component(
    graph: &Arc<CsrGraph>,
    candidate_sets: &Arc<Vec<HashSet<u32>>>,
    members: &[u32],
    max_size: usize,
    critical_rate: f32,
) -> Vec<ClusterPointManager> {
    let mut cpm = ClusterPointManager::from_members(
        0, // ID assigned later
        graph.clone(),
        candidate_sets.clone(),
        max_size,
        critical_rate,
        members,
    );

    if cpm.size() <= max_size {
        return vec![cpm];
    }

    // Overflow: pop excess + potentially split into sub-clusters
    match cpm.enforce_size_constraint() {
        Some((_popped, sub_managers)) => {
            if sub_managers.is_empty() {
                if cpm.size() > 0 { vec![cpm] } else { vec![] }
            } else {
                sub_managers
            }
        }
        None => vec![cpm],
    }
}

/// Handle large components by greedy sub-partitioning, then batch-build each partition.
/// This avoids O(n^2) pairwise cost for huge components.
#[cfg(not(feature = "indicatif"))]
fn build_large_component(
    graph: &Arc<CsrGraph>,
    candidate_sets: &Arc<Vec<HashSet<u32>>>,
    bidir_neighbors: &[Vec<u32>],
    members: &[u32],
    max_size: usize,
    critical_rate: f32,
) -> Vec<ClusterPointManager> {
    let member_set: HashSet<u32> = members.iter().copied().collect();
    let mut affiliated: HashSet<u32> = HashSet::new();
    let mut sub_clusters: Vec<Vec<u32>> = Vec::new();

    // Greedy BFS-based sub-partitioning within the component
    for &seed in members {
        if affiliated.contains(&seed) {
            continue;
        }

        let mut cluster = vec![seed];
        affiliated.insert(seed);

        // BFS grow from seed using bidirectional + candidate set edges
        let mut frontier: Vec<u32> = vec![seed];
        while cluster.len() < max_size && !frontier.is_empty() {
            let mut next_frontier = Vec::new();
            for &node in &frontier {
                for &nbr in &bidir_neighbors[node as usize] {
                    if cluster.len() >= max_size {
                        break;
                    }
                    if affiliated.contains(&nbr) || !member_set.contains(&nbr) {
                        continue;
                    }
                    // Candidate set check: seed's candidate set should relate to nbr
                    let seed_cs = &candidate_sets[seed as usize];
                    let nbr_cs = &candidate_sets[nbr as usize];
                    if seed_cs.contains(&nbr) || nbr_cs.contains(&seed) {
                        cluster.push(nbr);
                        affiliated.insert(nbr);
                        next_frontier.push(nbr);
                    }
                }
            }
            frontier = next_frontier;
        }

        if cluster.len() >= 2 {
            sub_clusters.push(cluster);
        }
    }

    // Build each sub-cluster in parallel
    sub_clusters
        .into_par_iter()
        .flat_map(|members| {
            build_and_refine_component(graph, candidate_sets, &members, max_size, critical_rate)
        })
        .collect()
}
