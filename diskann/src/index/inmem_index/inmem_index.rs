/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![allow(unused_imports)]

use std::cmp;
use std::collections::{HashMap, HashSet as StdHashSet};
use std::sync::{Mutex, RwLock};
use std::time::Duration;

use hashbrown::HashSet;
use hashbrown::hash_set::Entry::*;
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use crate::common::{ANNError, ANNResult};
use crate::index::ANNInmemIndex;
use crate::instrumentation::IndexLogger;
use crate::model::graph::AdjacencyList;
use crate::model::{
    ArcConcurrentBoxedQueue, InMemQueryScratch, InMemoryGraph, IndexConfiguration, InmemDataset,
    Neighbor, ScratchStoreManager, Vertex,
};

use crate::utils::file_util::{file_exists, load_metadata_from_file};
use crate::utils::rayon_util::execute_with_rayon;
use crate::utils::{Timer, set_rayon_num_threads};

/// In-memory Index
pub struct InmemIndex<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Dataset
    pub dataset: InmemDataset<T, N>,

    /// Graph
    pub final_graph: InMemoryGraph,

    /// Index configuration
    pub configuration: IndexConfiguration,

    /// Start point of the search.
    pub start: u32,

    /// Max observed out degree
    pub max_observed_degree: u32,

    /// Number of active points i.e. existing in the graph
    pub num_active_pts: usize,

    /// query scratch queue.
    query_scratch_queue: ArcConcurrentBoxedQueue<InMemQueryScratch<T, N>>,

    pub delete_set: RwLock<HashSet<u32>>,

    /// mmap-backed storage for `(location, pruned_id)` pairs per anchor.
    /// OS page cache manages residency; lock-free per-slot atomic append.
    /// Not tracked by Rust allocator — heap peak stays low.
    #[cfg(feature = "orion")]
    pub candidate_sets: Option<crate::model::MmapAnchorSlab>,
}

impl<T, const N: usize> InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Create Index obj based on configuration
    pub fn new(mut config: IndexConfiguration) -> ANNResult<Self> {
        if config.max_points == 0 {
            config.max_points = 1;
        }

        let total_internal_points = config.max_points + config.num_frozen_pts;

        if config.use_pq_dist {
            todo!("PQ is not supported now");
        }

        let start = config.max_points.try_into()?;

        let query_scratch_queue = ArcConcurrentBoxedQueue::<InMemQueryScratch<T, N>>::new();
        let delete_set = RwLock::new(HashSet::<u32>::new());

        #[cfg(feature = "orion")]
        let candidate_sets = if config.index_write_parameter.compute_candidate_sets {
            let max_pairs = config.index_write_parameter.max_degree as usize * 2;
            let tmp_path = std::env::temp_dir()
                .join(format!("orion_mmap_{}.bin", std::process::id()));
            Some(
                crate::model::MmapAnchorSlab::new(&tmp_path, total_internal_points, max_pairs)
                    .expect("MmapAnchorSlab creation failed"),
            )
        } else {
            None
        };

        Ok(Self {
            dataset: InmemDataset::<T, N>::new(total_internal_points, config.growth_potential)?,
            final_graph: InMemoryGraph::new(
                total_internal_points,
                config.index_write_parameter.max_degree,
            ),
            configuration: config,
            start,
            max_observed_degree: 0,
            num_active_pts: 0,
            query_scratch_queue,
            delete_set,
            #[cfg(feature = "orion")]
            candidate_sets,
        })
    }

    /// Get distance between two vertices.
    pub fn get_distance(&self, id1: u32, id2: u32) -> ANNResult<f32> {
        self.dataset
            .get_distance(id1, id2, self.configuration.dist_metric)
    }

    fn build_with_data_populated(&mut self) -> ANNResult<()> {
        log::info!(
            "Starting index build with {} points...",
            self.num_active_pts
        );

        if self.num_active_pts < 1 {
            return Err(ANNError::log_index_error(
                "Error: Trying to build an index with 0 points.".to_string(),
            ));
        }

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        self.link()?;

        self.print_stats()?;

        Ok(())
    }

    fn link(&mut self) -> ANNResult<()> {
        let mut visit_order =
            Vec::with_capacity(self.num_active_pts + self.configuration.num_frozen_pts);
        for i in 0..self.num_active_pts {
            visit_order.push(i as u32);
        }

        for frozen in self.configuration.max_points
            ..(self.configuration.max_points + self.configuration.num_frozen_pts)
        {
            visit_order.push(frozen as u32);
        }

        if self.configuration.num_frozen_pts > 0 {
            self.start = self.configuration.max_points as u32;
        } else {
            self.start = self.dataset.calculate_medoid_point_id()?;
        }

        // Establish an edge to the entry before parallel insertion. Otherwise
        // the entry can be processed while the graph is still empty, leaving
        // only itself in the candidate pool and producing an empty prune list.
        let seed = visit_order.iter().copied().find(|&id| id != self.start);
        if let Some(seed) = seed {
            self.insert_vertex_id(seed)?;
        }

        let timer = Timer::new();

        let range = visit_order.len();
        let logger = IndexLogger::new(range);

        execute_with_rayon(
            0..range,
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                if Some(visit_order[idx]) != seed && seed.is_some() {
                    self.insert_vertex_id(visit_order[idx])?;
                }
                logger.vertex_processed()?;

                Ok(())
            },
        )?;

        self.cleanup_graph(&visit_order)?;

        if self.num_active_pts > 0 {
            log::info!("{}", timer.elapsed_seconds_for_step("Link time: "));
        }

        Ok(())
    }

    fn insert_vertex_id(&self, vertex_id: u32) -> ANNResult<()> {
        let mut scratch_manager =
            ScratchStoreManager::new(self.query_scratch_queue.clone(), Duration::from_millis(10))?;
        let scratch = scratch_manager.scratch_space().ok_or_else(|| {
            ANNError::log_index_error(
                "ScratchStoreManager doesn't have InMemQueryScratch instance available".to_string(),
            )
        })?;

        #[cfg(feature = "orion")]
        {
            let (new_neighbors, dists) =
                self.search_for_point_and_prune_with_dists(scratch, vertex_id)?;
            let mut guard = self.final_graph.write_vertex_and_neighbors(vertex_id)?;
            guard.set_neighbors_sorted(new_neighbors, dists);
            assert!(guard.size() <= self.configuration.index_write_parameter.max_degree as usize);
            drop(guard);
        }
        #[cfg(not(feature = "orion"))]
        {
            let new_neighbors = self.search_for_point_and_prune(scratch, vertex_id)?;
            self.update_vertex_with_neighbors(vertex_id, new_neighbors)?;
        }
        self.update_neighbors_of_vertex(vertex_id, scratch)?;

        Ok(())
    }

    fn update_neighbors_of_vertex(
        &self,
        vertex_id: u32,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> Result<(), ANNError> {
        let vertex = self.final_graph.read_vertex_and_neighbors(vertex_id)?;
        assert!(vertex.size() <= self.configuration.index_write_parameter.max_degree as usize);
        self.inter_insert(
            vertex_id,
            vertex.get_neighbors(),
            self.configuration.index_write_parameter.max_degree,
            scratch,
        )?;
        Ok(())
    }

    #[cfg(not(feature = "orion"))]
    fn update_vertex_with_neighbors(
        &self,
        vertex_id: u32,
        new_neighbors: AdjacencyList,
    ) -> Result<(), ANNError> {
        let vertex = &mut self.final_graph.write_vertex_and_neighbors(vertex_id)?;
        vertex.set_neighbors(new_neighbors);
        assert!(vertex.size() <= self.configuration.index_write_parameter.max_degree as usize);
        Ok(())
    }

    #[cfg(not(feature = "orion"))]
    fn search_for_point_and_prune(
        &self,
        scratch: &mut InMemQueryScratch<T, N>,
        vertex_id: u32,
    ) -> ANNResult<AdjacencyList> {
        let mut pruned_list =
            AdjacencyList::for_range(self.configuration.index_write_parameter.max_degree as usize);
        let vertex = self.dataset.get_vertex(vertex_id)?;
        let mut visited_nodes = self.search_for_point(&vertex, scratch)?;

        self.prune_neighbors(vertex_id, &mut visited_nodes, &mut pruned_list, scratch)?;

        if pruned_list.is_empty() {
            return Err(ANNError::log_index_error(
                "pruned_list is empty.".to_string(),
            ));
        }

        if self.final_graph.size()
            != self.configuration.max_points + self.configuration.num_frozen_pts
        {
            return Err(ANNError::log_index_error(format!(
                "final_graph has {} vertices instead of {}",
                self.final_graph.size(),
                self.configuration.max_points + self.configuration.num_frozen_pts,
            )));
        }

        Ok(pruned_list)
    }

    /// Like `search_for_point_and_prune` but also returns per-neighbor distances.
    /// The pruned_list preserves occlude_list order (distance ascending from vertex_id).
    #[cfg(feature = "orion")]
    fn search_for_point_and_prune_with_dists(
        &self,
        scratch: &mut InMemQueryScratch<T, N>,
        vertex_id: u32,
    ) -> ANNResult<(AdjacencyList, Vec<f32>)> {
        let mut pruned_list =
            AdjacencyList::for_range(self.configuration.index_write_parameter.max_degree as usize);
        let vertex = self.dataset.get_vertex(vertex_id)?;
        let mut visited_nodes = self.search_for_point(&vertex, scratch)?;

        self.prune_neighbors(vertex_id, &mut visited_nodes, &mut pruned_list, scratch)?;

        if pruned_list.is_empty() {
            return Err(ANNError::log_index_error(
                "pruned_list is empty.".to_string(),
            ));
        }

        // Extract distances from the pool for pruned neighbors.
        let dists: Vec<f32> = pruned_list
            .iter()
            .map(|&id| {
                visited_nodes
                    .iter()
                    .find(|nb| nb.id == id)
                    .map(|nb| nb.distance)
                    .unwrap_or(f32::MAX)
            })
            .collect();

        Ok((pruned_list, dists))
    }

    /// ADSampling variant of [`Self::search`]. See
    /// [`Self::search_with_l_override_adsampling`] for preconditions.
    pub fn search_adsampling(
        &self,
        query: &Vertex<T, N>,
        k_value: usize,
        l_value: u32,
        ads_epsilon: f32,
        indices: &mut [u32],
    ) -> ANNResult<u32> {
        if k_value > l_value as usize {
            return Err(ANNError::log_index_error(format!(
                "Set L: {} to a value of at least K: {}",
                l_value, k_value
            )));
        }

        let mut scratch_manager =
            ScratchStoreManager::new(self.query_scratch_queue.clone(), Duration::from_millis(10))?;
        let scratch = scratch_manager.scratch_space().ok_or_else(|| {
            ANNError::log_index_error(
                "ScratchStoreManager doesn't have InMemQueryScratch instance available".to_string(),
            )
        })?;
        if l_value > scratch.candidate_size {
            scratch.resize_for_new_candidate_size(l_value);
        }

        let cmp =
            self.search_with_l_override_adsampling(query, scratch, l_value as usize, ads_epsilon)?;
        let mut pos = 0;

        for i in 0..scratch.best_candidates.size() {
            if scratch.best_candidates[i].id < self.configuration.max_points as u32 {
                if let Ok(delete_set_guard) = self.delete_set.read() {
                    if !delete_set_guard.contains(&scratch.best_candidates[i].id) {
                        indices[pos] = scratch.best_candidates[i].id;
                        pos += 1;
                    }
                } else {
                    return Err(ANNError::log_lock_poison_error(
                        "failed to acquire the lock for delete_set.".to_string(),
                    ));
                }
            }
            if pos == k_value {
                break;
            }
        }
        Ok(cmp)
    }

    fn search(
        &self,
        query: &Vertex<T, N>,
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> ANNResult<u32> {
        if k_value > l_value as usize {
            return Err(ANNError::log_index_error(format!(
                "Set L: {} to a value of at least K: {}",
                l_value, k_value
            )));
        }

        let mut scratch_manager =
            ScratchStoreManager::new(self.query_scratch_queue.clone(), Duration::from_millis(10))?;

        let scratch = scratch_manager.scratch_space().ok_or_else(|| {
            ANNError::log_index_error(
                "ScratchStoreManager doesn't have InMemQueryScratch instance available".to_string(),
            )
        })?;

        if l_value > scratch.candidate_size {
            println!(
                "Attempting to expand query scratch_space. Was created with Lsize: {} but search L is: {}",
                scratch.candidate_size, l_value
            );
            scratch.resize_for_new_candidate_size(l_value);
            println!(
                "Resize completed. New scratch size is: {}",
                scratch.candidate_size
            );
        }

        let cmp = self.search_with_l_override(query, scratch, l_value as usize)?;
        let mut pos = 0;

        for i in 0..scratch.best_candidates.size() {
            if scratch.best_candidates[i].id < self.configuration.max_points as u32 {
                if let Ok(delete_set_guard) = self.delete_set.read() {
                    if !delete_set_guard.contains(&scratch.best_candidates[i].id) {
                        indices[pos] = scratch.best_candidates[i].id;
                        pos += 1;
                    }
                } else {
                    return Err(ANNError::log_lock_poison_error(
                        "failed to acquire the lock for delete_set.".to_string(),
                    ));
                }
            }

            if pos == k_value {
                break;
            }
        }

        if pos < k_value {
            eprintln!(
                "Found fewer than K elements for query! Found: {} but K: {}",
                pos, k_value
            );
        }

        Ok(cmp)
    }

    fn cleanup_graph(&mut self, visit_order: &Vec<u32>) -> ANNResult<()> {
        if self.num_active_pts > 0 {
            log::info!("Starting final cleanup..");
        }

        execute_with_rayon(
            0..visit_order.len(),
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                let vertex_id = visit_order[idx];
                let num_nbrs = self.get_neighbor_count(vertex_id)?;

                if num_nbrs <= self.configuration.index_write_parameter.max_degree as usize {
                    return Ok(());
                }

                let mut scratch_manager = ScratchStoreManager::new(
                    self.query_scratch_queue.clone(),
                    Duration::from_millis(10),
                )?;
                let scratch = scratch_manager.scratch_space().ok_or_else(|| {
                    ANNError::log_index_error(
                        "ScratchStoreManager doesn't have InMemQueryScratch instance available"
                            .to_string(),
                    )
                })?;

                let mut dummy_pool = self.get_neighbors_for_vertex(vertex_id)?;

                let mut new_out_neighbors = AdjacencyList::for_range(
                    self.configuration.index_write_parameter.max_degree as usize,
                );
                self.prune_neighbors(vertex_id, &mut dummy_pool, &mut new_out_neighbors, scratch)?;

                #[cfg(feature = "orion")]
                {
                    let dists: Vec<f32> = new_out_neighbors
                        .iter()
                        .map(|&id| {
                            dummy_pool
                                .iter()
                                .find(|nb| nb.id == id)
                                .map(|nb| nb.distance)
                                .unwrap_or(f32::MAX)
                        })
                        .collect();
                    self.final_graph
                        .write_vertex_and_neighbors(vertex_id)?
                        .set_neighbors_sorted(new_out_neighbors, dists);
                }
                #[cfg(not(feature = "orion"))]
                {
                    self.final_graph
                        .write_vertex_and_neighbors(vertex_id)?
                        .set_neighbors(new_out_neighbors);
                }

                Ok(())
            },
        )
    }

    fn get_neighbors_for_vertex(&self, vertex_id: u32) -> ANNResult<Vec<Neighbor>> {
        let binding = self.final_graph.read_vertex_and_neighbors(vertex_id)?;
        let neighbors = binding.get_neighbors();
        let dummy_pool = self.get_unique_neighbors(neighbors, vertex_id)?;

        Ok(dummy_pool)
    }

    pub fn get_unique_neighbors(
        &self,
        neighbors: &Vec<u32>,
        vertex_id: u32,
    ) -> Result<Vec<Neighbor>, ANNError> {
        let vertex = self.dataset.get_vertex(vertex_id)?;

        let len = neighbors.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        self.dataset.prefetch_vector(neighbors[0]);

        let mut dummy_visited: HashSet<u32> = HashSet::with_capacity(len);
        let mut dummy_pool: Vec<Neighbor> = Vec::with_capacity(len);

        for current in neighbors.windows(2) {
            self.dataset.prefetch_vector(current[1]);
            let current = current[0];

            self.insert_neighbor_if_unique(
                &mut dummy_visited,
                current,
                vertex_id,
                &vertex,
                &mut dummy_pool,
            )?;
        }

        #[allow(clippy::unwrap_used)]
        self.insert_neighbor_if_unique(
            &mut dummy_visited,
            *neighbors.last().unwrap(),
            vertex_id,
            &vertex,
            &mut dummy_pool,
        )?;

        Ok(dummy_pool)
    }

    fn insert_neighbor_if_unique(
        &self,
        dummy_visited: &mut HashSet<u32>,
        current: u32,
        vertex_id: u32,
        vertex: &Vertex<'_, T, N>,
        dummy_pool: &mut Vec<Neighbor>,
    ) -> Result<(), ANNError> {
        if current != vertex_id {
            if let Vacant(entry) = dummy_visited.entry(current) {
                let cur_nbr_vertex = self.dataset.get_vertex(current)?;
                let dist = vertex.compare(&cur_nbr_vertex, self.configuration.dist_metric);
                dummy_pool.push(Neighbor::new(current, dist));
                entry.insert();
            }
        }

        Ok(())
    }

    fn get_neighbor_count(&self, vertex_id: u32) -> ANNResult<usize> {
        let num_nbrs = self
            .final_graph
            .read_vertex_and_neighbors(vertex_id)?
            .size();
        Ok(num_nbrs)
    }

    fn soft_delete_vertex(&self, vertex_id_to_delete: u32) -> ANNResult<()> {
        if vertex_id_to_delete as usize > self.num_active_pts {
            return Err(ANNError::log_index_error(format!(
                "vertex_id_to_delete: {} is greater than the number of active points in the graph: {}",
                vertex_id_to_delete, self.num_active_pts
            )));
        }

        let mut delete_set_guard = match self.delete_set.write() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(ANNError::log_index_error(format!(
                    "Failed to acquire delete_set lock, cannot delete vertex {}",
                    vertex_id_to_delete
                )));
            }
        };

        delete_set_guard.insert(vertex_id_to_delete);
        Ok(())
    }

    fn initialize_query_scratch(
        &mut self,
        num_threads: u32,
        search_candidate_size: u32,
    ) -> ANNResult<()> {
        self.query_scratch_queue.reserve(num_threads as usize)?;
        for _ in 0..num_threads {
            let scratch = Box::new(InMemQueryScratch::<T, N>::new(
                search_candidate_size,
                &self.configuration.index_write_parameter,
                false,
            )?);

            self.query_scratch_queue.push(scratch)?;
        }

        Ok(())
    }

    fn print_stats(&mut self) -> ANNResult<()> {
        let mut max = 0;
        let mut min = usize::MAX;
        let mut total = 0;
        let mut cnt = 0;

        for i in 0..self.num_active_pts {
            let vertex_id = i.try_into()?;
            let pool_size = self
                .final_graph
                .read_vertex_and_neighbors(vertex_id)?
                .size();
            max = cmp::max(max, pool_size);
            min = cmp::min(min, pool_size);
            total += pool_size;
            if pool_size < 2 {
                cnt += 1;
            }
        }

        log::info!(
            "Index built with degree: max: {} avg: {} min: {} count(deg<2): {}",
            max,
            (total as f32) / ((self.num_active_pts + self.configuration.num_frozen_pts) as f32),
            min,
            cnt
        );

        match self.delete_set.read() {
            Ok(guard) => {
                log::info!(
                    "Number of soft deleted vertices {}, soft deleted percentage: {}",
                    guard.len(),
                    (guard.len() as f32)
                        / ((self.num_active_pts + self.configuration.num_frozen_pts) as f32),
                );
            }
            Err(_) => {
                return Err(ANNError::log_lock_poison_error(
                    "Failed to acquire delete_set lock, cannot get the number of deleted vertices"
                        .to_string(),
                ));
            }
        };

        self.max_observed_degree = cmp::max(max as u32, self.max_observed_degree);

        Ok(())
    }

    /// Augment candidate sets with graph structure to produce final per-node candidate sets.
    ///
    /// For each node `origin`:
    /// 1. Add `origin` itself
    /// 2. For each bidirectional neighbor `n` (where `n` was the pruning anchor):
    ///    add `n` + all pruned_ids from `candidate_sets[n]` where location == `origin`
    ///
    /// Uses flat sorted `Vec<(location, pruned_id)>` per anchor for cache-friendly lookup.
    #[cfg(feature = "orion")]
    pub fn extract_candidate_sets(&mut self) -> ANNResult<Vec<StdHashSet<u32>>> {
        let slab = self.candidate_sets.as_mut().ok_or_else(|| {
            ANNError::log_index_error("Candidate sets have not been built".to_string())
        })?;
        slab.sort_all();

        // Pre-materialise bidirectional neighbors.
        let mut offsets = Vec::with_capacity(self.num_active_pts + 1);
        let mut neighbors: Vec<u32> =
            Vec::with_capacity(self.max_observed_degree as usize * self.num_active_pts);
        offsets.push(0u32);
        for i in 0..self.num_active_pts as u32 {
            if let Ok(v) = self.final_graph.read_vertex_and_neighbors(i) {
                neighbors.extend(v.get_neighbors());
            }
            offsets.push(neighbors.len() as u32);
        }
        let bidir_nbrs: Vec<Vec<u32>> = (0..self.num_active_pts as u32)
            .into_par_iter()
            .map(|node| -> ANNResult<Vec<u32>> {
                let start = offsets[node as usize] as usize;
                let end = offsets[(node + 1) as usize] as usize;
                let nbrs = &neighbors[start..end];
                let mut bidir = Vec::with_capacity(nbrs.len());
                for &nbr in nbrs.iter() {
                    if self.final_graph.contains_edge(nbr, node)? {
                        bidir.push(nbr);
                    }
                }
                Ok(bidir)
            })
            .collect::<ANNResult<Vec<Vec<u32>>>>()?;

        // Augment candidate sets via BPM page reads.
        let mut result: Vec<StdHashSet<u32>> = (0..self.num_active_pts)
            .map(|_| StdHashSet::new())
            .collect();
        for origin in 0..self.num_active_pts {
            let origin_u32 = origin as u32;
            let candidates = &mut result[origin];
            candidates.insert(origin_u32);
            for &neighbor in bidir_nbrs[origin].iter() {
                candidates.insert(neighbor);
                let pairs = slab.pairs(neighbor as usize);
                let start = pairs.partition_point(|&(loc, _)| loc < origin_u32);
                let mut i = start;
                while i < pairs.len() && pairs[i].0 == origin_u32 {
                    candidates.insert(pairs[i].1);
                    i += 1;
                }
            }
        }

        Ok(result)
    }

    /// Extract the final graph as a HashMap for external consumers.
    pub fn extract_graph(&self) -> HashMap<u32, Vec<u32>> {
        self.final_graph.to_hashmap()
    }

    /// Build a lock-free `CsrGraph` from `final_graph` AND compute candidate
    /// sets in one pass.  Reads each RwLock once to flatten neighbors into CSR,
    /// then uses the lock-free CSR for bidirectional edge detection and
    /// candidate set augmentation.
    ///
    /// The caller should `drop` the index after this call to free the
    /// `InmemDataset` and scratch queues.
    #[cfg(feature = "orion")]
    /// Extract InMemoryGraph + candidate_sets in a memory-efficient pipeline:
    ///
    /// 1. Sort each node's neighbors by distance in-place (needs dataset).
    /// 2. Enrich slab via single-pass prune (needs dataset).
    /// 3. Build candidate_sets from slab (location=node, anchor=key_nbr).
    /// 4. Free dataset + slab, return InMemoryGraph.
    ///
    /// Extract pre-partitioned node data for PhasedGraph construction.
    ///
    /// Returns `Vec<(local, remote, extra)>` per node, partitioned by the
    /// **top-X% rule** (matches the C++ side's `staged_export.h` v3):
    ///
    /// For each node `i`, combine its graph neighbours `G[i]` (with their
    /// original distances from the build) and the captured extras pool
    /// (with distances from the slab) into one list, sort by distance asc,
    /// and threshold at `cutoff = round(0.60 × len(G[i]))`:
    ///
    /// - top-cutoff ∩ G[i]:        → `local`   (close graph edges)
    /// - top-cutoff ∩ not-in-G[i]: → `extra`   (close pruned candidates)
    /// - past-cutoff ∩ G[i]:       → `remote`  (far graph edges)
    /// - past-cutoff ∩ not-in-G[i]: → discarded
    ///
    /// Replaces the legacy bidir-based local classification — local is now
    /// purely distance-driven. `dists` from `into_neighbors_and_dists()`
    /// supplies the original build-time distances directly.
    ///
    /// `max_extra > 0` applies a post-partition hard cap on the extras
    /// zone (kept for backward compatibility with the slab capacity).
    /// Pass 0 for "no cap"; cutoff alone bounds extras to ≤ 0.60 × deg.
    ///
    /// Steps:
    /// 0. Free dataset (no longer needed after build).
    /// 1. Snapshot InMemoryGraph → flat Vec (lock-free), drop InMemoryGraph.
    /// 2. Sort slab for sequential access during the per-node scan.
    /// 3. Per-node: build combined list, sort, partition by top-X% rule.
    #[cfg(feature = "orion")]
    pub fn extract_graph_and_candidates(
        &mut self,
        max_extra: usize,
    ) -> ANNResult<Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>> {
        use crate::utils::mem_usage;
        use std::collections::HashSet;

        // Top-X% local-cutoff fraction (60%, matches `staged_export.h`'s
        // `local_pct` default). Hardcoded for now; promote to a parameter
        // if/when callers need to tune it.
        const LOCAL_PCT: f64 = 0.60;

        let num_pts = self.num_active_pts;
        log::info!(
            "    extract entry (max_extra_cap={}, local_pct={:.0}%):  mem={}",
            max_extra,
            LOCAL_PCT * 100.0,
            mem_usage()
        );

        // P0: Free dataset — not needed after build.
        self.dataset = InmemDataset::new(0, 1.0)?;
        log::info!("    P0 (dataset freed):   mem={}", mem_usage());

        // P1: Snapshot InMemoryGraph → flat Vec, drop InMemoryGraph.
        // Each node: (neighbor_ids, neighbor_dists). Lock-free after this.
        let t1 = std::time::Instant::now();
        let max_degree = self.configuration.index_write_parameter.max_degree;
        let old_graph = std::mem::replace(&mut self.final_graph, InMemoryGraph::new(0, max_degree));
        let snapshot: Vec<(Vec<u32>, Vec<f32>)> = old_graph
            .final_graph
            .into_iter()
            .map(|lock| {
                let vn = lock.into_inner().unwrap_or_else(|e| e.into_inner());
                vn.into_neighbors_and_dists()
            })
            .collect();
        log::info!(
            "    P1 (snapshot):        {:.3}s  mem={}",
            t1.elapsed().as_secs_f32(),
            mem_usage()
        );

        // P2: Sort slab for sequential access.
        let t2 = std::time::Instant::now();
        if let Some(ref mut slab) = self.candidate_sets {
            let (at_cap, overflow, max_count) = slab.truncation_stats();
            log::info!(
                "    slab stats: max_pairs={}, at_cap={}/{}, overflow={}, max_count={}",
                slab.max_pairs(),
                at_cap,
                num_pts,
                overflow,
                max_count
            );
            slab.sort_all();
            slab.advise_sequential();
        }
        log::info!(
            "    P2 (sort slab):       {:.3}s  mem={}",
            t2.elapsed().as_secs_f32(),
            mem_usage()
        );

        // P3: Top-X% partition. For each node, combine G[i] (with original
        // build-time distances) and the captured extras pool, sort by dist,
        // split at `cutoff = round(LOCAL_PCT × deg)`. No bidir lookup any
        // more — pure distance-based classification.
        let t3 = std::time::Instant::now();
        let partitions: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = (0..num_pts)
            .into_par_iter()
            .map(|node| {
                let (ref nbrs, ref dists) = snapshot[node];
                let node_u32 = node as u32;
                let degree = nbrs.len();

                if degree == 0 {
                    return (Vec::new(), Vec::new(), Vec::new());
                }

                // O(1) `is in G[i]?` lookup for the extras-vs-G[i] check.
                let nbr_set: HashSet<u32> = nbrs.iter().copied().collect();

                // Combined list: (dist, id, in_g).
                let mut combined: Vec<(f32, u32, bool)> = Vec::with_capacity(degree);

                // Add G[i] neighbours with their build-time distances.
                for (j, &n) in nbrs.iter().enumerate() {
                    let d = dists.get(j).copied().unwrap_or(f32::MAX);
                    combined.push((d, n, true));
                }

                // Add extras from slab: filter self + dups against G[i],
                // dedup by id (slab can contain the same id twice across
                // different anchors).
                if let Some(ref slab) = self.candidate_sets {
                    let pairs = slab.pairs(node);
                    let mut seen_extras: HashSet<u32> = HashSet::with_capacity(pairs.len());
                    for &(dist_bits, pid) in pairs {
                        if pid == node_u32 {
                            continue;
                        }
                        if nbr_set.contains(&pid) {
                            continue;
                        }
                        if !seen_extras.insert(pid) {
                            continue;
                        }
                        combined.push((f32::from_bits(dist_bits), pid, false));
                    }
                }

                combined.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                // cutoff anchored to graph degree (not combined.size()) so
                // |local + extra| stays proportional to graph density,
                // independent of how many extras the slab happened to
                // capture for this node.
                let cutoff = (LOCAL_PCT * degree as f64).round() as usize;

                let mut local: Vec<u32> = Vec::with_capacity(cutoff);
                let mut remote: Vec<u32> = Vec::with_capacity(degree.saturating_sub(cutoff));
                let mut extra: Vec<u32> = Vec::with_capacity(cutoff);

                for (k, &(_, id, in_g)) in combined.iter().enumerate() {
                    if k < cutoff {
                        if in_g {
                            local.push(id);
                        } else {
                            extra.push(id);
                        }
                    } else if in_g {
                        remote.push(id);
                    }
                    // else: past-cutoff non-G[i] candidates are discarded.
                }

                // Hard cap on extras zone (slab can carry many
                // candidates; cap the array length even when the
                // top-X% rule would admit more). `max_extra == 0`
                // means "no extras allowed" — the previous guard
                // `max_extra > 0` accidentally skipped truncation in
                // that case and let the partition extras overflow
                // the cache-line-aligned slab, corrupting the next
                // node's header (ablation no-extra variant panicked
                // in `extra_candidates` reading the corrupted count).
                if extra.len() > max_extra {
                    extra.truncate(max_extra);
                }

                (local, remote, extra)
            })
            .collect();
        log::info!(
            "    P3 (partition):       {:.3}s  mem={}",
            t3.elapsed().as_secs_f32(),
            mem_usage()
        );

        // P4: Free slab + snapshot.
        drop(snapshot);
        self.query_scratch_queue = ArcConcurrentBoxedQueue::new();
        self.candidate_sets = None;
        log::info!("    P4 (cleanup):         mem={}", mem_usage());

        Ok(partitions)
    }

    /// Sort each node's neighbor list by distance in-place.
    /// Must be called while the dataset is still alive.
    #[cfg(feature = "orion")]
    pub fn sort_neighbors_by_distance_in_place(&self) -> ANNResult<()> {
        let metric = self.configuration.dist_metric;
        let num_pts = self.num_active_pts;

        for node in 0..num_pts {
            let node_u32 = node as u32;
            let mut guard = self.final_graph.write_vertex_and_neighbors(node_u32)?;
            let nbrs = guard.get_neighbors();
            let mut with_dist: Vec<(u32, f32)> = nbrs
                .iter()
                .filter_map(|&n| {
                    self.dataset
                        .get_distance(node_u32, n, metric)
                        .ok()
                        .map(|d| (n, d))
                })
                .collect();
            with_dist.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let sorted_ids: Vec<u32> = with_dist.into_iter().map(|(id, _)| id).collect();
            guard.set_neighbors(sorted_ids.into());
        }
        Ok(())
    }

    /// Free the candidate set slab.
    #[cfg(feature = "orion")]
    pub fn drop_candidate_slab(&mut self) {
        self.candidate_sets = None;
    }

    /// Get number of active points.
    pub fn num_active_points(&self) -> usize {
        self.num_active_pts
    }
}

impl<T, const N: usize> ANNInmemIndex<T> for InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
    [T; N]: FullPrecisionDistance<T, N>,
{
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn build(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()> {
        if !file_exists(filename) {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Data file {} does not exist.",
                filename
            )));
        }

        let (file_num_points, file_dim) = load_metadata_from_file(filename)?;
        if file_num_points > self.configuration.max_points {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has {} points,
                but index can support only {} points as specified in configuration.",
                num_points_to_load, file_num_points, self.configuration.max_points
            )));
        }

        if num_points_to_load > file_num_points {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has only {} points.",
                num_points_to_load, file_num_points
            )));
        }

        if file_dim != self.configuration.dim {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} dimension, but file has {} dimension.",
                self.configuration.dim, file_dim
            )));
        }

        if self.configuration.use_pq_dist {
            todo!("PQ is not supported now");
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        self.dataset.build_from_file(filename, num_points_to_load)?;

        println!("Using only first {} from file.", num_points_to_load);

        self.num_active_pts = num_points_to_load;
        self.build_with_data_populated()?;

        Ok(())
    }

    fn insert(&mut self, filename: &str, num_points_to_insert: usize) -> ANNResult<()> {
        if !file_exists(filename) {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Data file {} does not exist.",
                filename
            )));
        }

        let (file_num_points, file_dim) = load_metadata_from_file(filename)?;

        if num_points_to_insert > file_num_points {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has only {} points.",
                num_points_to_insert, file_num_points
            )));
        }

        if file_dim != self.configuration.dim {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} dimension, but file has {} dimension.",
                self.configuration.dim, file_dim
            )));
        }

        if self.configuration.use_pq_dist {
            todo!("PQ is not supported now");
        }

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        self.dataset
            .append_from_file(filename, num_points_to_insert)?;
        self.final_graph.extend(
            num_points_to_insert,
            self.configuration.index_write_parameter.max_degree,
        );

        let previous_last_pt = self.num_active_pts;
        self.num_active_pts += num_points_to_insert;
        self.configuration.max_points += num_points_to_insert;

        println!("Inserting {} vectors from file.", num_points_to_insert);

        let logger = IndexLogger::new(num_points_to_insert);
        let timer = Timer::new();
        execute_with_rayon(
            previous_last_pt..self.num_active_pts,
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                self.insert_vertex_id(idx as u32)?;
                logger.vertex_processed()?;

                Ok(())
            },
        )?;

        let mut visit_order =
            Vec::with_capacity(self.num_active_pts + self.configuration.num_frozen_pts);
        for i in 0..self.num_active_pts {
            visit_order.push(i as u32);
        }

        self.cleanup_graph(&visit_order)?;
        println!("{}", timer.elapsed_seconds_for_step("Insert time: "));

        self.print_stats()?;

        Ok(())
    }

    fn save(&mut self, filename: &str) -> ANNResult<()> {
        let data_file = filename.to_string() + ".data";
        let delete_file = filename.to_string() + ".delete";

        self.save_graph(filename)?;
        self.save_data(data_file.as_str())?;
        self.save_delete_list(delete_file.as_str())?;

        Ok(())
    }

    fn load(&mut self, filename: &str, expected_num_points: usize) -> ANNResult<()> {
        self.num_active_pts = expected_num_points;
        self.dataset
            .build_from_file(&format!("{}.data", filename), expected_num_points)?;

        self.load_graph(filename, expected_num_points)?;
        self.load_delete_list(&format!("{}.delete", filename))?;

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        Ok(())
    }

    fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> ANNResult<u32> {
        let query_vector = Vertex::new(<&[T; N]>::try_from(query)?, 0);
        InmemIndex::search(self, &query_vector, k_value, l_value, indices)
    }

    fn search_adsampling(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        ads_epsilon: f32,
        indices: &mut [u32],
    ) -> ANNResult<u32> {
        let query_vector = Vertex::new(<&[T; N]>::try_from(query)?, 0);
        InmemIndex::search_adsampling(self, &query_vector, k_value, l_value, ads_epsilon, indices)
    }

    fn soft_delete(
        &mut self,
        vertex_ids_to_delete: Vec<u32>,
        num_points_to_delete: usize,
    ) -> ANNResult<()> {
        println!("Deleting {} vectors from file.", num_points_to_delete);

        let logger = IndexLogger::new(num_points_to_delete);
        let timer = Timer::new();

        execute_with_rayon(
            0..num_points_to_delete,
            self.configuration.index_write_parameter.num_threads,
            |idx: usize| {
                self.soft_delete_vertex(vertex_ids_to_delete[idx])?;
                logger.vertex_processed()?;

                Ok(())
            },
        )?;

        println!("{}", timer.elapsed_seconds_for_step("Delete time: "));
        self.print_stats()?;

        Ok(())
    }

    fn save_graph_mmap(&self, path: &str) -> ANNResult<()> {
        self.final_graph.save_graph_mmap(path)
    }

    fn save_graph_mmap_with_vectors(&self, path: &str) -> ANNResult<()> {
        let metric_byte = match self.configuration.dist_metric {
            vector::Metric::L2 => 0u8,
            vector::Metric::Cosine => 1u8,
        };
        self.final_graph.save_graph_mmap_with_data(
            path,
            self.dataset.get_data(),
            self.configuration.dim,
            metric_byte,
        )
    }

    fn start_node(&self) -> u32 {
        self.start
    }

    #[cfg(feature = "orion")]
    fn extract_candidate_sets(&mut self) -> Option<Vec<StdHashSet<u32>>> {
        InmemIndex::extract_candidate_sets(self).ok()
    }

    fn extract_graph(&self) -> HashMap<u32, Vec<u32>> {
        InmemIndex::extract_graph(self)
    }

    fn extract_final_graph(&mut self, _num_points: usize, max_degree: u32) -> InMemoryGraph {
        std::mem::replace(&mut self.final_graph, InMemoryGraph::new(0, max_degree))
    }

    #[cfg(feature = "orion")]
    fn extract_graph_and_candidates(
        &mut self,
        max_extra: usize,
    ) -> ANNResult<Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>> {
        InmemIndex::extract_graph_and_candidates(self, max_extra)
    }

    #[cfg(feature = "orion")]
    fn drop_candidate_slab(&mut self) {
        InmemIndex::drop_candidate_slab(self)
    }

    fn num_active_points(&self) -> usize {
        InmemIndex::num_active_points(self)
    }

    fn build_from_data(&mut self, data: &[T], num_points: usize) -> ANNResult<()> {
        let expected_len = num_points * N;
        if data.len() < expected_len {
            return Err(ANNError::log_index_error(format!(
                "data too short: got {} elements, expected at least {}",
                data.len(),
                expected_len
            )));
        }

        self.dataset.data.memcpy(&data[..expected_len])?;
        self.dataset.num_active_pts = num_points;
        self.num_active_pts = num_points;

        if self.configuration.index_write_parameter.num_threads > 0 {
            crate::utils::set_rayon_num_threads(
                self.configuration.index_write_parameter.num_threads,
            );
        }

        self.build_with_data_populated()?;
        Ok(())
    }
}
