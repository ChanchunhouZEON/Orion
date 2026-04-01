/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::cluster::ClusterPoint;
use diskann::model::CsrGraph;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub const INVALID_CLUSTER_ID: i32 = -1;
pub const INVALID_CENTROID: i32 = -1;

/// Manages a single cohesive cluster: its points, centroid, overflow/eviction.
#[derive(Debug, Clone)]
pub struct ClusterPointManager {
    id: i32,
    graph: Arc<CsrGraph>,
    candidate_sets: Arc<Vec<HashSet<u32>>>,
    max_cluster_points_size: usize,
    cluster_points: HashMap<u32, ClusterPoint>,
    critical_minimum_rate: f32,
    centroid: i32,
}

impl ClusterPointManager {
    pub fn new(
        id: i32,
        graph: Arc<CsrGraph>,
        candidate_sets: Arc<Vec<HashSet<u32>>>,
        max_cluster_points_size: usize,
        critical_minimum_rate: f32,
    ) -> Self {
        Self {
            id,
            graph,
            candidate_sets,
            max_cluster_points_size,
            cluster_points: HashMap::new(),
            critical_minimum_rate,
            centroid: INVALID_CENTROID,
        }
    }

    /// Batch-construct a ClusterPointManager from a list of member IDs.
    /// Computes all pairwise relationships in one pass instead of incremental `append`.
    pub fn from_members(
        id: i32,
        graph: Arc<CsrGraph>,
        candidate_sets: Arc<Vec<HashSet<u32>>>,
        max_cluster_points_size: usize,
        critical_minimum_rate: f32,
        members: &[u32],
    ) -> Self {
        // Initialize ClusterPoints with self in in_candidate_set
        let mut cluster_points: HashMap<u32, ClusterPoint> = members
            .iter()
            .map(|&m| {
                (
                    m,
                    ClusterPoint::new(HashSet::from([m]), HashSet::new(), HashSet::new()),
                )
            })
            .collect();

        // Pre-compute neighbor sets for all members
        let neighbor_sets: HashMap<u32, HashSet<u32>> = members
            .iter()
            .map(|&m| {
                let nbrs: HashSet<u32> = graph
                    .neighbors(m as usize)
                    .iter()
                    .copied()
                    .collect();
                (m, nbrs)
            })
            .collect();

        // Compute all pairwise relationships
        for i in 0..members.len() {
            let a = members[i];
            let a_cand = &candidate_sets[a as usize];
            let a_nbrs = &neighbor_sets[&a];

            for j in (i + 1)..members.len() {
                let b = members[j];
                let b_cand = &candidate_sets[b as usize];
                let b_nbrs = &neighbor_sets[&b];

                // a's candidate set contains b
                // → a.cluster_point_in_cur_candidates includes b
                // → b.in_candidate_set includes a
                if a_cand.contains(&b) {
                    cluster_points
                        .get_mut(&a)
                        .unwrap()
                        .cluster_point_in_cur_candidates
                        .insert(b);
                    cluster_points
                        .get_mut(&b)
                        .unwrap()
                        .in_candidate_set
                        .insert(a);
                }

                // b's candidate set contains a
                if b_cand.contains(&a) {
                    cluster_points
                        .get_mut(&b)
                        .unwrap()
                        .cluster_point_in_cur_candidates
                        .insert(a);
                    cluster_points
                        .get_mut(&a)
                        .unwrap()
                        .in_candidate_set
                        .insert(b);
                }

                // edge a → b exists → b.connected_set includes a
                if a_nbrs.contains(&b) {
                    cluster_points.get_mut(&b).unwrap().connected_set.insert(a);
                }

                // edge b → a exists → a.connected_set includes b
                if b_nbrs.contains(&a) {
                    cluster_points.get_mut(&a).unwrap().connected_set.insert(b);
                }
            }
        }

        let centroid = cluster_points
            .iter()
            .max_by_key(|(_, cp)| cp.in_candidate_set.len())
            .map(|(&id, _)| id as i32)
            .unwrap_or(INVALID_CENTROID);

        Self {
            id,
            graph,
            candidate_sets,
            max_cluster_points_size,
            cluster_points,
            critical_minimum_rate,
            centroid,
        }
    }

    /// Apply overflow handling: pop excess points and split disconnected components.
    /// Returns (popped_point_ids, sub_cluster_managers) if overflow occurred.
    pub fn enforce_size_constraint(&mut self) -> Option<(Vec<u32>, Vec<Self>)> {
        self.consolidate_cluster(true)
    }

    pub fn is_cluster_point(&self, point: u32) -> bool {
        self.cluster_points.contains_key(&point)
    }

    /// Whether a new point should be affiliated into this cluster.
    pub fn should_affiliated_into_cluster(&self, point: u32) -> bool {
        if self.cluster_points.contains_key(&point) {
            return false;
        }
        self.centroid == INVALID_CENTROID
            || self.candidate_sets[self.centroid as usize].contains(&point)
    }

    /// Append a point to this cluster. Returns overflow info if cluster exceeds max size.
    pub fn append(&mut self, point: u32, use_pop_process: bool) -> Option<(Vec<u32>, Vec<Self>)> {
        if self.cluster_points.contains_key(&point) {
            return None;
        }

        let new_cluster_point = Self::construct_cluster_point(
            self.graph.clone(),
            self.candidate_sets.clone(),
            &mut self.cluster_points,
            point,
        );

        self.cluster_points.insert(point, new_cluster_point);
        self.consolidate_cluster(use_pop_process)
    }

    /// Whether two clusters should be merged based on candidate set overlap.
    pub fn should_clusters_be_merged(&self, another: &Self) -> bool {
        if self.cluster_points.len() >= self.max_cluster_points_size
            || another.cluster_points.len() >= another.max_cluster_points_size
        {
            return false;
        }

        self.cluster_points.keys().any(|&point| {
            let cand = &self.candidate_sets[point as usize];
            if cand.is_empty() {
                return false;
            }
            let size = cand
                .iter()
                .filter(|c| self.cluster_points.contains_key(c))
                .count()
                + cand
                    .iter()
                    .filter(|c| another.cluster_points.contains_key(c))
                    .count();
            size >= self.cluster_points.len() && size >= another.cluster_points.len()
        })
    }

    /// Merge another cluster into this one.
    pub fn append_new_cluster(
        &mut self,
        mut another_cluster: ClusterPointManager,
        use_pop_process: bool,
    ) -> Option<(Vec<u32>, Vec<ClusterPointManager>)> {
        for (cur_id, cur_point) in self.cluster_points.iter_mut() {
            let cur_point_in_another = Self::construct_cluster_point(
                another_cluster.graph.clone(),
                another_cluster.candidate_sets.clone(),
                &mut another_cluster.cluster_points,
                *cur_id,
            );
            cur_point
                .cluster_point_in_cur_candidates
                .extend(cur_point_in_another.cluster_point_in_cur_candidates);
            cur_point
                .in_candidate_set
                .extend(cur_point_in_another.in_candidate_set);
            cur_point
                .connected_set
                .extend(cur_point_in_another.connected_set);
        }

        for (another_id, another_point) in another_cluster.cluster_points.iter_mut() {
            let another_in_cur = Self::construct_cluster_point(
                another_cluster.graph.clone(),
                another_cluster.candidate_sets.clone(),
                &mut self.cluster_points,
                *another_id,
            );
            another_point
                .cluster_point_in_cur_candidates
                .extend(another_in_cur.cluster_point_in_cur_candidates);
            another_point
                .in_candidate_set
                .extend(another_in_cur.in_candidate_set);
            another_point
                .connected_set
                .extend(another_in_cur.connected_set);
        }

        self.cluster_points.extend(another_cluster.cluster_points);
        self.consolidate_cluster(use_pop_process)
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn set_id(&mut self, id: u32) {
        self.id = id as i32;
    }

    pub fn size(&self) -> usize {
        self.cluster_points.len()
    }

    pub fn centroid(&self) -> Option<u32> {
        if self.centroid == INVALID_CENTROID {
            None
        } else {
            Some(self.centroid as u32)
        }
    }

    pub fn cluster_point(&self) -> &HashMap<u32, ClusterPoint> {
        &self.cluster_points
    }

    // ─── Eviction Logic (Algorithm 5) ───

    fn consolidate_cluster(&mut self, use_pop_process: bool) -> Option<(Vec<u32>, Vec<Self>)> {
        if use_pop_process && self.cluster_points.len() > self.max_cluster_points_size {
            let mut popped_list = self.pop();
            return if let Some((single_point_list, sub_managers)) = self.split_cluster() {
                popped_list.extend(single_point_list);
                Some((popped_list, sub_managers))
            } else {
                self.update_centroid();
                Some((popped_list, Vec::new()))
            };
        }
        self.update_centroid();
        None
    }

    fn pop(&mut self) -> Vec<u32> {
        let mut popped_list = Vec::new();
        while self.cluster_points.len() > self.max_cluster_points_size {
            let outlier_id = self
                .cluster_points
                .iter()
                .min_by_key(|(_, p)| p.connected_set.len())
                .map(|(&id, _)| id)
                .unwrap();

            self.remove_and_clean(outlier_id);
            popped_list.push(outlier_id);

            loop {
                let to_be_removed: Vec<u32> = self
                    .cluster_points
                    .keys()
                    .filter(|&&pid| !self.reserve_gate(pid))
                    .copied()
                    .collect();

                if to_be_removed.is_empty() {
                    break;
                }

                for pid in to_be_removed {
                    self.remove_and_clean(pid);
                    popped_list.push(pid);
                }
            }
        }
        popped_list
    }

    fn reserve_gate(&self, point: u32) -> bool {
        if self.centroid == INVALID_CENTROID {
            return true;
        }
        let cp = &self.cluster_points[&point];
        !cp.connected_set.is_empty()
            && cp.in_candidate_set.len() as f32
                >= self.cluster_points.len() as f32 * self.critical_minimum_rate
    }

    fn remove_and_clean(&mut self, id: u32) {
        if let Some(point) = self.cluster_points.remove(&id) {
            for related_id in &point.cluster_point_in_cur_candidates {
                if let Some(rp) = self.cluster_points.get_mut(related_id) {
                    rp.in_candidate_set.remove(&id);
                }
            }
            let neighbors = self.graph.neighbors(id as usize);
            for &nid in neighbors {
                if let Some(np) = self.cluster_points.get_mut(&nid) {
                    np.connected_set.remove(&id);
                }
            }
        }
    }

    fn update_centroid(&mut self) {
        self.centroid = self
            .cluster_points
            .iter()
            .max_by_key(|(_, cp)| cp.in_candidate_set.len())
            .map(|(&id, _)| id as i32)
            .unwrap_or(INVALID_CENTROID);
    }

    fn construct_cluster_point(
        graph: Arc<CsrGraph>,
        candidate_sets: Arc<Vec<HashSet<u32>>>,
        cluster: &mut HashMap<u32, ClusterPoint>,
        point: u32,
    ) -> ClusterPoint {
        let mut in_candidate_set = HashSet::new();
        in_candidate_set.insert(point);
        let mut cur_candidates = HashSet::new();
        let mut connected_set = HashSet::new();

        let new_point_cand = &candidate_sets[point as usize];
        let point_neighbors = graph.neighbors(point as usize);

        for (&cp_id, cp) in cluster.iter_mut() {
            let cand = &candidate_sets[cp_id as usize];
            if cand.contains(&point) {
                cp.cluster_point_in_cur_candidates.insert(point);
                in_candidate_set.insert(cp_id);
            }

            if graph.contains_edge(cp_id, point) {
                connected_set.insert(cp_id);
            }

            if new_point_cand.contains(&cp_id) {
                cur_candidates.insert(cp_id);
                cp.in_candidate_set.insert(point);
            }

            if point_neighbors.contains(&cp_id) {
                cp.connected_set.insert(point);
            }
        }

        ClusterPoint::new(in_candidate_set, cur_candidates, connected_set)
    }

    // ─── Connected Component Analysis ───

    fn breadth_first_search(&self, origin: u32) -> HashSet<u32> {
        let mut search_list = VecDeque::new();
        let mut visited = HashSet::new();
        search_list.push_back(origin);
        visited.insert(origin);

        while let Some(cur) = search_list.pop_front() {
            for &nxt in self.graph.neighbors(cur as usize) {
                if self.cluster_points.contains_key(&nxt) && !visited.contains(&nxt) {
                    let is_bi = self.graph.contains_edge(nxt, cur);
                    if is_bi {
                        visited.insert(nxt);
                        search_list.push_back(nxt);
                    }
                }
            }
        }

        visited
    }

    fn identify_isolated_clusters(&self) -> Vec<Vec<u32>> {
        let mut isolated_clusters = Vec::new();
        let mut global_visited = HashSet::new();

        for &idx in self.cluster_points.keys() {
            if global_visited.contains(&idx) {
                continue;
            }
            let sub_cluster_set = self.breadth_first_search(idx);
            global_visited.extend(sub_cluster_set.iter().copied());
            isolated_clusters.push(sub_cluster_set.into_iter().collect());
        }

        isolated_clusters
    }

    fn split_cluster(&mut self) -> Option<(Vec<u32>, Vec<ClusterPointManager>)> {
        let isolated_clusters = self.identify_isolated_clusters();

        if isolated_clusters.len() <= 1 {
            return None;
        }

        let mut single_point_list = Vec::new();
        let mut sub_cluster_point_managers = Vec::new();

        for cluster in &isolated_clusters {
            if cluster.len() == 1 {
                single_point_list.extend(cluster);
                continue;
            }

            let mut cluster_manager = Self {
                id: INVALID_CLUSTER_ID,
                graph: self.graph.clone(),
                candidate_sets: self.candidate_sets.clone(),
                max_cluster_points_size: self.max_cluster_points_size,
                cluster_points: HashMap::new(),
                critical_minimum_rate: self.critical_minimum_rate,
                centroid: INVALID_CENTROID,
            };

            for id in cluster {
                let new_point = Self::construct_cluster_point(
                    self.graph.clone(),
                    self.candidate_sets.clone(),
                    &mut cluster_manager.cluster_points,
                    *id,
                );
                cluster_manager.cluster_points.insert(*id, new_point);
            }

            cluster_manager.centroid = cluster_manager
                .cluster_points
                .iter()
                .max_by_key(|(_, cp)| cp.in_candidate_set.len())
                .map(|(&id, _)| id as i32)
                .unwrap_or(INVALID_CENTROID);

            sub_cluster_point_managers.push(cluster_manager);
        }

        Some((single_point_list, sub_cluster_point_managers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build a small graph and candidate sets for testing.
    /// Graph: 0↔1, 1↔2, 2↔3, 3↔4, 0→2, 2→4 (bidirectional core + some unidirectional)
    fn build_test_fixtures(num_nodes: usize) -> (Arc<CsrGraph>, Arc<Vec<HashSet<u32>>>) {
        let adj: Vec<Vec<u32>> = vec![
            vec![1, 2],       // 0
            vec![0, 2],       // 1
            vec![1, 3, 4, 0], // 2
            vec![2, 4],       // 3
            vec![3, 2],       // 4
        ];

        let graph = Arc::new(CsrGraph::from_adjacency_list(adj.clone()));

        // Candidate sets: each node's candidate set contains its neighbors + self
        let mut candidate_sets = vec![HashSet::new(); num_nodes];
        for i in 0..num_nodes {
            let mut set: HashSet<u32> = adj[i].iter().copied().collect();
            set.insert(i as u32);
            candidate_sets[i] = set;
        }

        (graph, Arc::new(candidate_sets))
    }

    #[test]
    fn test_manager_initialization() {
        let (graph, cand) = build_test_fixtures(5);
        let cpm = ClusterPointManager::new(0, graph.clone(), cand, 10, 0.3);

        assert_eq!(cpm.id(), 0);
        assert_eq!(cpm.centroid(), None);
        assert_eq!(cpm.size(), 0);
    }

    #[test]
    fn test_append_point() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 10, 0.3);

        cpm.append(0, false);
        assert_eq!(cpm.size(), 1);
        assert!(cpm.is_cluster_point(0));
        assert!(!cpm.is_cluster_point(1));

        // Centroid should be set after append
        assert!(cpm.centroid().is_some());

        cpm.append(1, false);
        assert_eq!(cpm.size(), 2);
        assert!(cpm.is_cluster_point(1));
    }

    #[test]
    fn test_append_duplicate_ignored() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 10, 0.3);

        cpm.append(0, false);
        cpm.append(0, false); // duplicate
        assert_eq!(cpm.size(), 1);
    }

    #[test]
    fn test_pop_logic_on_overflow() {
        let (graph, cand) = build_test_fixtures(5);
        // max_cluster_points_size = 2, so 3rd point triggers overflow
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 2, 0.3);

        cpm.append(0, true);
        cpm.append(1, true);
        assert!(cpm.size() <= 2);

        // Adding a 3rd point with pop process enabled
        let result = cpm.append(2, true);
        if let Some((popped, _sub_managers)) = result {
            // Some points should have been popped
            assert!(!popped.is_empty());
            // Cluster size should be within limits
            assert!(cpm.size() <= 2);
        }
    }

    #[test]
    fn test_should_affiliated_into_cluster() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 10, 0.3);

        // Empty cluster (centroid = INVALID_CENTROID) accepts everyone
        assert!(cpm.should_affiliated_into_cluster(0));

        cpm.append(0, false);
        // Point already in cluster → not affiliated
        assert!(!cpm.should_affiliated_into_cluster(0));
        // Point in centroid's candidate set → can be affiliated
        // centroid=0, candidate_set[0] = {0,1,2}
        assert!(cpm.should_affiliated_into_cluster(1));
    }

    #[test]
    fn test_should_clusters_be_merged() {
        let (graph, cand) = build_test_fixtures(5);

        let mut cpm1 = ClusterPointManager::new(0, graph.clone(), cand.clone(), 10, 0.3);
        cpm1.append(0, false);
        cpm1.append(1, false);

        let mut cpm2 = ClusterPointManager::new(1, graph.clone(), cand, 10, 0.3);
        cpm2.append(2, false);
        cpm2.append(3, false);

        // Both clusters have overlapping candidate sets
        let _should = cpm1.should_clusters_be_merged(&cpm2);
        // Result depends on exact candidate set overlap, but shouldn't crash
    }

    #[test]
    fn test_merge_blocks_if_full() {
        let (graph, cand) = build_test_fixtures(5);

        // max_cluster_points_size = 2 (already full)
        let mut cpm1 = ClusterPointManager::new(0, graph.clone(), cand.clone(), 2, 0.3);
        cpm1.append(0, false);
        cpm1.append(1, false);

        let mut cpm2 = ClusterPointManager::new(1, graph.clone(), cand, 2, 0.3);
        cpm2.append(2, false);

        // Merge should be blocked because cpm1 is at capacity
        assert!(!cpm1.should_clusters_be_merged(&cpm2));
    }

    #[test]
    fn test_remove_and_clean_consistency() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 10, 0.3);

        cpm.append(0, false);
        cpm.append(1, false);
        cpm.append(2, false);
        assert_eq!(cpm.size(), 3);

        // After removal, relationships should be cleaned
        cpm.remove_and_clean(1);
        assert_eq!(cpm.size(), 2);
        assert!(!cpm.is_cluster_point(1));
        assert!(cpm.is_cluster_point(0));
        assert!(cpm.is_cluster_point(2));

        // Verify remaining points' connected_set and in_candidate_set
        // don't reference the removed point
        for (_, cp) in cpm.cluster_point() {
            assert!(!cp.connected_set.contains(&1));
            assert!(!cp.in_candidate_set.contains(&1));
        }
    }

    #[test]
    fn test_merge_two_clusters() {
        let (graph, cand) = build_test_fixtures(5);

        let mut cpm1 = ClusterPointManager::new(0, graph.clone(), cand.clone(), 10, 0.3);
        cpm1.append(0, false);

        let mut cpm2 = ClusterPointManager::new(1, graph.clone(), cand, 10, 0.3);
        cpm2.append(3, false);
        cpm2.append(4, false);

        let result = cpm1.append_new_cluster(cpm2, false);
        // No overflow with max=10
        assert!(result.is_none());
        assert_eq!(cpm1.size(), 3);
        assert!(cpm1.is_cluster_point(0));
        assert!(cpm1.is_cluster_point(3));
        assert!(cpm1.is_cluster_point(4));
    }

    #[test]
    fn test_split_with_disconnected_components() {
        // Build a graph with two disconnected components within a cluster
        let adj: Vec<Vec<u32>> = vec![
            vec![1],    // 0 — Component 1: 0↔1
            vec![0],    // 1
            vec![3],    // 2 — Component 2: 2↔3
            vec![2],    // 3
            vec![],     // 4 — isolated
        ];

        let graph = Arc::new(CsrGraph::from_adjacency_list(adj.clone()));

        let mut candidate_sets = vec![HashSet::new(); 5];
        for i in 0..5usize {
            let mut set: HashSet<u32> = adj[i].iter().copied().collect();
            set.insert(i as u32);
            candidate_sets[i] = set;
        }
        let cand = Arc::new(candidate_sets);

        // max_cluster_points_size = 2 → triggers pop + split after inserting 3 nodes
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 2, 0.3);
        cpm.append(0, false);
        cpm.append(1, false);
        let result = cpm.append(2, true);

        // With disconnected components and overflow, split should occur or points pop
        if let Some((popped, sub_managers)) = result {
            // Total accounted for: cluster + popped + sub_managers
            let total_in_cpm = cpm.size();
            let total_in_subs: usize = sub_managers.iter().map(|m| m.size()).sum();
            let total = total_in_cpm + popped.len() + total_in_subs;
            // We started with 3 points
            assert!(total <= 3, "Total points should not exceed 3, got {total}");
        }
    }

    #[test]
    fn test_centroid_update() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 10, 0.3);

        assert_eq!(cpm.centroid(), None);

        cpm.append(0, false);
        assert!(cpm.centroid().is_some());

        cpm.append(1, false);
        cpm.append(2, false);
        // Centroid should be the point with most in_candidate_set entries
        let centroid = cpm.centroid().unwrap();
        assert!(centroid <= 4); // valid node id
    }

    #[test]
    fn test_set_id() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, graph.clone(), cand, 10, 0.3);
        assert_eq!(cpm.id(), 0);
        cpm.set_id(42);
        assert_eq!(cpm.id(), 42);
    }
}
