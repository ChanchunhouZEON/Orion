/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::cluster::{ClusterPoint, SortedSmallMap, SortedSmallSet};
use diskann::model::CsrGraph;
use std::collections::VecDeque;

pub const INVALID_CLUSTER_ID: i32 = -1;
pub const INVALID_CENTROID: i32 = -1;

/// Manages a single cohesive cluster: its points, centroid, overflow/eviction.
///
/// All internal collections are bounded by `MAX_CLUSTER_CAP` (16) and use
/// sorted inline storage (`SortedSmallSet` / `SortedSmallMap`) instead of
/// `HashSet` / `HashMap`, eliminating hashing overhead for small clusters.
#[derive(Debug)]
pub struct ClusterPointManager<'a> {
    id: i32,
    graph: &'a CsrGraph,
    candidate_sets: &'a [Vec<u32>],
    max_cluster_points_size: usize,
    cluster_points: SortedSmallMap<ClusterPoint>,
    critical_minimum_rate: f32,
    centroid: i32,
}

impl<'a> ClusterPointManager<'a> {
    pub fn new(
        id: i32,
        graph: &'a CsrGraph,
        candidate_sets: &'a [Vec<u32>],
        max_cluster_points_size: usize,
        critical_minimum_rate: f32,
    ) -> Self {
        Self {
            id,
            graph,
            candidate_sets,
            max_cluster_points_size,
            cluster_points: SortedSmallMap::new(),
            critical_minimum_rate,
            centroid: INVALID_CENTROID,
        }
    }

    /// Batch-construct from a list of member IDs.
    /// Computes all pairwise relationships in one pass.
    pub fn from_members(
        id: i32,
        graph: &'a CsrGraph,
        candidate_sets: &'a [Vec<u32>],
        max_cluster_points_size: usize,
        critical_minimum_rate: f32,
        members: &[u32],
    ) -> Self {
        // Initialize ClusterPoints with self in in_candidate_set.
        let mut cluster_points = SortedSmallMap::<ClusterPoint>::new();
        for &m in members {
            cluster_points.insert(m, ClusterPoint::new_with_self(m));
        }

        // Pre-compute neighbor sets for all members (sorted vec instead of HashSet).
        let neighbor_vecs: Vec<(u32, Vec<u32>)> = members
            .iter()
            .map(|&m| {
                let mut nbrs: Vec<u32> = graph.neighbors(m as usize).to_vec();
                nbrs.sort_unstable();
                (m, nbrs)
            })
            .collect();

        // Compute all pairwise relationships.
        for i in 0..members.len() {
            let a = members[i];
            let a_cand = &candidate_sets[a as usize];
            let a_nbrs = &neighbor_vecs[i].1;

            for j in (i + 1)..members.len() {
                let b = members[j];
                let b_cand = &candidate_sets[b as usize];
                let b_nbrs = &neighbor_vecs[j].1;

                if a_cand.binary_search(&b).is_ok() {
                    cluster_points
                        .get_mut(a)
                        .unwrap()
                        .cluster_point_in_cur_candidates
                        .insert(b);
                    cluster_points.get_mut(b).unwrap().in_candidate_set.insert(a);
                }

                if b_cand.binary_search(&a).is_ok() {
                    cluster_points
                        .get_mut(b)
                        .unwrap()
                        .cluster_point_in_cur_candidates
                        .insert(a);
                    cluster_points.get_mut(a).unwrap().in_candidate_set.insert(b);
                }

                if a_nbrs.binary_search(&b).is_ok() {
                    cluster_points.get_mut(b).unwrap().connected_set.insert(a);
                }

                if b_nbrs.binary_search(&a).is_ok() {
                    cluster_points.get_mut(a).unwrap().connected_set.insert(b);
                }
            }
        }

        let centroid = cluster_points
            .max_by_key(|cp| cp.in_candidate_set.len())
            .map(|(id, _)| id as i32)
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

    pub fn enforce_size_constraint(
        &mut self,
    ) -> Option<(Vec<u32>, Vec<ClusterPointManager<'a>>)> {
        self.consolidate_cluster(true)
    }

    pub fn is_cluster_point(&self, point: u32) -> bool {
        self.cluster_points.contains_key(point)
    }

    pub fn should_affiliated_into_cluster(&self, point: u32) -> bool {
        if self.cluster_points.contains_key(point) {
            return false;
        }
        self.centroid == INVALID_CENTROID
            || self.candidate_sets[self.centroid as usize]
                .binary_search(&point)
                .is_ok()
    }

    pub fn append(
        &mut self,
        point: u32,
        use_pop_process: bool,
    ) -> Option<(Vec<u32>, Vec<ClusterPointManager<'a>>)> {
        if self.cluster_points.contains_key(point) {
            return None;
        }

        let new_cluster_point = Self::construct_cluster_point(
            self.graph,
            self.candidate_sets,
            &mut self.cluster_points,
            point,
        );

        self.cluster_points.insert(point, new_cluster_point);
        self.consolidate_cluster(use_pop_process)
    }

    pub fn should_clusters_be_merged(&self, another: &Self) -> bool {
        if self.cluster_points.len() >= self.max_cluster_points_size
            || another.cluster_points.len() >= another.max_cluster_points_size
        {
            return false;
        }

        self.cluster_points.iter().any(|(point, _)| {
            let cand = &self.candidate_sets[point as usize];
            if cand.is_empty() {
                return false;
            }
            let size = cand
                .iter()
                .filter(|c| self.cluster_points.contains_key(**c))
                .count()
                + cand
                    .iter()
                    .filter(|c| another.cluster_points.contains_key(**c))
                    .count();
            size >= self.cluster_points.len() && size >= another.cluster_points.len()
        })
    }

    pub fn append_new_cluster(
        &mut self,
        mut another_cluster: ClusterPointManager<'a>,
        use_pop_process: bool,
    ) -> Option<(Vec<u32>, Vec<ClusterPointManager<'a>>)> {
        // Update existing points' relationships with another cluster's points.
        let cur_keys: Vec<u32> = self.cluster_points.keys().to_vec();
        for cur_id in cur_keys {
            let cur_point_in_another = Self::construct_cluster_point(
                another_cluster.graph,
                another_cluster.candidate_sets,
                &mut another_cluster.cluster_points,
                cur_id,
            );
            let cp = self.cluster_points.get_mut(cur_id).unwrap();
            cp.cluster_point_in_cur_candidates
                .extend_from(&cur_point_in_another.cluster_point_in_cur_candidates);
            cp.in_candidate_set
                .extend_from(&cur_point_in_another.in_candidate_set);
            cp.connected_set
                .extend_from(&cur_point_in_another.connected_set);
        }

        // Update another cluster's points with this cluster's points.
        let another_keys: Vec<u32> = another_cluster.cluster_points.keys().to_vec();
        for another_id in another_keys {
            let another_in_cur = Self::construct_cluster_point(
                another_cluster.graph,
                another_cluster.candidate_sets,
                &mut self.cluster_points,
                another_id,
            );
            let cp = another_cluster.cluster_points.get_mut(another_id).unwrap();
            cp.cluster_point_in_cur_candidates
                .extend_from(&another_in_cur.cluster_point_in_cur_candidates);
            cp.in_candidate_set
                .extend_from(&another_in_cur.in_candidate_set);
            cp.connected_set
                .extend_from(&another_in_cur.connected_set);
        }

        // Merge another's entries into self.
        for (k, v) in another_cluster.cluster_points.drain() {
            self.cluster_points.insert(k, v);
        }
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

    pub fn cluster_point(&self) -> &SortedSmallMap<ClusterPoint> {
        &self.cluster_points
    }

    // ─── Eviction Logic (Algorithm 5) ───

    fn consolidate_cluster(
        &mut self,
        use_pop_process: bool,
    ) -> Option<(Vec<u32>, Vec<ClusterPointManager<'a>>)> {
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
                .min_by_key(|cp| cp.connected_set.len())
                .map(|(id, _)| id)
                .unwrap();

            self.remove_and_clean(outlier_id);
            popped_list.push(outlier_id);

            loop {
                let to_be_removed: Vec<u32> = self
                    .cluster_points
                    .key_slice()
                    .iter()
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
        let cp = self.cluster_points.get(point).unwrap();
        !cp.connected_set.is_empty()
            && cp.in_candidate_set.len() as f32
                >= self.cluster_points.len() as f32 * self.critical_minimum_rate
    }

    fn remove_and_clean(&mut self, id: u32) {
        if let Some(point) = self.cluster_points.remove(id) {
            for &related_id in point.cluster_point_in_cur_candidates.as_slice() {
                if let Some(rp) = self.cluster_points.get_mut(related_id) {
                    rp.in_candidate_set.remove(id);
                }
            }
            let neighbors = self.graph.neighbors(id as usize);
            for &nid in neighbors {
                if let Some(np) = self.cluster_points.get_mut(nid) {
                    np.connected_set.remove(id);
                }
            }
        }
    }

    fn update_centroid(&mut self) {
        self.centroid = self
            .cluster_points
            .max_by_key(|cp| cp.in_candidate_set.len())
            .map(|(id, _)| id as i32)
            .unwrap_or(INVALID_CENTROID);
    }

    fn construct_cluster_point(
        graph: &CsrGraph,
        candidate_sets: &[Vec<u32>],
        cluster: &mut SortedSmallMap<ClusterPoint>,
        point: u32,
    ) -> ClusterPoint {
        let mut in_candidate_set = SortedSmallSet::with_one(point);
        let mut cur_candidates = SortedSmallSet::new();
        let mut connected_set = SortedSmallSet::new();

        let new_point_cand = &candidate_sets[point as usize];
        let point_neighbors = graph.neighbors(point as usize);

        // Collect keys first to avoid borrow conflict.
        let keys: Vec<u32> = cluster.key_slice().to_vec();
        for cp_id in keys {
            let cand = &candidate_sets[cp_id as usize];
            if cand.binary_search(&point).is_ok() {
                cluster
                    .get_mut(cp_id)
                    .unwrap()
                    .cluster_point_in_cur_candidates
                    .insert(point);
                in_candidate_set.insert(cp_id);
            }

            if graph.contains_edge(cp_id, point) {
                connected_set.insert(cp_id);
            }

            if new_point_cand.binary_search(&cp_id).is_ok() {
                cur_candidates.insert(cp_id);
                cluster
                    .get_mut(cp_id)
                    .unwrap()
                    .in_candidate_set
                    .insert(point);
            }

            if point_neighbors.contains(&cp_id) {
                cluster
                    .get_mut(cp_id)
                    .unwrap()
                    .connected_set
                    .insert(point);
            }
        }

        ClusterPoint::new(in_candidate_set, cur_candidates, connected_set)
    }

    // ─── Connected Component Analysis ───

    fn breadth_first_search(&self, origin: u32) -> SortedSmallSet {
        let mut search_list = VecDeque::new();
        let mut visited = SortedSmallSet::with_one(origin);
        search_list.push_back(origin);

        while let Some(cur) = search_list.pop_front() {
            for &nxt in self.graph.neighbors(cur as usize) {
                if self.cluster_points.contains_key(nxt)
                    && !visited.contains(nxt)
                    && self.graph.contains_edge(nxt, cur)
                {
                    visited.insert(nxt);
                    search_list.push_back(nxt);
                }
            }
        }

        visited
    }

    fn identify_isolated_clusters(&self) -> Vec<Vec<u32>> {
        let mut isolated_clusters = Vec::new();
        let mut global_visited = SortedSmallSet::new();

        for &idx in self.cluster_points.key_slice() {
            if global_visited.contains(idx) {
                continue;
            }
            let sub_cluster_set = self.breadth_first_search(idx);
            for &v in sub_cluster_set.as_slice() {
                global_visited.insert(v);
            }
            isolated_clusters.push(sub_cluster_set.as_slice().to_vec());
        }

        isolated_clusters
    }

    fn split_cluster(&mut self) -> Option<(Vec<u32>, Vec<ClusterPointManager<'a>>)> {
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
                graph: self.graph,
                candidate_sets: self.candidate_sets,
                max_cluster_points_size: self.max_cluster_points_size,
                cluster_points: SortedSmallMap::new(),
                critical_minimum_rate: self.critical_minimum_rate,
                centroid: INVALID_CENTROID,
            };

            for id in cluster {
                let new_point = Self::construct_cluster_point(
                    self.graph,
                    self.candidate_sets,
                    &mut cluster_manager.cluster_points,
                    *id,
                );
                cluster_manager.cluster_points.insert(*id, new_point);
            }

            cluster_manager.centroid = cluster_manager
                .cluster_points
                .max_by_key(|cp| cp.in_candidate_set.len())
                .map(|(id, _)| id as i32)
                .unwrap_or(INVALID_CENTROID);

            sub_cluster_point_managers.push(cluster_manager);
        }

        Some((single_point_list, sub_cluster_point_managers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_fixtures(num_nodes: usize) -> (CsrGraph, Vec<Vec<u32>>) {
        let adj: Vec<Vec<u32>> = vec![
            vec![1, 2],
            vec![0, 2],
            vec![1, 3, 4, 0],
            vec![2, 4],
            vec![3, 2],
        ];
        let graph = CsrGraph::from_adjacency_list(adj.clone(), 10);
        let mut candidate_sets: Vec<Vec<u32>> = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let mut cs: Vec<u32> = adj[i]
                .iter()
                .copied()
                .chain(std::iter::once(i as u32))
                .collect();
            cs.sort_unstable();
            cs.dedup();
            candidate_sets.push(cs);
        }
        (graph, candidate_sets)
    }

    #[test]
    fn test_manager_initialization() {
        let (graph, cand) = build_test_fixtures(5);
        let cpm = ClusterPointManager::new(0, &graph, &cand, 10, 0.3);
        assert_eq!(cpm.id(), 0);
        assert_eq!(cpm.centroid(), None);
        assert_eq!(cpm.size(), 0);
    }

    #[test]
    fn test_append_point() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, &graph, &cand, 10, 0.3);
        cpm.append(0, false);
        assert_eq!(cpm.size(), 1);
        assert!(cpm.is_cluster_point(0));
        assert!(!cpm.is_cluster_point(1));
        assert!(cpm.centroid().is_some());
        cpm.append(1, false);
        assert_eq!(cpm.size(), 2);
        assert!(cpm.is_cluster_point(1));
    }

    #[test]
    fn test_append_duplicate_ignored() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, &graph, &cand, 10, 0.3);
        cpm.append(0, false);
        cpm.append(0, false);
        assert_eq!(cpm.size(), 1);
    }

    #[test]
    fn test_pop_logic_on_overflow() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, &graph, &cand, 2, 0.3);
        cpm.append(0, true);
        cpm.append(1, true);
        assert!(cpm.size() <= 2);
        let result = cpm.append(2, true);
        if let Some((popped, _)) = result {
            assert!(!popped.is_empty());
            assert!(cpm.size() <= 2);
        }
    }

    #[test]
    fn test_should_affiliated_into_cluster() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, &graph, &cand, 10, 0.3);
        assert!(cpm.should_affiliated_into_cluster(0));
        cpm.append(0, false);
        assert!(!cpm.should_affiliated_into_cluster(0));
        assert!(cpm.should_affiliated_into_cluster(1));
    }

    #[test]
    fn test_merge_two_clusters() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm1 = ClusterPointManager::new(0, &graph, &cand, 10, 0.3);
        cpm1.append(0, false);
        let mut cpm2 = ClusterPointManager::new(1, &graph, &cand, 10, 0.3);
        cpm2.append(3, false);
        cpm2.append(4, false);
        let result = cpm1.append_new_cluster(cpm2, false);
        assert!(result.is_none());
        assert_eq!(cpm1.size(), 3);
        assert!(cpm1.is_cluster_point(0));
        assert!(cpm1.is_cluster_point(3));
        assert!(cpm1.is_cluster_point(4));
    }

    #[test]
    fn test_remove_and_clean_consistency() {
        let (graph, cand) = build_test_fixtures(5);
        let mut cpm = ClusterPointManager::new(0, &graph, &cand, 10, 0.3);
        cpm.append(0, false);
        cpm.append(1, false);
        cpm.append(2, false);
        assert_eq!(cpm.size(), 3);
        cpm.remove_and_clean(1);
        assert_eq!(cpm.size(), 2);
        assert!(!cpm.is_cluster_point(1));
        for (_, cp) in cpm.cluster_point().iter() {
            assert!(!cp.connected_set.contains(1));
            assert!(!cp.in_candidate_set.contains(1));
        }
    }

    #[test]
    fn test_split_with_disconnected_components() {
        let adj: Vec<Vec<u32>> = vec![vec![1], vec![0], vec![3], vec![2], vec![]];
        let graph = CsrGraph::from_adjacency_list(adj.clone(), 10);
        let mut candidate_sets: Vec<Vec<u32>> = Vec::with_capacity(5);
        for i in 0..5usize {
            let mut cs: Vec<u32> = adj[i]
                .iter()
                .copied()
                .chain(std::iter::once(i as u32))
                .collect();
            cs.sort_unstable();
            cs.dedup();
            candidate_sets.push(cs);
        }
        let mut cpm = ClusterPointManager::new(0, &graph, &candidate_sets, 2, 0.3);
        cpm.append(0, false);
        cpm.append(1, false);
        let result = cpm.append(2, true);
        if let Some((popped, sub_managers)) = result {
            let total = cpm.size() + popped.len() + sub_managers.iter().map(|s| s.size()).sum::<usize>();
            assert!(total >= 1);
        }
    }
}
