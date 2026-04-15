/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::AdjacencyList;

#[derive(Debug)]
pub struct VertexAndNeighbors {
    pub vertex_id: u32,
    neighbors: AdjacencyList,
    /// Parallel distance array (sorted ascending), maintained under `staged_diskann`.
    #[cfg(feature = "staged_diskann")]
    pub neighbor_dists: Vec<f32>,
}

impl VertexAndNeighbors {
    pub fn for_range(id: u32, range: usize) -> Self {
        Self {
            vertex_id: id,
            neighbors: AdjacencyList::for_range(range),
            #[cfg(feature = "staged_diskann")]
            neighbor_dists: Vec::with_capacity(range),
        }
    }

    pub fn new(vertex_id: u32, neighbors: AdjacencyList) -> Self {
        #[cfg(feature = "staged_diskann")]
        let neighbor_dists = Vec::new();
        Self {
            vertex_id,
            neighbors,
            #[cfg(feature = "staged_diskann")]
            neighbor_dists,
        }
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.neighbors.len()
    }

    #[inline(always)]
    pub fn set_neighbors(&mut self, new_neighbors: AdjacencyList) {
        self.neighbors = new_neighbors;
        #[cfg(feature = "staged_diskann")]
        {
            self.neighbor_dists.clear();
        }
    }

    /// Set neighbors with parallel distance array (already sorted by distance ascending).
    /// Used after `prune_neighbors` where the pruned list preserves pool order.
    #[cfg(feature = "staged_diskann")]
    pub fn set_neighbors_sorted(&mut self, new_neighbors: AdjacencyList, dists: Vec<f32>) {
        debug_assert_eq!(new_neighbors.len(), dists.len());
        self.neighbors = new_neighbors;
        self.neighbor_dists = dists;
    }

    #[inline(always)]
    pub fn get_neighbors(&self) -> &AdjacencyList {
        &self.neighbors
    }

    /// Get the parallel distance array (only populated under `staged_diskann`).
    #[cfg(feature = "staged_diskann")]
    #[inline(always)]
    pub fn get_neighbor_dists(&self) -> &[f32] {
        &self.neighbor_dists
    }

    /// Consume self and return the neighbor list as a plain `Vec<u32>`.
    #[inline(always)]
    pub fn into_neighbors(self) -> Vec<u32> {
        self.neighbors.into_vec()
    }

    /// Consume self and return (neighbor_ids, distances).
    /// Under `staged_diskann`, distances are maintained during build;
    /// without the feature, the distance vec is empty.
    #[inline(always)]
    pub fn into_neighbors_and_dists(self) -> (Vec<u32>, Vec<f32>) {
        #[cfg(feature = "staged_diskann")]
        {
            (self.neighbors.into_vec(), self.neighbor_dists)
        }
        #[cfg(not(feature = "staged_diskann"))]
        {
            (self.neighbors.into_vec(), Vec::new())
        }
    }

    /// Consume self and return (neighbor_ids, cliff_position).
    /// Cliff = position of largest distance ratio gap (min = degree/2).
    /// Distances are dropped immediately after computation.
    #[inline(always)]
    pub fn into_neighbors_and_cliff(self) -> (Vec<u32>, usize) {
        let nbrs = self.neighbors.into_vec();
        let degree = nbrs.len();

        #[cfg(feature = "staged_diskann")]
        {
            let dists = self.neighbor_dists;
            if dists.len() == degree && degree >= 3 {
                let half = degree / 2;
                let mut cliff = half;
                let mut max_ratio = 0.0f32;
                for i in half..degree - 1 {
                    if dists[i] > 0.0 {
                        let ratio = dists[i + 1] / dists[i];
                        if ratio > max_ratio {
                            max_ratio = ratio;
                            cliff = i + 1;
                        }
                    }
                }
                return (nbrs, cliff.min(degree));
            }
        }

        (nbrs, degree / 2)
    }

    /// Original unsorted add — used when `staged_diskann` is NOT enabled.
    pub fn add_to_neighbors(&mut self, node_id: u32, range: u32) -> Option<Vec<u32>> {
        if self.neighbors.contains(&node_id) {
            return None;
        }

        let neighbor_len = self.neighbors.len();

        if neighbor_len < (GRAPH_SLACK_FACTOR * range as f64) as usize {
            self.neighbors.push(node_id);
            return None;
        }

        let mut copy_of_neighbors = Vec::with_capacity(neighbor_len + 1);
        unsafe {
            let dst = copy_of_neighbors.as_mut_ptr();
            std::ptr::copy_nonoverlapping(self.neighbors.as_ptr(), dst, neighbor_len);
            dst.add(neighbor_len).write(node_id);
            copy_of_neighbors.set_len(neighbor_len + 1);
        }

        Some(copy_of_neighbors)
    }

    /// Sorted insert: maintains distance-ascending order in neighbors + neighbor_dists.
    ///
    /// Returns `Some(copy)` if degree overflows (caller should re-prune),
    /// `None` if inserted successfully within capacity.
    #[cfg(feature = "staged_diskann")]
    pub fn add_sorted(&mut self, node_id: u32, distance: f32, range: u32) -> Option<Vec<u32>> {
        if self.neighbors.contains(&node_id) {
            return None;
        }

        let neighbor_len = self.neighbors.len();

        if neighbor_len >= (GRAPH_SLACK_FACTOR * range as f64) as usize {
            // Overflow — return copy for re-prune (distances not needed, pool rebuilt).
            let mut copy = Vec::with_capacity(neighbor_len + 1);
            copy.extend_from_slice(&self.neighbors);
            copy.push(node_id);
            return Some(copy);
        }

        // Find sorted insertion position via binary search on distances.
        let pos = self.neighbor_dists.partition_point(|&d| d < distance);

        // Insert at pos, shifting later elements.
        self.neighbors.insert(pos, node_id);
        self.neighbor_dists.insert(pos, distance);

        None
    }
}

pub const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;

#[cfg(test)]
mod vertex_and_neighbors_tests {
    use super::*;

    #[test]
    fn test_set_with_capacity() {
        let neighbors = VertexAndNeighbors::for_range(20, 10);
        assert_eq!(neighbors.vertex_id, 20);
        assert_eq!(
            neighbors.neighbors.capacity(),
            (10_f32 * GRAPH_SLACK_FACTOR as f32).ceil() as usize
        );
    }

    #[test]
    fn test_size() {
        let mut neighbors = VertexAndNeighbors::for_range(20, 10);
        for i in 0..5 {
            neighbors.neighbors.push(i);
        }
        assert_eq!(neighbors.size(), 5);
    }

    #[test]
    fn test_set_neighbors() {
        let mut neighbors = VertexAndNeighbors::for_range(20, 10);
        let new_vec = AdjacencyList::from(vec![1, 2, 3, 4, 5]);
        neighbors.set_neighbors(AdjacencyList::from(vec![1, 2, 3, 4, 5]));
        assert_eq!(neighbors.neighbors, new_vec);
    }

    #[test]
    fn test_add_to_neighbors() {
        let mut neighbors = VertexAndNeighbors::for_range(20, 10);

        assert_eq!(neighbors.add_to_neighbors(1, 1), None);
        assert_eq!(neighbors.neighbors, AdjacencyList::from(vec![1]));

        assert_eq!(neighbors.add_to_neighbors(1, 1), None);
        assert_eq!(neighbors.neighbors, AdjacencyList::from(vec![1]));

        let ret = neighbors.add_to_neighbors(2, 1);
        assert!(ret.is_some());
        assert_eq!(ret.unwrap(), vec![1, 2]);
        assert_eq!(neighbors.neighbors, AdjacencyList::from(vec![1]));

        assert_eq!(neighbors.add_to_neighbors(2, 2), None);
        assert_eq!(neighbors.neighbors, AdjacencyList::from(vec![1, 2]));
    }

    #[cfg(feature = "staged_diskann")]
    #[test]
    fn test_add_sorted() {
        let mut vn = VertexAndNeighbors::for_range(0, 10);

        // Insert in non-sorted order, should end up sorted.
        assert_eq!(vn.add_sorted(3, 5.0, 10), None);
        assert_eq!(vn.add_sorted(1, 2.0, 10), None);
        assert_eq!(vn.add_sorted(2, 3.0, 10), None);
        assert_eq!(vn.add_sorted(4, 1.0, 10), None);

        assert_eq!(&*vn.neighbors, &[4, 1, 2, 3]);
        assert_eq!(vn.neighbor_dists, vec![1.0, 2.0, 3.0, 5.0]);

        // Duplicate — ignored.
        assert_eq!(vn.add_sorted(1, 2.0, 10), None);
        assert_eq!(vn.size(), 4);

        // Overflow: range=3, slack=3*1.3=3.9 → capacity 3
        let mut vn2 = VertexAndNeighbors::for_range(0, 10);
        vn2.add_sorted(1, 1.0, 3);
        vn2.add_sorted(2, 2.0, 3);
        vn2.add_sorted(3, 3.0, 3);
        let overflow = vn2.add_sorted(4, 0.5, 3);
        assert!(overflow.is_some());
        assert_eq!(overflow.unwrap(), vec![1, 2, 3, 4]);
    }
}
