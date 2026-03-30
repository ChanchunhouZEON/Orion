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
}

impl VertexAndNeighbors {
    pub fn for_range(id: u32, range: usize) -> Self {
        Self {
            vertex_id: id,
            neighbors: AdjacencyList::for_range(range),
        }
    }

    pub fn new(vertex_id: u32, neighbors: AdjacencyList) -> Self {
        Self {
            vertex_id,
            neighbors,
        }
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.neighbors.len()
    }

    #[inline(always)]
    pub fn set_neighbors(&mut self, new_neighbors: AdjacencyList) {
        self.neighbors = new_neighbors;
    }

    #[inline(always)]
    pub fn get_neighbors(&self) -> &AdjacencyList {
        &self.neighbors
    }

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
}
