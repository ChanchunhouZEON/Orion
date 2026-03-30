use std::sync::RwLock;

use super::NSGGraphAccess;
use crate::common::{NSGError, NSGResult};

/// NSG graph: single-layer directed graph with a navigation node.
pub struct NSGGraph {
    /// adjacency[node_id] = neighbor list
    adjacency: Vec<RwLock<Vec<u32>>>,
    /// Navigation node (closest to centroid).
    pub navigation_node: u32,
    /// Maximum out-degree.
    pub max_degree: usize,
}

impl NSGGraph {
    pub fn new(capacity: usize, max_degree: usize) -> Self {
        let adjacency: Vec<RwLock<Vec<u32>>> = (0..capacity)
            .map(|_| RwLock::new(Vec::with_capacity(max_degree)))
            .collect();

        Self {
            adjacency,
            navigation_node: 0,
            max_degree,
        }
    }

    pub fn size(&self) -> usize {
        self.adjacency.len()
    }

    pub fn get_neighbors(&self, node_id: u32) -> NSGResult<Vec<u32>> {
        let guard = self.adjacency[node_id as usize]
            .read()
            .map_err(|_| NSGError::LockPoisoned("get_neighbors".to_string()))?;
        Ok(guard.clone())
    }

    pub fn set_neighbors(&self, node_id: u32, neighbors: Vec<u32>) -> NSGResult<()> {
        let mut guard = self.adjacency[node_id as usize]
            .write()
            .map_err(|_| NSGError::LockPoisoned("set_neighbors".to_string()))?;
        *guard = neighbors;
        Ok(())
    }
}

impl NSGGraphAccess for NSGGraph {
    fn get_neighbors(&self, node_id: u32) -> NSGResult<Vec<u32>> {
        NSGGraph::get_neighbors(self, node_id)
    }
    fn num_nodes(&self) -> usize {
        self.size()
    }
    fn navigation_node(&self) -> u32 {
        self.navigation_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let graph = NSGGraph::new(10, 5);
        assert_eq!(graph.size(), 10);
        assert_eq!(graph.max_degree, 5);
        assert_eq!(graph.navigation_node, 0);
    }

    #[test]
    fn test_get_set_neighbors() {
        let graph = NSGGraph::new(5, 4);
        graph.set_neighbors(0, vec![1, 2, 3]).unwrap();
        let neighbors = graph.get_neighbors(0).unwrap();
        assert_eq!(neighbors, vec![1, 2, 3]);
    }

    #[test]
    fn test_empty_neighbors() {
        let graph = NSGGraph::new(5, 4);
        let neighbors = graph.get_neighbors(0).unwrap();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_overwrite_neighbors() {
        let graph = NSGGraph::new(5, 4);
        graph.set_neighbors(0, vec![1, 2]).unwrap();
        graph.set_neighbors(0, vec![3, 4]).unwrap();
        let neighbors = graph.get_neighbors(0).unwrap();
        assert_eq!(neighbors, vec![3, 4]);
    }
}
