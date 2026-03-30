use std::sync::RwLock;

use super::HNSWGraphAccess;
use crate::common::{HNSWError, HNSWResult};

/// Multi-layer graph for HNSW.
///
/// `layers[l][i]` contains the neighbor list for node `i` at layer `l`.
/// Nodes that do not exist at layer `l` have empty neighbor lists.
pub struct HNSWGraph {
    /// layers[layer_idx][node_idx] = neighbor list
    layers: Vec<Vec<RwLock<Vec<u32>>>>,
    /// Current entry point node ID.
    pub entry_point: u32,
    /// Maximum layer currently in the graph.
    pub max_level: usize,
    /// Per-node assigned level.
    node_levels: Vec<usize>,
    /// Number of nodes currently in the graph.
    num_nodes: usize,
    /// Max connections per node (per layer, except layer 0).
    m: usize,
    /// Max connections at layer 0.
    m_max0: usize,
}

impl HNSWGraph {
    pub fn new(capacity: usize, m: usize, m_max0: usize) -> Self {
        // Start with layer 0 only.
        let layer0: Vec<RwLock<Vec<u32>>> = (0..capacity)
            .map(|_| RwLock::new(Vec::with_capacity(m_max0)))
            .collect();

        Self {
            layers: vec![layer0],
            entry_point: 0,
            max_level: 0,
            node_levels: vec![0; capacity],
            num_nodes: 0,
            m,
            m_max0,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn set_num_nodes(&mut self, n: usize) {
        self.num_nodes = n;
    }

    pub fn node_level(&self, node_id: u32) -> usize {
        self.node_levels[node_id as usize]
    }

    pub fn set_node_level(&mut self, node_id: u32, level: usize) {
        self.node_levels[node_id as usize] = level;
    }

    /// Ensure the graph has enough layers.
    pub fn ensure_layers(&mut self, level: usize, capacity: usize) {
        while self.layers.len() <= level {
            let layer: Vec<RwLock<Vec<u32>>> = (0..capacity)
                .map(|_| RwLock::new(Vec::with_capacity(self.m)))
                .collect();
            self.layers.push(layer);
        }
    }

    /// Get max connections allowed at a given layer.
    pub fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.m
        }
    }

    /// Read neighbors of node at a given layer.
    pub fn get_neighbors(&self, node_id: u32, layer: usize) -> HNSWResult<Vec<u32>> {
        if layer >= self.layers.len() {
            return Ok(Vec::new());
        }
        let guard = self.layers[layer][node_id as usize]
            .read()
            .map_err(|_| HNSWError::LockPoisoned("get_neighbors".to_string()))?;
        Ok(guard.clone())
    }

    /// Set neighbors for a node at a given layer.
    pub fn set_neighbors(&self, node_id: u32, layer: usize, neighbors: Vec<u32>) -> HNSWResult<()> {
        let mut guard = self.layers[layer][node_id as usize]
            .write()
            .map_err(|_| HNSWError::LockPoisoned("set_neighbors".to_string()))?;
        *guard = neighbors;
        Ok(())
    }

    /// Add a neighbor to a node's list at a given layer (if not already present).
    pub fn add_neighbor(&self, node_id: u32, neighbor_id: u32, layer: usize) -> HNSWResult<()> {
        let mut guard = self.layers[layer][node_id as usize]
            .write()
            .map_err(|_| HNSWError::LockPoisoned("add_neighbor".to_string()))?;
        if !guard.contains(&neighbor_id) {
            guard.push(neighbor_id);
        }
        Ok(())
    }
}

impl HNSWGraphAccess for HNSWGraph {
    fn get_neighbors(&self, node_id: u32, layer: usize) -> HNSWResult<Vec<u32>> {
        HNSWGraph::get_neighbors(self, node_id, layer)
    }
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }
    fn max_level(&self) -> usize {
        self.max_level
    }
    fn entry_point(&self) -> u32 {
        self.entry_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_graph_structure() {
        let graph = HNSWGraph::new(10, 4, 8);
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.max_level, 0);
        assert_eq!(graph.entry_point, 0);
        assert_eq!(graph.max_connections(0), 8); // m_max0
        assert_eq!(graph.max_connections(1), 4); // m
    }

    #[test]
    fn test_get_set_neighbors() {
        let graph = HNSWGraph::new(5, 4, 8);
        graph.set_neighbors(0, 0, vec![1, 2, 3]).unwrap();
        let neighbors = graph.get_neighbors(0, 0).unwrap();
        assert_eq!(neighbors, vec![1, 2, 3]);
    }

    #[test]
    fn test_get_neighbors_empty_layer() {
        let graph = HNSWGraph::new(5, 4, 8);
        // Layer 5 doesn't exist yet
        let neighbors = graph.get_neighbors(0, 5).unwrap();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_add_neighbor_dedup() {
        let graph = HNSWGraph::new(5, 4, 8);
        graph.add_neighbor(0, 1, 0).unwrap();
        graph.add_neighbor(0, 1, 0).unwrap(); // duplicate
        graph.add_neighbor(0, 2, 0).unwrap();
        let neighbors = graph.get_neighbors(0, 0).unwrap();
        assert_eq!(neighbors, vec![1, 2]); // no duplicate
    }

    #[test]
    fn test_ensure_layers() {
        let mut graph = HNSWGraph::new(5, 4, 8);
        assert_eq!(graph.layers.len(), 1); // only layer 0
        graph.ensure_layers(3, 5);
        assert_eq!(graph.layers.len(), 4); // layers 0,1,2,3
    }

    #[test]
    fn test_node_level() {
        let mut graph = HNSWGraph::new(5, 4, 8);
        graph.set_node_level(2, 3);
        assert_eq!(graph.node_level(2), 3);
    }

    #[test]
    fn test_set_num_nodes() {
        let mut graph = HNSWGraph::new(5, 4, 8);
        graph.set_num_nodes(3);
        assert_eq!(graph.num_nodes(), 3);
    }
}
