/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::common::ANNError;

use super::{AdjacencyList, VertexAndNeighbors};

#[derive(Debug)]
pub struct InMemoryGraph {
    pub final_graph: Vec<RwLock<VertexAndNeighbors>>,
    max_degree: u32,
}

impl InMemoryGraph {
    pub fn new(size: usize, max_degree: u32) -> Self {
        let mut graph = Vec::with_capacity(size);
        for id in 0..size {
            graph.push(RwLock::new(VertexAndNeighbors::for_range(
                id as u32,
                max_degree as usize,
            )));
        }
        Self {
            final_graph: graph,
            max_degree,
        }
    }

    pub fn max_degree(&self) -> u32 {
        self.max_degree
    }

    pub fn size(&self) -> usize {
        self.final_graph.len()
    }

    pub fn extend(&mut self, size: usize, max_degree: u32) {
        for id in 0..size {
            self.final_graph
                .push(RwLock::new(VertexAndNeighbors::for_range(
                    id as u32,
                    max_degree as usize,
                )));
        }
    }

    pub fn read_vertex_and_neighbors(
        &self,
        vertex_id: u32,
    ) -> Result<RwLockReadGuard<VertexAndNeighbors>, ANNError> {
        self.final_graph[vertex_id as usize].read().map_err(|err| {
            ANNError::log_lock_poison_error(format!(
                "PoisonError: Lock poisoned when reading final_graph for vertex_id {}, err={}",
                vertex_id, err
            ))
        })
    }

    pub fn write_vertex_and_neighbors(
        &self,
        vertex_id: u32,
    ) -> Result<RwLockWriteGuard<VertexAndNeighbors>, ANNError> {
        self.final_graph[vertex_id as usize].write().map_err(|err| {
            ANNError::log_lock_poison_error(format!(
                "PoisonError: Lock poisoned when writing final_graph for vertex_id {}, err={}",
                vertex_id, err
            ))
        })
    }

    /// Create an InMemoryGraph from a HashMap-based adjacency list.
    pub fn from_hashmap(
        graph: &std::collections::HashMap<u32, Vec<u32>>,
        num_nodes: usize,
        max_degree: u32,
    ) -> Self {
        let inmem = Self::new(num_nodes, max_degree);
        for (&node_id, neighbors) in graph {
            if (node_id as usize) < num_nodes {
                let mut vertex = inmem.final_graph[node_id as usize].write().unwrap();
                vertex.set_neighbors(AdjacencyList::from(neighbors.clone()));
            }
        }
        inmem
    }

    /// Get the neighbors of a node as a Vec<u32>.
    pub fn to_neighbor_vec(&self, node_id: u32) -> Result<Vec<u32>, ANNError> {
        let vertex = self.read_vertex_and_neighbors(node_id)?;
        Ok(vertex.get_neighbors().to_vec())
    }

    /// Check if an edge exists from `from` to `to`.
    pub fn contains_edge(&self, from: u32, to: u32) -> Result<bool, ANNError> {
        let vertex = self.read_vertex_and_neighbors(from)?;
        Ok(vertex.get_neighbors().contains(&to))
    }

    /// Set neighbors for a node from a Vec<u32>.
    pub fn set_neighbors_from_vec(
        &self,
        node_id: u32,
        neighbors: Vec<u32>,
    ) -> Result<(), ANNError> {
        let mut vertex = self.write_vertex_and_neighbors(node_id)?;
        vertex.set_neighbors(AdjacencyList::from(neighbors));
        Ok(())
    }

    /// Convert to a HashMap representation for serialization.
    pub fn to_hashmap(&self) -> std::collections::HashMap<u32, Vec<u32>> {
        let mut map = std::collections::HashMap::new();
        for (i, lock) in self.final_graph.iter().enumerate() {
            if let Ok(vertex) = lock.read() {
                let neighbors = vertex.get_neighbors().to_vec();
                if !neighbors.is_empty() {
                    map.insert(i as u32, neighbors);
                }
            }
        }
        map
    }

    /// Save graph adjacency data to an mmap-compatible file using the platform MmapGraph format.
    ///
    /// This saves only the graph structure (no vector data). Use `dimension=0` in the header.
    pub fn save_graph_mmap(&self, path: &str) -> Result<(), ANNError> {
        use platform::graph_mmap::{AlgorithmId, GraphHeader, GraphWriter};

        let header = GraphHeader {
            num_nodes: self.size() as u32,
            max_degree: self.max_degree,
            dimension: 0, // no vector data
            metric: 0,
            algorithm: AlgorithmId::DiskAnn,
        };

        let writer = GraphWriter::new(header);
        writer
            .write(
                path,
                |node_id| {
                    let neighbors = self.to_neighbor_vec(node_id).unwrap_or_default();
                    (neighbors.len() as u32, neighbors)
                },
                |_| Vec::new(), // no vector data
            )
            .map_err(|e| ANNError::log_index_error(format!("Failed to save mmap graph: {e}")))?;

        Ok(())
    }

    /// Save graph adjacency data AND vector data to an mmap-compatible file.
    ///
    /// Unlike `save_graph_mmap`, this includes vector data so that mmap search can
    /// read vectors directly from the file without keeping them in memory.
    pub fn save_graph_mmap_with_data<T: Into<f32> + Copy>(
        &self,
        path: &str,
        data: &[T],
        dimension: usize,
        metric: u8,
    ) -> Result<(), ANNError> {
        use platform::graph_mmap::{AlgorithmId, GraphHeader, GraphWriter};

        let num_nodes = self.size() as u32;
        let header = GraphHeader {
            num_nodes,
            max_degree: self.max_degree,
            dimension: dimension as u32,
            metric,
            algorithm: AlgorithmId::DiskAnn,
        };

        let writer = GraphWriter::new(header);
        writer
            .write(
                path,
                |node_id| {
                    let neighbors = self.to_neighbor_vec(node_id).unwrap_or_default();
                    (neighbors.len() as u32, neighbors)
                },
                |node_id| {
                    let start = node_id as usize * dimension;
                    let end = start + dimension;
                    if end <= data.len() {
                        data[start..end].iter().map(|&v| v.into()).collect()
                    } else {
                        vec![0.0f32; dimension]
                    }
                },
            )
            .map_err(|e| {
                ANNError::log_index_error(format!("Failed to save mmap graph with data: {e}"))
            })?;

        Ok(())
    }

    /// Load graph adjacency data from an mmap-compatible file.
    pub fn load_graph_mmap(path: &str) -> Result<Self, ANNError> {
        use platform::graph_mmap::MmapGraph;

        let mmap_graph = MmapGraph::open(path, 0)
            .map_err(|e| ANNError::log_index_error(format!("Failed to open mmap graph: {e}")))?;

        let num_nodes = mmap_graph.header.num_nodes as usize;
        let max_degree = mmap_graph.header.max_degree;
        let graph = Self::new(num_nodes, max_degree);

        for i in 0..num_nodes as u32 {
            let neighbors = mmap_graph.neighbors(i).to_vec();
            graph
                .set_neighbors_from_vec(i, neighbors)
                .map_err(|e| ANNError::log_index_error(format!("Failed to set neighbors: {e}")))?;
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod graph_tests {
    use crate::model::{GRAPH_SLACK_FACTOR, graph::AdjacencyList};

    use super::*;

    #[test]
    fn test_new() {
        let graph = InMemoryGraph::new(10, 10);
        let capacity = (GRAPH_SLACK_FACTOR * 10_f64).ceil() as usize;

        assert_eq!(graph.final_graph.len(), 10);
        for i in 0..10 {
            let neighbor = graph.final_graph[i].read().unwrap();
            assert_eq!(neighbor.vertex_id, i as u32);
            assert_eq!(neighbor.get_neighbors().capacity(), capacity);
        }
    }

    #[test]
    fn test_size() {
        let graph = InMemoryGraph::new(10, 10);
        assert_eq!(graph.size(), 10);
    }

    #[test]
    fn test_extend() {
        let mut graph = InMemoryGraph::new(10, 10);
        graph.extend(10, 10);
        assert_eq!(graph.size(), 20);

        let capacity = (GRAPH_SLACK_FACTOR * 10_f64).ceil() as usize;
        let mut id: u32 = 0;

        for i in 10..20 {
            let neighbor = graph.final_graph[i].read().unwrap();
            assert_eq!(neighbor.vertex_id, id);
            assert_eq!(neighbor.get_neighbors().capacity(), capacity);
            id += 1;
        }
    }

    #[test]
    fn test_read_vertex_and_neighbors() {
        let graph = InMemoryGraph::new(10, 10);
        let neighbor = graph.read_vertex_and_neighbors(0);
        assert!(neighbor.is_ok());
        assert_eq!(neighbor.unwrap().vertex_id, 0);
    }

    #[test]
    fn test_write_vertex_and_neighbors() {
        let graph = InMemoryGraph::new(10, 10);
        {
            let neighbor = graph.write_vertex_and_neighbors(0);
            assert!(neighbor.is_ok());
            neighbor.unwrap().add_to_neighbors(10, 10);
        }

        let neighbor = graph.read_vertex_and_neighbors(0).unwrap();
        assert_eq!(neighbor.get_neighbors(), &AdjacencyList::from(vec![10_u32]));
    }
}
