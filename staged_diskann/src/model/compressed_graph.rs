/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use diskann::common::ANNResult;
use serde::{Deserialize, Serialize};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Slack factor for neighbor list capacity allocation (same as diskann-core).
const GRAPH_SLACK_FACTOR: f64 = 1.3;

/// Per-node data in the compressed graph.
///
/// Mirrors diskann's `VertexAndNeighbors` but adds a `compressed_degree`
/// field that splits the neighbor list into two logical segments:
///
/// ```text
/// neighbors[0..compressed_degree]           → compressed neighbors (phase 2)
/// neighbors[compressed_degree..degree]      → remaining full neighbors (phase 1 only)
/// ```
#[derive(Debug)]
pub struct CompressedVertexAndNeighbors {
    pub vertex_id: u32,
    neighbors: Vec<u32>,

    // Size of compressed neighbors at the front of `neighbors`.
    compressed_degree: u32,
}

impl CompressedVertexAndNeighbors {
    pub fn for_range(id: u32, max_degree: usize) -> Self {
        let capacity = (max_degree as f64 * GRAPH_SLACK_FACTOR).ceil() as usize;
        Self {
            vertex_id: id,
            neighbors: Vec::with_capacity(capacity),
            compressed_degree: 0,
        }
    }

    pub fn new(vertex_id: u32, neighbors: Vec<u32>, compressed_degree: u32) -> Self {
        Self {
            vertex_id,
            neighbors,
            compressed_degree,
        }
    }

    /// Total degree (all neighbors).
    #[inline(always)]
    pub fn degree(&self) -> usize {
        self.neighbors.len()
    }

    /// Compressed degree.
    #[inline(always)]
    pub fn compressed_degree(&self) -> u32 {
        self.compressed_degree
    }

    /// All neighbors (compressed + rest).
    #[inline(always)]
    pub fn get_neighbors(&self) -> &[u32] {
        &self.neighbors
    }

    /// Only compressed neighbors (first `compressed_degree` entries).
    #[inline(always)]
    pub fn get_compressed_neighbors(&self) -> &[u32] {
        let cd = (self.compressed_degree as usize).min(self.neighbors.len());
        &self.neighbors[..cd]
    }

    /// Only the remaining (full-graph-only) neighbors.
    #[inline(always)]
    pub fn get_rest_neighbors(&self) -> &[u32] {
        let cd = (self.compressed_degree as usize).min(self.neighbors.len());
        &self.neighbors[cd..]
    }

    /// Replace all neighbors and compressed degree.
    pub fn set_neighbors(&mut self, neighbors: Vec<u32>, compressed_degree: u32) {
        self.neighbors = neighbors;
        self.compressed_degree = compressed_degree;
    }

    /// Set neighbors from separate compressed + rest lists.
    pub fn set_neighbors_split(&mut self, compressed: &[u32], rest: &[u32]) {
        self.neighbors.clear();
        self.neighbors.reserve(compressed.len() + rest.len());
        self.neighbors.extend_from_slice(compressed);
        self.neighbors.extend_from_slice(rest);
        self.compressed_degree = compressed.len() as u32;
    }

    /// Add a neighbor to the back (non-compressed portion).
    /// Returns a copy of all neighbors if capacity would be exceeded.
    pub fn add_to_neighbors(&mut self, node_id: u32, range: u32) -> Option<Vec<u32>> {
        if self.neighbors.contains(&node_id) {
            return None;
        }
        if self.neighbors.len() < (GRAPH_SLACK_FACTOR * range as f64) as usize {
            self.neighbors.push(node_id);
            return None;
        }
        let mut copy = Vec::with_capacity(self.neighbors.len() + 1);
        copy.extend_from_slice(&self.neighbors);
        copy.push(node_id);
        Some(copy)
    }
}

/// Compressed graph following diskann's `InMemoryGraph` pattern with
/// per-node `RwLock<CompressedVertexAndNeighbors>`.
///
/// On-disk layout metadata (serialized in header):
/// ```text
/// max_degree (u32) | max_compressed_degree (u32) | num_nodes (u32)
/// ```
///
/// Per-node record:
/// ```text
/// degree (u32) | compressed_degree (u32) |
/// compressed_neighbors [compressed_degree × u32] |
/// rest_neighbors [(degree - compressed_degree) × u32]
/// ```
#[derive(Debug)]
pub struct CompressedGraph {
    pub final_graph: Vec<RwLock<CompressedVertexAndNeighbors>>,
    max_degree: u32,
    max_compressed_degree: u32,
}

/// Serializable on-disk representation.
#[derive(Serialize, Deserialize, Debug)]
pub struct CompressedGraphOnDisk {
    pub max_degree: u32,
    pub max_compressed_degree: u32,
    pub num_nodes: u32,
    /// Per-node: (compressed_degree, neighbors_vec)
    /// neighbors_vec is ordered: [compressed | rest]
    pub nodes: Vec<(u32, Vec<u32>)>,
}

impl CompressedGraph {
    /// Create an empty graph with the given size and degree limits.
    pub fn new(num_nodes: usize, max_degree: u32) -> Self {
        let mut graph = Vec::with_capacity(num_nodes);
        for id in 0..num_nodes {
            graph.push(RwLock::new(CompressedVertexAndNeighbors::for_range(
                id as u32,
                max_degree as usize,
            )));
        }
        Self {
            final_graph: graph,
            max_degree,
            max_compressed_degree: 0,
        }
    }

    /// Build from a diskann InMemoryGraph (imports full neighbors, compressed_degree = 0).
    pub fn from_inmem_graph(graph: &diskann::model::InMemoryGraph) -> Self {
        let num_nodes = graph.size();
        let max_degree = graph.max_degree();
        let cg = Self::new(num_nodes, max_degree);
        for i in 0..num_nodes as u32 {
            if let Ok(neighbors) = graph.to_neighbor_vec(i) {
                let mut v = cg.final_graph[i as usize].write().unwrap();
                v.set_neighbors(neighbors, 0);
            }
        }
        cg
    }

    pub fn max_degree(&self) -> u32 {
        self.max_degree
    }

    pub fn max_compressed_degree(&self) -> u32 {
        self.max_compressed_degree
    }

    pub fn size(&self) -> usize {
        self.final_graph.len()
    }

    /// Read-lock a vertex.
    pub fn read_vertex(
        &self,
        vertex_id: u32,
    ) -> Result<RwLockReadGuard<CompressedVertexAndNeighbors>, String> {
        self.final_graph[vertex_id as usize]
            .read()
            .map_err(|e| format!("PoisonError reading vertex {}: {}", vertex_id, e))
    }

    /// Write-lock a vertex.
    pub fn write_vertex(
        &self,
        vertex_id: u32,
    ) -> Result<RwLockWriteGuard<CompressedVertexAndNeighbors>, String> {
        self.final_graph[vertex_id as usize]
            .write()
            .map_err(|e| format!("PoisonError writing vertex {}: {}", vertex_id, e))
    }

    /// Get all neighbors as Vec.
    pub fn to_neighbor_vec(&self, node_id: u32) -> Result<Vec<u32>, String> {
        let v = self.read_vertex(node_id)?;
        Ok(v.get_neighbors().to_vec())
    }

    /// Get only compressed neighbors as Vec.
    pub fn to_compressed_neighbor_vec(&self, node_id: u32) -> Result<Vec<u32>, String> {
        let v = self.read_vertex(node_id)?;
        Ok(v.get_compressed_neighbors().to_vec())
    }

    /// Get compressed degree for a node.
    pub fn compressed_degree(&self, node_id: u32) -> u32 {
        self.read_vertex(node_id)
            .map(|v| v.compressed_degree())
            .unwrap_or(0)
    }

    /// Set full + compressed neighbors for a node.
    pub fn set_neighbors(
        &self,
        node_id: u32,
        neighbors: Vec<u32>,
        compressed_degree: u32,
    ) -> Result<(), String> {
        let mut v = self.write_vertex(node_id)?;
        v.set_neighbors(neighbors, compressed_degree);
        Ok(())
    }

    /// Set neighbors from separate compressed + rest lists.
    pub fn set_neighbors_split(
        &self,
        node_id: u32,
        compressed: &[u32],
        rest: &[u32],
    ) -> Result<(), String> {
        let mut v = self.write_vertex(node_id)?;
        v.set_neighbors_split(compressed, rest);
        Ok(())
    }

    /// Recompute `max_compressed_degree` from all vertices.
    pub fn update_max_compressed_degree(&mut self) {
        let mut max_cd = 0u32;
        for lock in &self.final_graph {
            if let Ok(v) = lock.read() {
                max_cd = max_cd.max(v.compressed_degree());
            }
        }
        self.max_compressed_degree = max_cd;
    }

    /// Convert to a HashMap representation (for serialization / visualization).
    pub fn to_hashmap(&self) -> std::collections::HashMap<u32, Vec<u32>> {
        let mut map = std::collections::HashMap::new();
        for (i, lock) in self.final_graph.iter().enumerate() {
            if let Ok(v) = lock.read() {
                let neighbors = v.get_neighbors().to_vec();
                if !neighbors.is_empty() {
                    map.insert(i as u32, neighbors);
                }
            }
        }
        map
    }

    /// Convert to hashmap showing only compressed neighbors.
    pub fn to_compressed_hashmap(&self) -> std::collections::HashMap<u32, Vec<u32>> {
        let mut map = std::collections::HashMap::new();
        for (i, lock) in self.final_graph.iter().enumerate() {
            if let Ok(v) = lock.read() {
                let compressed = v.get_compressed_neighbors().to_vec();
                if !compressed.is_empty() {
                    map.insert(i as u32, compressed);
                }
            }
        }
        map
    }

    /// Serialize to the on-disk format.
    pub fn to_on_disk(&self) -> CompressedGraphOnDisk {
        let num_nodes = self.final_graph.len();
        let mut nodes = Vec::with_capacity(num_nodes);
        for lock in &self.final_graph {
            let v = lock.read().unwrap();
            nodes.push((v.compressed_degree(), v.get_neighbors().to_vec()));
        }
        CompressedGraphOnDisk {
            max_degree: self.max_degree,
            max_compressed_degree: self.max_compressed_degree,
            num_nodes: num_nodes as u32,
            nodes,
        }
    }

    /// Reconstruct from on-disk format.
    pub fn from_on_disk(on_disk: CompressedGraphOnDisk) -> Self {
        let num_nodes = on_disk.num_nodes as usize;
        let mut graph = Vec::with_capacity(num_nodes);
        for (id, (cd, neighbors)) in on_disk.nodes.into_iter().enumerate() {
            graph.push(RwLock::new(CompressedVertexAndNeighbors::new(
                id as u32, neighbors, cd,
            )));
        }
        Self {
            final_graph: graph,
            max_degree: on_disk.max_degree,
            max_compressed_degree: on_disk.max_compressed_degree,
        }
    }

    /// Save to file via bincode.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> ANNResult<()> {
        let on_disk = self.to_on_disk();
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(on_disk, &mut writer, config)?;
        Ok(())
    }

    /// Load from file via bincode.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> ANNResult<Self> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let on_disk: CompressedGraphOnDisk =
            bincode::serde::decode_from_std_read(&mut reader, config)?;
        Ok(Self::from_on_disk(on_disk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_compressed_graph() {
        let g = CompressedGraph::new(10, 32);
        assert_eq!(g.size(), 10);
        assert_eq!(g.max_degree(), 32);
        assert_eq!(g.max_compressed_degree(), 0);
    }

    #[test]
    fn test_set_and_read_neighbors() {
        let g = CompressedGraph::new(5, 16);
        g.set_neighbors(0, vec![1, 2, 3, 4, 5], 3).unwrap();

        let v = g.read_vertex(0).unwrap();
        assert_eq!(v.get_neighbors(), &[1, 2, 3, 4, 5]);
        assert_eq!(v.get_compressed_neighbors(), &[1, 2, 3]);
        assert_eq!(v.get_rest_neighbors(), &[4, 5]);
        assert_eq!(v.compressed_degree(), 3);
        assert_eq!(v.degree(), 5);
    }

    #[test]
    fn test_set_neighbors_split() {
        let g = CompressedGraph::new(5, 16);
        g.set_neighbors_split(1, &[10, 20], &[30, 40, 50]).unwrap();

        let v = g.read_vertex(1).unwrap();
        assert_eq!(v.get_neighbors(), &[10, 20, 30, 40, 50]);
        assert_eq!(v.compressed_degree(), 2);
    }

    #[test]
    fn test_to_compressed_hashmap() {
        let g = CompressedGraph::new(3, 16);
        g.set_neighbors(0, vec![1, 2, 3], 2).unwrap();
        g.set_neighbors(1, vec![0, 2], 1).unwrap();

        let full = g.to_hashmap();
        assert_eq!(full[&0], vec![1, 2, 3]);
        assert_eq!(full[&1], vec![0, 2]);

        let compressed = g.to_compressed_hashmap();
        assert_eq!(compressed[&0], vec![1, 2]);
        assert_eq!(compressed[&1], vec![0]);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let g = CompressedGraph::new(3, 16);
        g.set_neighbors(0, vec![1, 2], 1).unwrap();
        g.set_neighbors(1, vec![0, 2, 3], 2).unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("test_compressed_graph_roundtrip.bin");
        g.save(&path).unwrap();

        let loaded = CompressedGraph::load(&path).unwrap();
        assert_eq!(loaded.size(), 3);
        assert_eq!(loaded.max_degree(), 16);

        let v0 = loaded.read_vertex(0).unwrap();
        assert_eq!(v0.get_neighbors(), &[1, 2]);
        assert_eq!(v0.compressed_degree(), 1);

        let v1 = loaded.read_vertex(1).unwrap();
        assert_eq!(v1.get_neighbors(), &[0, 2, 3]);
        assert_eq!(v1.compressed_degree(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_update_max_compressed_degree() {
        let mut g = CompressedGraph::new(3, 16);
        g.set_neighbors(0, vec![1, 2, 3, 4], 3).unwrap();
        g.set_neighbors(1, vec![0, 2], 1).unwrap();
        g.set_neighbors(2, vec![0, 1, 3], 2).unwrap();

        g.update_max_compressed_degree();
        assert_eq!(g.max_compressed_degree(), 3);
    }
}
