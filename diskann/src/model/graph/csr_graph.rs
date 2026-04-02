/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::sync::Arc;

use crate::common::AlignedBoxWithSlice;

/// Cache-line aligned CSR graph with per-node bidirectional bitset.
///
/// All node data lives in a single `AlignedBoxWithSlice<u32>` buffer with
/// 64-byte alignment. Each node occupies a fixed `stride` (in u32 units)
/// so that every node starts on a cache-line boundary.
///
/// Per-node slot layout (u32 units, 32-byte aligned header):
/// ```text
///   [0]         degree:       u32
///   [1..5]      bidir_bits:   u128 (4 × u32, little-endian word order)
///   [5..8]      _reserved:    3 × u32 padding → header = 8 u32 = 32 bytes
///   [8..8+max]  neighbors:    u32 × max_degree (original Vamana order)
///   [8+max..]   padding to stride (cache-line multiple)
/// ```
///
/// Neighbor order is preserved from the Vamana graph (search-list rank).
/// `contains_edge` uses linear scan (degree ≤ 32 → 128 bytes, 2 aligned cache lines).
/// `bidir_bits[i]` = 1 iff `neighbors[i]` also has a reverse edge back.
const HEADER_U32: usize = 8;
const CACHE_LINE_BYTES: usize = 64;

#[inline]
const fn compute_stride(max_degree: usize) -> usize {
    let raw_bytes = (HEADER_U32 + max_degree) * 4;
    let aligned = (raw_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    aligned / 4
}

#[derive(Debug)]
pub struct CsrGraph {
    data: Arc<AlignedBoxWithSlice<u32>>,
    num_nodes: u32,
    pub max_degree: u32,
    stride_u32: u32,
}

impl Clone for CsrGraph {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            num_nodes: self.num_nodes,
            max_degree: self.max_degree,
            stride_u32: self.stride_u32,
        }
    }
}

impl CsrGraph {
    /// Build from per-node neighbor lists. Order is preserved (not sorted).
    /// Bidir bits are zeroed; call `compute_bidir()` to populate them.
    pub fn from_adjacency_list(adj: Vec<Vec<u32>>, max_degree: u32) -> Self {
        let num_nodes = adj.len();
        let stride = compute_stride(max_degree as usize) as u32;
        let total_u32 = num_nodes * stride as usize;

        let mut buf = AlignedBoxWithSlice::<u32>::new(total_u32, CACHE_LINE_BYTES)
            .expect("CsrGraph allocation failed");

        for (i, nbrs) in adj.iter().enumerate() {
            let base = i * stride as usize;
            let deg = nbrs.len().min(max_degree as usize);
            buf[base] = deg as u32;
            buf[base + HEADER_U32..base + HEADER_U32 + deg]
                .copy_from_slice(&nbrs[..deg]);
        }

        Self {
            data: Arc::new(buf),
            num_nodes: num_nodes as u32,
            max_degree,
            stride_u32: stride,
        }
    }

    /// Build directly from an `InMemoryGraph` by consuming it.
    ///
    /// Each node's `Vec<u32>` neighbors are moved out of the RwLock via
    /// `into_inner()` + `into_neighbors()`, copied into the aligned buffer,
    /// and immediately dropped — so at most ONE node's `Vec<u32>` is alive
    /// at a time beyond the aligned buffer, minimizing peak memory.
    pub fn from_inmem_graph(
        graph: super::InMemoryGraph,
        max_degree: u32,
    ) -> crate::common::ANNResult<Self> {
        let num_nodes = graph.size();
        let stride = compute_stride(max_degree as usize) as u32;
        let total_u32 = num_nodes * stride as usize;

        let mut buf = AlignedBoxWithSlice::<u32>::new(total_u32, CACHE_LINE_BYTES)?;

        for (i, lock) in graph.final_graph.into_iter().enumerate() {
            let nbrs = lock.into_inner().map_err(|e| {
                crate::common::ANNError::log_lock_poison_error(format!(
                    "CsrGraph::from_inmem_graph: RwLock poisoned at node {}: {}",
                    i, e
                ))
            })?.into_neighbors();
            let base = i * stride as usize;
            let deg = nbrs.len().min(max_degree as usize);
            buf[base] = deg as u32;
            buf[base + HEADER_U32..base + HEADER_U32 + deg]
                .copy_from_slice(&nbrs[..deg]);
        }

        Ok(Self {
            data: Arc::new(buf),
            num_nodes: num_nodes as u32,
            max_degree,
            stride_u32: stride,
        })
    }

    /// Compute bidir bits for all nodes in parallel.
    pub fn compute_bidir(&mut self) {
        use rayon::prelude::*;

        let num_nodes = self.num_nodes as usize;
        let stride = self.stride_u32 as usize;
        let slice = self.data.as_slice();

        let bidir_results: Vec<u128> = (0..num_nodes)
            .into_par_iter()
            .map(|node| {
                let base = node * stride;
                let deg = slice[base] as usize;
                let nbrs = &slice[base + HEADER_U32..base + HEADER_U32 + deg];
                let mut bits: u128 = 0;
                for (idx, &nbr) in nbrs.iter().enumerate() {
                    let nb = nbr as usize * stride;
                    let nd = slice[nb] as usize;
                    let nn = &slice[nb + HEADER_U32..nb + HEADER_U32 + nd];
                    if nn.contains(&(node as u32)) {
                        bits |= 1u128 << idx;
                    }
                }
                bits
            })
            .collect();

        let buf = Arc::get_mut(&mut self.data).expect("compute_bidir requires exclusive Arc");
        for (i, bits) in bidir_results.into_iter().enumerate() {
            let base = i * stride;
            let w = bits_to_words(bits);
            buf[base + 1] = w[0];
            buf[base + 2] = w[1];
            buf[base + 3] = w[2];
            buf[base + 4] = w[3];
        }
    }

    #[inline]
    pub fn num_nodes(&self) -> usize { self.num_nodes as usize }

    #[inline]
    pub fn stride(&self) -> usize { self.stride_u32 as usize }

    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        self.data[i * self.stride_u32 as usize] as usize
    }

    #[inline]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let base = i * self.stride_u32 as usize;
        let deg = self.data[base] as usize;
        &self.data[base + HEADER_U32..base + HEADER_U32 + deg]
    }

    #[inline]
    pub fn contains_edge(&self, from: u32, to: u32) -> bool {
        self.neighbors(from as usize).contains(&to)
    }

    #[inline]
    pub fn bidir_bits(&self, i: usize) -> u128 {
        let base = i * self.stride_u32 as usize;
        let s = &*self.data;
        words_to_bits([s[base + 1], s[base + 2], s[base + 3], s[base + 4]])
    }

    #[inline]
    pub fn bidir_neighbors(&self, i: usize) -> BidirIter<'_> {
        let base = i * self.stride_u32 as usize;
        let s = &*self.data;
        let deg = s[base] as usize;
        let bits = words_to_bits([s[base + 1], s[base + 2], s[base + 3], s[base + 4]]);
        BidirIter {
            nbrs: &s[base + HEADER_U32..base + HEADER_U32 + deg],
            remaining_bits: bits,
        }
    }
}

pub struct BidirIter<'a> {
    nbrs: &'a [u32],
    remaining_bits: u128,
}

impl<'a> Iterator for BidirIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.remaining_bits == 0 { return None; }
        let idx = self.remaining_bits.trailing_zeros() as usize;
        self.remaining_bits &= self.remaining_bits - 1;
        Some(self.nbrs[idx])
    }
}

#[inline]
fn bits_to_words(v: u128) -> [u32; 4] {
    [v as u32, (v >> 32) as u32, (v >> 64) as u32, (v >> 96) as u32]
}

#[inline]
fn words_to_bits(w: [u32; 4]) -> u128 {
    w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_build() {
        let adj = vec![vec![1, 2, 3], vec![0, 2], vec![0, 1, 3], vec![0, 2]];
        let g = CsrGraph::from_adjacency_list(adj, 8);
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.degree(0), 3);
        assert_eq!(g.neighbors(0), &[1, 2, 3]);
        assert_eq!(g.neighbors(1), &[0, 2]);
    }

    #[test]
    fn test_contains_edge() {
        let adj = vec![vec![3, 1], vec![0], vec![], vec![1, 0]];
        let g = CsrGraph::from_adjacency_list(adj, 4);
        assert!(g.contains_edge(0, 1));
        assert!(g.contains_edge(0, 3));
        assert!(!g.contains_edge(0, 2));
        assert_eq!(g.neighbors(0), &[3, 1]); // order preserved
    }

    #[test]
    fn test_bidir() {
        let adj = vec![vec![1, 2], vec![0], vec![3], vec![2]];
        let mut g = CsrGraph::from_adjacency_list(adj, 4);
        g.compute_bidir();
        let b0: Vec<u32> = g.bidir_neighbors(0).collect();
        assert_eq!(b0, vec![1]);
        let b2: Vec<u32> = g.bidir_neighbors(2).collect();
        assert_eq!(b2, vec![3]);
    }

    #[test]
    fn test_alignment() {
        let adj = vec![vec![1]; 100];
        let g = CsrGraph::from_adjacency_list(adj, 32);
        let ptr = g.data.as_ptr() as usize;
        assert_eq!(ptr % 64, 0);
        assert_eq!(g.stride() * 4 % 64, 0);
    }

    #[test]
    fn test_stride() {
        assert_eq!(compute_stride(32), 48); // (8+32)*4=160 → 192 → 48
        assert_eq!(compute_stride(4), 16);  // (8+4)*4=48 → 64 → 16
    }

    #[test]
    fn test_clone_is_arc() {
        let adj = vec![vec![1, 2], vec![0]];
        let g1 = CsrGraph::from_adjacency_list(adj, 4);
        let g2 = g1.clone();
        assert_eq!(g1.data.as_ptr(), g2.data.as_ptr());
    }
}
