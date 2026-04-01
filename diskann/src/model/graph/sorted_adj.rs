/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Flat sorted adjacency list — CSR layout with sorted neighbor slices.
///
/// Each node's neighbors are stored sorted, enabling O(log d) `contains`
/// via binary search. Uses a single flat allocation (no per-node `Vec`).
///
/// Built from `Vec<Vec<u32>>` by sorting + flattening in one pass.
#[derive(Clone, Debug)]
pub struct SortedAdjacencyList {
    offsets: Vec<u32>,
    data: Vec<u32>,
}

impl SortedAdjacencyList {
    /// Build from per-node neighbor lists. Each list is sorted during construction.
    pub fn from_vecs(mut adj: Vec<Vec<u32>>) -> Self {
        let n = adj.len();
        let total: usize = adj.iter().map(|v| v.len()).sum();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut data = Vec::with_capacity(total);
        offsets.push(0u32);
        for nbrs in &mut adj {
            nbrs.sort_unstable();
            data.extend_from_slice(nbrs);
            offsets.push(data.len() as u32);
        }
        Self { offsets, data }
    }

    /// Build from a `CsrGraph` by copying + sorting each node's neighbors.
    /// Single flat allocation — no per-node `Vec`.
    pub fn from_csr(csr: &super::CsrGraph) -> Self {
        let n = csr.num_nodes();
        let offsets = csr.offsets.as_ref().clone();
        let mut data = csr.neighbors.as_ref().clone();
        for i in 0..n {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            data[start..end].sort_unstable();
        }
        Self { offsets, data }
    }

    /// Sorted neighbor slice for node `i`.
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.data[start..end]
    }

    /// O(log d) check whether node `i` has neighbor `id`.
    #[inline]
    pub fn contains(&self, i: usize, id: u32) -> bool {
        self.neighbors(i).binary_search(&id).is_ok()
    }

    /// Number of nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Degree of node `i`.
    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }
}
