/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::sync::Arc;

/// Lock-free CSR (Compressed Sparse Row) graph.
///
/// - `offsets[i]` .. `offsets[i+1]` → range in `neighbors` for node `i`
/// - `offsets.len()` == `num_nodes + 1`
///
/// Both `offsets` and `neighbors` are wrapped in `Arc` so that cloning a
/// `CsrGraph` is O(1) reference-count increment — no data is copied.
/// This makes it cheap to share across `DiskANNBuildResult`, `StagedDiskANN`,
/// and the clustering `Arc<CsrGraph>`.
///
/// `max_degree` is precomputed during construction to avoid a full scan
/// when the value is needed later (e.g. for `CompressedGraph` allocation).
/// This duplicates information derivable from `offsets`, but the O(n) scan
/// is measurable at 30K+ nodes and the field costs only 4 bytes.
#[derive(Clone, Debug)]
pub struct CsrGraph {
    pub offsets: Arc<Vec<u32>>,
    pub neighbors: Arc<Vec<u32>>,
    /// Precomputed maximum out-degree across all nodes.
    pub max_degree: u32,
}

impl CsrGraph {
    /// Create from raw parts. `max_degree` is precomputed by the caller.
    pub fn from_raw(offsets: Vec<u32>, neighbors: Vec<u32>, max_degree: u32) -> Self {
        Self {
            offsets: Arc::new(offsets),
            neighbors: Arc::new(neighbors),
            max_degree,
        }
    }

    /// Number of nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Neighbor slice for node `i`.
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.neighbors[start..end]
    }

    /// Does edge `from → to` exist?
    #[inline]
    pub fn contains_edge(&self, from: u32, to: u32) -> bool {
        self.neighbors(from as usize).contains(&to)
    }

    /// Degree of node `i`.
    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }

    /// Build from a `Vec<Vec<u32>>` adjacency list (consumes it).
    ///
    /// Computes `max_degree` during the single construction pass.
    pub fn from_adjacency_list(adj: Vec<Vec<u32>>) -> Self {
        let n = adj.len();
        let total: usize = adj.iter().map(|v| v.len()).sum();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut neighbors = Vec::with_capacity(total);
        let mut max_deg: u32 = 0;
        offsets.push(0u32);
        for nbrs in adj {
            let deg = nbrs.len() as u32;
            if deg > max_deg {
                max_deg = deg;
            }
            neighbors.extend_from_slice(&nbrs);
            offsets.push(neighbors.len() as u32);
        }
        Self {
            offsets: Arc::new(offsets),
            neighbors: Arc::new(neighbors),
            max_degree: max_deg,
        }
    }
}
