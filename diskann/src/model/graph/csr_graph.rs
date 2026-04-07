/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::io::{BufReader, BufWriter, Read, Write};

use super::node_slab_buffer::{BIDIR_OFFSET, HEADER_U32, NodeSlabBuffer};

const CACHE_LINE_BYTES: usize = 64;

#[inline]
const fn compute_stride(max_degree: usize) -> usize {
    let raw_bytes = (HEADER_U32 + max_degree) * 4;
    let aligned = (raw_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    aligned / 4
}

/// Cache-line aligned CSR graph backed by a `NodeSlabBuffer`.
///
/// Delegates all buffer management, per-node reader tracking, and write
/// queue logic to `NodeSlabBuffer`. This struct adds graph semantics:
/// degree, compressed_degree, neighbors, bidir bits, save/load.
pub struct CsrGraph {
    slab: NodeSlabBuffer,
    pub max_degree: u32,
}

unsafe impl Send for CsrGraph {}
unsafe impl Sync for CsrGraph {}

impl std::fmt::Debug for CsrGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsrGraph")
            .field("num_nodes", &self.slab.num_nodes())
            .field("max_degree", &self.max_degree)
            .field("stride_u32", &self.slab.stride())
            .finish()
    }
}

impl Clone for CsrGraph {
    fn clone(&self) -> Self {
        Self {
            slab: self.slab.clone(),
            max_degree: self.max_degree,
        }
    }
}

impl CsrGraph {
    /// Build from per-node neighbor lists.
    pub fn from_adjacency_list(adj: Vec<Vec<u32>>, max_degree: u32) -> Self {
        let num_nodes = adj.len();
        let stride = compute_stride(max_degree as usize);
        let slab = NodeSlabBuffer::new(num_nodes, stride);

        // Initial construction: all neighbors as "rest" (compressed_degree = 0).
        for (i, nbrs) in adj.iter().enumerate() {
            let deg = nbrs.len().min(max_degree as usize);
            unsafe {
                slab.write_node_unchecked(i, &[], &nbrs[..deg]);
            }
        }

        Self { slab, max_degree }
    }

    /// Build from InMemoryGraph (consumes via into_inner).
    pub fn from_inmem_graph(
        graph: super::InMemoryGraph,
        max_degree: u32,
    ) -> crate::common::ANNResult<Self> {
        let num_nodes = graph.size();
        let stride = compute_stride(max_degree as usize);
        let slab = NodeSlabBuffer::new(num_nodes, stride);

        for (i, lock) in graph.final_graph.into_iter().enumerate() {
            let nbrs = lock
                .into_inner()
                .map_err(|e| {
                    crate::common::ANNError::log_lock_poison_error(format!(
                        "CsrGraph::from_inmem_graph: poisoned at node {}: {}",
                        i, e
                    ))
                })?
                .into_neighbors();
            let deg = nbrs.len().min(max_degree as usize);
            unsafe {
                slab.write_node_unchecked(i, &[], &nbrs[..deg]);
            }
        }

        Ok(Self { slab, max_degree })
    }

    /// Compute bidir bits (exclusive access, no readers expected).
    pub fn compute_bidir(&mut self) {
        use rayon::prelude::*;
        let n = self.slab.num_nodes();
        let stride = self.slab.stride();
        let slice = self.slab.as_slice();

        let results: Vec<u128> = (0..n)
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

        for (i, bits) in results.into_iter().enumerate() {
            let w = bits_to_words(bits);
            unsafe {
                self.slab.write_raw(i, BIDIR_OFFSET, &w);
            }
        }
    }

    // ── Direct read accessors (zero-overhead, no atomics) ────────────────────

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.slab.num_nodes()
    }
    #[inline]
    pub fn stride(&self) -> usize {
        self.slab.stride()
    }

    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        self.slab.slot(i)[0] as usize
    }

    #[inline]
    pub fn compressed_degree(&self, i: usize) -> usize {
        self.slab.slot(i)[1] as usize
    }

    #[inline]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let s = self.slab.slot(i);
        let deg = s[0] as usize;
        &s[HEADER_U32..HEADER_U32 + deg]
    }

    /// Prefetch node's slot into L1 cache.
    #[inline]
    pub fn prefetch_node(&self, i: usize) {
        vector::prefetch_vector(self.slab.slot(i));
    }

    #[inline]
    pub fn compressed_neighbors(&self, i: usize) -> &[u32] {
        let s = self.slab.slot(i);
        let cd = s[1] as usize;
        &s[HEADER_U32..HEADER_U32 + cd]
    }

    #[inline]
    pub fn contains_edge(&self, from: u32, to: u32) -> bool {
        self.neighbors(from as usize).contains(&to)
    }

    #[inline]
    pub fn bidir_bits(&self, i: usize) -> u128 {
        let s = self.slab.slot(i);
        words_to_bits([
            s[BIDIR_OFFSET],
            s[BIDIR_OFFSET + 1],
            s[BIDIR_OFFSET + 2],
            s[BIDIR_OFFSET + 3],
        ])
    }

    #[inline]
    pub fn bidir_neighbors(&self, i: usize) -> BidirIter<'_> {
        let s = self.slab.slot(i);
        let deg = s[0] as usize;
        let bits = words_to_bits([
            s[BIDIR_OFFSET],
            s[BIDIR_OFFSET + 1],
            s[BIDIR_OFFSET + 2],
            s[BIDIR_OFFSET + 3],
        ]);
        BidirIter {
            nbrs: &s[HEADER_U32..HEADER_U32 + deg],
            remaining_bits: bits,
        }
    }

    // ── Writer interface (delegates to NodeSlabBuffer) ───────────────────────

    /// Enqueue a node update via the bounded write queue.
    pub fn update_node(&self, node: usize, compressed: &[u32], rest: &[u32]) {
        self.slab.write_node(node, compressed, rest);
    }

    /// Flush all pending writes.
    pub fn flush(&self) {
        self.slab.flush();
    }

    /// Build-phase batch reorder: direct write, no concurrent readers.
    ///
    /// # Safety
    /// No concurrent readers on `node`. Different threads must write
    /// different nodes.
    pub unsafe fn set_neighbors_reordered_unchecked(
        &self,
        node: usize,
        compressed: &[u32],
        rest: &[u32],
    ) {
        unsafe { self.slab.write_node_unchecked(node, compressed, rest) };
    }

    // ── Per-node guarded read (for concurrent update scenarios) ──────────────

    /// Acquire a per-node read guard. Use for concurrent read+write safety.
    /// For the search hot path (no concurrent writes), use direct accessors.
    #[inline]
    pub fn read_node(&self, i: usize) -> super::node_slab_buffer::SegmentGuard<'_> {
        self.slab.read_node(i)
    }

    // ── IO ───────────────────────────────────────────────────────────────────

    /// Save CsrGraph to a file: header + buffer data.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);
        // Header: [num_nodes, max_degree, stride_u32, reserved]
        let header: [u32; 4] = [
            self.slab.num_nodes() as u32,
            self.max_degree,
            self.slab.stride() as u32,
            0,
        ];
        let hdr_bytes = unsafe { std::slice::from_raw_parts(header.as_ptr() as *const u8, 16) };
        w.write_all(hdr_bytes)?;
        self.slab.save_to(&mut w)?;
        w.flush()
    }

    /// Load CsrGraph from a file.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let mut r = BufReader::new(f);
        let mut hdr_bytes = [0u8; 16];
        r.read_exact(&mut hdr_bytes)?;
        let header: [u32; 4] = unsafe { std::mem::transmute(hdr_bytes) };
        let num_nodes = header[0] as usize;
        let max_degree = header[1];
        let stride_u32 = header[2] as usize;
        let slab = NodeSlabBuffer::load_from(&mut r, num_nodes, stride_u32)?;
        Ok(Self { slab, max_degree })
    }
}

// ── BidirIter ────────────────────────────────────────────────────────────────

pub struct BidirIter<'a> {
    nbrs: &'a [u32],
    remaining_bits: u128,
}

impl<'a> Iterator for BidirIter<'a> {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.remaining_bits == 0 {
            return None;
        }
        let idx = self.remaining_bits.trailing_zeros() as usize;
        self.remaining_bits &= self.remaining_bits - 1;
        Some(self.nbrs[idx])
    }
}

#[inline]
fn bits_to_words(v: u128) -> [u32; 4] {
    [
        v as u32,
        (v >> 32) as u32,
        (v >> 64) as u32,
        (v >> 96) as u32,
    ]
}

#[inline]
fn words_to_bits(w: [u32; 4]) -> u128 {
    w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let adj = vec![vec![1, 2, 3], vec![0, 2], vec![0, 1, 3], vec![0, 2]];
        let g = CsrGraph::from_adjacency_list(adj, 32);
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.degree(0), 3);
        assert_eq!(g.neighbors(0), &[1, 2, 3]);
        assert_eq!(g.compressed_degree(0), 0);
    }

    #[test]
    fn test_bidir() {
        let adj = vec![vec![1, 2], vec![0], vec![3], vec![2]];
        let mut g = CsrGraph::from_adjacency_list(adj, 32);
        g.compute_bidir();
        let b0: Vec<u32> = g.bidir_neighbors(0).collect();
        assert_eq!(b0, vec![1]);
    }

    #[test]
    fn test_update_node() {
        let adj = vec![vec![1, 2, 3], vec![0]];
        let g = CsrGraph::from_adjacency_list(adj, 32);
        g.update_node(0, &[3], &[1, 2]);
        assert_eq!(g.compressed_degree(0), 1);
        assert_eq!(g.compressed_neighbors(0), &[3]);
        assert_eq!(g.neighbors(0), &[3, 1, 2]);
    }

    #[test]
    fn test_unsafe_reorder() {
        let adj = vec![vec![1, 2, 3], vec![0, 2], vec![0, 1]];
        let g = CsrGraph::from_adjacency_list(adj, 32);
        unsafe {
            g.set_neighbors_reordered_unchecked(0, &[3], &[1, 2]);
        }
        assert_eq!(g.compressed_degree(0), 1);
        assert_eq!(g.neighbors(0), &[3, 1, 2]);
    }

    #[test]
    fn test_concurrent_read_update() {
        use std::sync::Arc;
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let g = Arc::new(CsrGraph::from_adjacency_list(adj, 32));

        let g2 = g.clone();
        let reader = std::thread::spawn(move || {
            for _ in 0..1000 {
                let _guard = g2.read_node(0);
                let _ = g2.neighbors(0);
            }
        });

        for _ in 0..100 {
            g.update_node(0, &[2], &[1]);
        }

        reader.join().unwrap();
        g.flush();
        assert_eq!(g.compressed_degree(0), 1);
        assert_eq!(g.neighbors(0), &[2, 1]);
    }

    #[test]
    fn test_save_load() {
        let adj = vec![vec![1, 2, 3], vec![0, 2], vec![0, 1, 3], vec![0, 2]];
        let g = CsrGraph::from_adjacency_list(adj, 32);
        unsafe {
            g.set_neighbors_reordered_unchecked(0, &[3], &[1, 2]);
        }

        let dir = std::env::temp_dir().join("csr_graph_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.csrgraph");
        g.save(&path).expect("save failed");

        let loaded = CsrGraph::load(&path).expect("load failed");
        assert_eq!(loaded.num_nodes(), 4);
        assert_eq!(loaded.max_degree, 32);
        assert_eq!(loaded.compressed_degree(0), 1);
        assert_eq!(loaded.neighbors(0), &[3, 1, 2]);
        assert_eq!(loaded.neighbors(1), &[0, 2]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
