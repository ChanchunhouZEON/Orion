/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! **PhasedGraph** — cache-line aligned, concurrency-safe graph for two-phase
//! staged search.
//!
//! Per-node slot layout (fixed stride, cache-line aligned):
//! ```text
//! ┌─── Header (4 × u32 = 16 bytes) ─────────────────────────────────────────────────┐
//! │ degree (R) │ extra_count (E) │ local_count (L) │ reserved                       │
//! ├─── Data ────────────────────────────────────────────────────────────────────────┤
//! │ local_neighbors  [u32 × L]       ← navigation + reranking                       │
//! │ remote_neighbors [u32 × (R − L)] ← navigation only                              │
//! │ extra_candidates [u32 × E]       ← reranking only(starts at `start+ max_degree`)│
//! │ padding to max_extra * u32                                                      │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Backed by `AlignedBoxWithSlice<u32>` with per-node reader tracking
//! and bounded write queue, mirroring `NodeSlabBuffer`'s concurrency model.

use diskann::common::ANNResult;
use diskann::common::AlignedBoxWithSlice;
use std::collections::VecDeque;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

const CACHE_LINE_BYTES: usize = 64;
const HEADER_U32: usize = 4;
const DEFAULT_WRITE_QUEUE_CAPACITY: usize = 64;

#[inline]
const fn compute_stride(max_data_u32: usize) -> usize {
    let raw_bytes = (HEADER_U32 + max_data_u32) * 4;
    let aligned = (raw_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    aligned / 4
}

/// Queued write for a single node.
struct PhasedNodeWrite {
    node: usize,
    local: Vec<u32>,
    remote: Vec<u32>,
    extra: Vec<u32>,
}

/// Two-phase graph backed by a cache-line aligned slab with per-node
/// reader tracking and bounded write queue.
pub struct PhasedGraph {
    buffer: AlignedBoxWithSlice<u32>,
    node_readers: Box<[AtomicUsize]>,
    write_blocked: AtomicBool,
    write_queue: Mutex<VecDeque<PhasedNodeWrite>>,
    write_queue_capacity: usize,
    num_nodes: usize,
    stride: usize,
    pub max_degree: u32,
    /// Global t parameter (minimum local count before common promotion).
    pub base_local_count: u32,
}

impl std::fmt::Debug for PhasedGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhasedGraph")
            .field("num_nodes", &self.num_nodes)
            .field("max_degree", &self.max_degree)
            .field("base_local_count", &self.base_local_count)
            .field("stride_u32", &self.stride)
            .finish()
    }
}

unsafe impl Send for PhasedGraph {}
unsafe impl Sync for PhasedGraph {}

impl PhasedGraph {
    // ── Construction ─────────────────────────────────────────────────────

    /// Allocate an empty PhasedGraph with the given capacity.
    fn allocate(
        num_nodes: usize,
        max_data_u32: usize,
        max_degree: u32,
        base_local_count: u32,
    ) -> Self {
        let stride = compute_stride(max_data_u32);
        let total = num_nodes * stride;
        let buffer =
            AlignedBoxWithSlice::<u32>::new(total, CACHE_LINE_BYTES).expect("PhasedGraph alloc");
        let node_readers: Vec<AtomicUsize> = (0..num_nodes).map(|_| AtomicUsize::new(0)).collect();
        Self {
            buffer,
            node_readers: node_readers.into_boxed_slice(),
            write_blocked: AtomicBool::new(false),
            write_queue: Mutex::new(VecDeque::new()),
            write_queue_capacity: DEFAULT_WRITE_QUEUE_CAPACITY,
            num_nodes,
            stride,
            max_degree,
            base_local_count,
        }
    }

    /// Build from pre-computed partitions (from extract_graph_and_candidates).
    ///
    /// Each element: `(local_ids, remote_ids, extra_ids)`.
    /// Writes directly into the slab — no bidir computation, no candidate
    /// filtering, no distance information needed.
    pub fn build_from_partitions(
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        max_degree: u32,
        max_extra: usize,
    ) -> Self {
        use rayon::prelude::*;

        let num_nodes = partitions.len();
        let max_data = max_degree as usize + max_extra;
        let pg = Self::allocate(num_nodes, max_data, max_degree, 0);

        partitions
            .par_iter()
            .enumerate()
            .for_each(|(node, (local, remote, extra))| unsafe {
                pg.write_node_unchecked(node, local, remote, extra);
            });

        pg
    }

    /// Import STAG v3 directly into the final slab, retaining only one node's IDs.
    /// Counts and IDs are validated before the unsafe slab writer is called.
    pub fn load_staged<P: AsRef<std::path::Path>>(
        path: P,
        expected_nodes: usize,
        expected_degree: u32,
    ) -> ANNResult<Self> {
        use std::io::{Error, ErrorKind};
        let bad = |message: &str| Error::new(ErrorKind::InvalidData, message);
        let file = std::fs::File::open(path)?;
        let file_bytes = file.metadata()?.len();
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        fn word<R: Read>(r: &mut R) -> std::io::Result<u32> {
            let mut b = [0; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        }
        if word(&mut reader)? != 0x53544147 || word(&mut reader)? != 3 {
            return Err(bad("expected STAG v3").into());
        }
        let n = word(&mut reader)? as usize;
        let degree = word(&mut reader)?;
        let extra = word(&mut reader)? as usize;
        let entry = word(&mut reader)? as usize;
        if n == 0 || n != expected_nodes || entry >= n || degree != expected_degree {
            return Err(bad("staged node count, degree, or entry mismatch").into());
        }
        if degree as usize > n || extra > n || file_bytes < 24 + n as u64 * 12 {
            return Err(bad("invalid staged capacity or truncated records").into());
        }
        let max_data = (degree as usize)
            .checked_add(extra)
            .ok_or_else(|| bad("degree overflow"))?;
        let stride = compute_stride(max_data);
        let total = n
            .checked_mul(stride)
            .ok_or_else(|| bad("graph size overflow"))?;
        let pg = Self {
            buffer: AlignedBoxWithSlice::new(total, CACHE_LINE_BYTES)?,
            node_readers: (0..n)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            write_blocked: AtomicBool::new(false),
            write_queue: Mutex::new(VecDeque::new()),
            write_queue_capacity: DEFAULT_WRITE_QUEUE_CAPACITY,
            num_nodes: n,
            stride,
            max_degree: degree,
            base_local_count: 0,
        };
        let mut ids = Vec::new();
        for node in 0..n {
            let lc = word(&mut reader)? as usize;
            let rc = word(&mut reader)? as usize;
            let ec = word(&mut reader)? as usize;
            let neighbors = lc.checked_add(rc).ok_or_else(|| bad("degree overflow"))?;
            if neighbors > degree as usize || ec > extra {
                return Err(bad("staged record exceeds header capacities").into());
            }
            let count = neighbors
                .checked_add(ec)
                .ok_or_else(|| bad("degree overflow"))?;
            ids.clear();
            for _ in 0..count {
                let id = word(&mut reader)?;
                if id as usize >= n {
                    return Err(bad("staged neighbor ID outside graph").into());
                }
                ids.push(id);
            }
            unsafe {
                pg.write_node_unchecked(node, &ids[..lc], &ids[lc..neighbors], &ids[neighbors..]);
            }
            if node > 0 && node % 10_000_000 == 0 {
                log::info!("Imported {node}/{n} staged nodes");
            }
        }
        let mut tail = [0];
        if reader.read(&mut tail)? != 0 {
            return Err(bad("trailing bytes after staged graph").into());
        }
        Ok(pg)
    }

    // ── Direct read accessors (zero overhead, no atomics) ───────────────

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Raw slab as bytes — used by benchmarks to `mlock` the graph
    /// pages so they cannot be swapped between warmup and the timed
    /// sweep.
    #[inline]
    pub fn buffer_bytes(&self) -> &[u8] {
        let n = self.buffer.len() * std::mem::size_of::<u32>();
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr() as *const u8, n) }
    }

    #[inline]
    fn slot(&self, i: usize) -> &[u32] {
        let base = i * self.stride;
        &self.buffer[base..base + self.stride]
    }

    /// Total graph degree.
    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        self.slot(i)[0] as usize
    }

    /// Number of extra candidates (not in graph).
    #[inline]
    pub fn extra_count(&self, i: usize) -> usize {
        self.slot(i)[1] as usize
    }

    /// Per-node local count (base t + promoted common).
    #[inline]
    pub fn local_count(&self, i: usize) -> usize {
        self.slot(i)[2] as usize
    }

    /// All graph neighbors = local + remote. Used for **navigation**.
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let deg = s[0] as usize;
        &s[HEADER_U32..HEADER_U32 + deg]
    }

    /// Local neighbors (candidate-set-confirmed). Navigation + reranking.
    #[inline]
    pub fn local_neighbors(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let lc = s[2] as usize;
        &s[HEADER_U32..HEADER_U32 + lc]
    }

    /// Remote neighbors (long-range shortcuts). Navigation only.
    #[inline]
    pub fn remote_neighbors(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let deg = s[0] as usize;
        let lc = s[2] as usize;
        &s[HEADER_U32 + lc..HEADER_U32 + deg]
    }

    /// Extra candidates (in candidate set, not in graph). Reranking only.
    #[inline]
    pub fn extra_candidates(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let deg = s[0] as usize;
        let extra = s[1] as usize;
        &s[HEADER_U32 + deg..HEADER_U32 + deg + extra]
    }

    /// Reranking slices: (local_neighbors, extra_candidates).
    #[inline]
    pub fn rerank_candidates(&self, i: usize) -> (&[u32], &[u32]) {
        (self.local_neighbors(i), self.extra_candidates(i))
    }

    /// Total reranking candidate count = local + extra.
    #[inline]
    pub fn rerank_count(&self, i: usize) -> usize {
        let s = self.slot(i);
        s[2] as usize + s[1] as usize
    }

    /// Check if edge (from → to) exists in the graph.
    #[inline]
    pub fn contains_edge(&self, from: u32, to: u32) -> bool {
        self.neighbors(from as usize).contains(&to)
    }

    /// Prefetch node slot into L1 cache.
    #[inline]
    pub fn prefetch_node(&self, i: usize) {
        vector::prefetch_vector(self.slot(i));
    }

    // ── Guarded read (per-node reader tracking) ─────────────────────────

    /// Acquire a per-node read guard. Spins during stop-the-world flush.
    #[inline]
    pub fn read_node(&self, i: usize) -> PhasedSlotGuard<'_> {
        loop {
            if !self.write_blocked.load(Ordering::Acquire) {
                self.node_readers[i].fetch_add(1, Ordering::Acquire);
                if !self.write_blocked.load(Ordering::Acquire) {
                    return PhasedSlotGuard {
                        graph: self,
                        node: i,
                    };
                }
                self.node_readers[i].fetch_sub(1, Ordering::Release);
            }
            std::hint::spin_loop();
        }
    }

    // ── Write interface ─────────────────────────────────────────────────

    /// Enqueue a node update via the bounded write queue.
    pub fn write_node(&self, node: usize, local: &[u32], remote: &[u32], extra: &[u32]) {
        let mut queue = self.write_queue.lock().unwrap();
        self.try_flush_queue(&mut queue);

        if self.node_readers[node].load(Ordering::Acquire) == 0 {
            self.apply_write(node, local, remote, extra);
            return;
        }

        queue.push_back(PhasedNodeWrite {
            node,
            local: local.to_vec(),
            remote: remote.to_vec(),
            extra: extra.to_vec(),
        });

        if queue.len() >= self.write_queue_capacity {
            self.stop_the_world_flush(&mut queue);
        }
    }

    /// Flush all pending writes.
    pub fn flush(&self) {
        let mut queue = self.write_queue.lock().unwrap();
        if queue.is_empty() {
            return;
        }
        self.try_flush_queue(&mut queue);
        if !queue.is_empty() {
            self.stop_the_world_flush(&mut queue);
        }
    }

    /// Build-phase direct write. No concurrent readers on `node`.
    ///
    /// # Safety
    /// Different threads must target different nodes.
    pub unsafe fn write_node_unchecked(
        &self,
        node: usize,
        local: &[u32],
        remote: &[u32],
        extra: &[u32],
    ) {
        self.apply_write(node, local, remote, extra);
    }

    fn apply_write(&self, node: usize, local: &[u32], remote: &[u32], extra: &[u32]) {
        let base = node * self.stride;
        let degree = local.len() + remote.len();
        let ptr = self.buffer.as_ptr() as *mut u32;
        unsafe {
            // Header
            std::ptr::write(ptr.add(base), degree as u32);
            std::ptr::write(ptr.add(base + 1), extra.len() as u32);
            std::ptr::write(ptr.add(base + 2), local.len() as u32);
            std::ptr::write(ptr.add(base + 3), 0); // reserved
                                                   // Data: local | remote | extra
            let data_start = base + HEADER_U32;
            std::ptr::copy_nonoverlapping(local.as_ptr(), ptr.add(data_start), local.len());
            std::ptr::copy_nonoverlapping(
                remote.as_ptr(),
                ptr.add(data_start + local.len()),
                remote.len(),
            );
            std::ptr::copy_nonoverlapping(
                extra.as_ptr(),
                ptr.add(data_start + degree),
                extra.len(),
            );
        }
    }

    fn try_flush_queue(&self, queue: &mut VecDeque<PhasedNodeWrite>) {
        let mut i = 0;
        while i < queue.len() {
            let node = queue[i].node;
            if self.node_readers[node].load(Ordering::Acquire) == 0 {
                let req = queue.remove(i).unwrap();
                self.apply_write(req.node, &req.local, &req.remote, &req.extra);
            } else {
                i += 1;
            }
        }
    }

    fn stop_the_world_flush(&self, queue: &mut VecDeque<PhasedNodeWrite>) {
        self.write_blocked.store(true, Ordering::Release);
        for reader in self.node_readers.iter() {
            while reader.load(Ordering::Acquire) > 0 {
                std::hint::spin_loop();
            }
        }
        while let Some(req) = queue.pop_front() {
            self.apply_write(req.node, &req.local, &req.remote, &req.extra);
        }
        self.write_blocked.store(false, Ordering::Release);
    }

    // ── IO ──────────────────────────────────────────────────────────────

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> ANNResult<()> {
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);
        // File header: [num_nodes, max_degree, stride, base_local_count]
        let header: [u32; 4] = [
            self.num_nodes as u32,
            self.max_degree,
            self.stride as u32,
            self.base_local_count,
        ];
        let hdr_bytes = unsafe { std::slice::from_raw_parts(header.as_ptr() as *const u8, 16) };
        w.write_all(hdr_bytes)?;
        // Slab data
        let data_bytes = unsafe {
            std::slice::from_raw_parts(self.buffer.as_ptr() as *const u8, self.buffer.len() * 4)
        };
        w.write_all(data_bytes)?;
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> ANNResult<Self> {
        let f = std::fs::File::open(path)?;
        let mut r = BufReader::new(f);
        let mut hdr_bytes = [0u8; 16];
        r.read_exact(&mut hdr_bytes)?;
        let header: [u32; 4] = unsafe { std::mem::transmute(hdr_bytes) };
        let num_nodes = header[0] as usize;
        let max_degree = header[1];
        let stride = header[2] as usize;
        let base_local_count = header[3];

        let total = num_nodes * stride;
        let mut pg = Self {
            buffer: AlignedBoxWithSlice::<u32>::new(total, CACHE_LINE_BYTES)
                .expect("PhasedGraph load alloc"),
            node_readers: (0..num_nodes)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            write_blocked: AtomicBool::new(false),
            write_queue: Mutex::new(VecDeque::new()),
            write_queue_capacity: DEFAULT_WRITE_QUEUE_CAPACITY,
            num_nodes,
            stride,
            max_degree,
            base_local_count,
        };
        let data_bytes =
            unsafe { std::slice::from_raw_parts_mut(pg.buffer.as_mut_ptr() as *mut u8, total * 4) };
        r.read_exact(data_bytes)?;
        Ok(pg)
    }

    // ── Stats ───────────────────────────────────────────────────────────

    pub fn print_stats(&self) {
        let n = self.num_nodes;
        if n == 0 {
            return;
        }
        let total_deg: usize = (0..n).map(|i| self.degree(i)).sum();
        let total_extra: usize = (0..n).map(|i| self.extra_count(i)).sum();
        let total_local: usize = (0..n).map(|i| self.local_count(i)).sum();
        let total_rerank: usize = (0..n).map(|i| self.rerank_count(i)).sum();

        println!("PhasedGraph ({} nodes):", n);
        println!(
            "  R={}, base_local={}, stride={} u32",
            self.max_degree, self.base_local_count, self.stride
        );
        println!("  Avg degree:       {:.1}", total_deg as f64 / n as f64);
        println!(
            "  Avg local:        {:.1} (base + promoted)",
            total_local as f64 / n as f64
        );
        println!("  Avg extra:        {:.1}", total_extra as f64 / n as f64);
        println!(
            "  Avg rerank:       {:.1} (local+extra)",
            total_rerank as f64 / n as f64
        );
        println!(
            "  Memory:           {:.1} MB",
            (self.buffer.len() * 4) as f64 / 1_048_576.0,
        );
    }
}

impl Clone for PhasedGraph {
    fn clone(&self) -> Self {
        let src = self.buffer.as_slice();
        let mut buf =
            AlignedBoxWithSlice::<u32>::new(src.len(), CACHE_LINE_BYTES).expect("clone alloc");
        buf.as_mut_slice().copy_from_slice(src);
        let node_readers: Vec<AtomicUsize> =
            (0..self.num_nodes).map(|_| AtomicUsize::new(0)).collect();
        Self {
            buffer: buf,
            node_readers: node_readers.into_boxed_slice(),
            write_blocked: AtomicBool::new(false),
            write_queue: Mutex::new(VecDeque::new()),
            write_queue_capacity: DEFAULT_WRITE_QUEUE_CAPACITY,
            num_nodes: self.num_nodes,
            stride: self.stride,
            max_degree: self.max_degree,
            base_local_count: self.base_local_count,
        }
    }
}

// ── PhasedSlotGuard ─────────────────────────────────────────────────────────

/// RAII guard for a single node slot. Decrements the per-node reader
/// count on drop, allowing queued writes to proceed.
pub struct PhasedSlotGuard<'a> {
    graph: &'a PhasedGraph,
    node: usize,
}

impl<'a> PhasedSlotGuard<'a> {
    #[inline]
    pub fn slot(&self) -> &[u32] {
        self.graph.slot(self.node)
    }

    #[inline]
    pub fn neighbors(&self) -> &[u32] {
        self.graph.neighbors(self.node)
    }

    #[inline]
    pub fn local_neighbors(&self) -> &[u32] {
        self.graph.local_neighbors(self.node)
    }

    #[inline]
    pub fn rerank_candidates(&self) -> (&[u32], &[u32]) {
        self.graph.rerank_candidates(self.node)
    }
}

impl<'a> Drop for PhasedSlotGuard<'a> {
    fn drop(&mut self) {
        self.graph.node_readers[self.node].fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phased_graph_layout() {
        let partitions = vec![
            (vec![1, 2, 3], vec![], vec![5]), // node 0: all local, extra=[5]
            (vec![0, 2], vec![], vec![]),     // node 1
            (vec![0, 1, 3], vec![], vec![]),  // node 2
            (vec![0, 2], vec![], vec![]),     // node 3
        ];
        let pg = PhasedGraph::build_from_partitions(&partitions, 4, 4);

        // Node 0: degree=3, nbr 3 promoted → local_count=3, extra=[5]
        assert_eq!(pg.degree(0), 3);
        assert_eq!(pg.local_count(0), 3);
        assert_eq!(pg.extra_count(0), 1);
        assert_eq!(pg.neighbors(0), &[1, 2, 3]);
        assert_eq!(pg.local_neighbors(0), &[1, 2, 3]);
        assert_eq!(pg.remote_neighbors(0), &[] as &[u32]);
        assert_eq!(pg.extra_candidates(0), &[5]);
        assert_eq!(pg.rerank_count(0), 4);

        // Node 1: degree=2, t=2, all local
        assert_eq!(pg.degree(1), 2);
        assert_eq!(pg.local_count(1), 2);
        assert_eq!(pg.extra_candidates(1), &[] as &[u32]);

        assert!(pg.contains_edge(0, 1));
        assert!(pg.contains_edge(1, 0));
        assert!(!pg.contains_edge(0, 5));

        pg.print_stats();
    }

    #[test]
    fn test_write_node_and_guard() {
        let partitions = vec![(vec![1], vec![], vec![]), (vec![0], vec![], vec![])];
        let pg = PhasedGraph::build_from_partitions(&partitions, 4, 4);

        assert_eq!(pg.neighbors(0), &[1]);

        // Write new neighbors via the concurrent path.
        pg.write_node(0, &[1], &[], &[5, 6]);
        assert_eq!(pg.local_neighbors(0), &[1]);
        assert_eq!(pg.extra_candidates(0), &[5, 6]);

        // Guard-based read.
        let guard = pg.read_node(0);
        assert_eq!(guard.neighbors(), &[1]);
        assert_eq!(guard.local_neighbors(), &[1]);
        let (local, extra) = guard.rerank_candidates();
        assert_eq!(local, &[1]);
        assert_eq!(extra, &[5, 6]);
        drop(guard);
    }

    #[test]
    fn test_save_load() {
        let partitions = vec![
            (vec![1], vec![2], vec![]),
            (vec![0], vec![2], vec![]),
            (vec![0], vec![1], vec![]),
        ];
        let pg = PhasedGraph::build_from_partitions(&partitions, 4, 4);

        let dir = std::env::temp_dir().join("phased_graph_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.pgraph");
        pg.save(&path).expect("save failed");

        let loaded = PhasedGraph::load(&path).expect("load failed");
        assert_eq!(loaded.num_nodes(), 3);
        assert_eq!(loaded.max_degree, 4);
        assert_eq!(loaded.degree(0), 2);
        assert_eq!(loaded.neighbors(0), pg.neighbors(0));
        assert_eq!(loaded.extra_candidates(0), pg.extra_candidates(0));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_clone() {
        let partitions = vec![(vec![1], vec![], vec![]), (vec![0], vec![], vec![])];
        let pg = PhasedGraph::build_from_partitions(&partitions, 4, 4);

        let pg2 = pg.clone();
        assert_eq!(pg2.neighbors(0), pg.neighbors(0));
        assert_eq!(pg2.degree(1), pg.degree(1));

        // Writes to clone don't affect original.
        pg2.write_node(0, &[1], &[], &[9]);
        assert_eq!(pg2.extra_candidates(0), &[9]);
        assert_eq!(pg.extra_candidates(0), &[] as &[u32]);
    }
}

#[cfg(test)]
mod staged_validation_tests {
    use super::*;
    #[test]
    fn streamed_import_rejects_bad_counts_ids_truncation_and_trailing_bytes() {
        let p = std::env::temp_dir().join(format!("orion-staged-invalid-{}", std::process::id()));
        let valid = [0x53544147u32, 3, 2, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0];
        let write = |words: &[u32]| {
            std::fs::write(
                &p,
                words
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        write(&valid);
        assert!(PhasedGraph::load_staged(&p, 2, 1).is_ok());
        assert!(PhasedGraph::load_staged(&p, 3, 1).is_err());
        assert!(PhasedGraph::load_staged(&p, 2, 2).is_err());
        let mut bad = valid;
        bad[6] = 2;
        write(&bad);
        assert!(PhasedGraph::load_staged(&p, 2, 1).is_err());
        let mut bad = valid;
        bad[9] = 2;
        write(&bad);
        assert!(PhasedGraph::load_staged(&p, 2, 1).is_err());
        write(&valid[..13]);
        assert!(PhasedGraph::load_staged(&p, 2, 1).is_err());
        let mut bad = valid.to_vec();
        bad.push(0);
        write(&bad);
        assert!(PhasedGraph::load_staged(&p, 2, 1).is_err());
        std::fs::remove_file(p).unwrap();
    }
}
