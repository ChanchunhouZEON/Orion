/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::common::ANNResult;
use crate::common::AlignedBoxWithSlice;

const CACHE_LINE_BYTES: usize = 64;
const DEFAULT_WRITE_QUEUE_CAPACITY: usize = 64;

/// Per-node header layout (u32 offsets within a slot).
pub const HEADER_U32: usize = 8;
pub const BIDIR_OFFSET: usize = 4;

/// Concurrent slab buffer for fixed-size node records.
///
/// A contiguous aligned buffer is divided into `num_nodes` slots of
/// `stride_u32` u32s each. Each slot has an independent `AtomicUsize`
/// reader count so that writes to one node only need to wait for
/// readers on *that* node — not the entire graph.
///
/// ## Write queue
///
/// A single bounded queue manages write requests for all nodes.
/// When a write targets a node with active readers, the request is
/// enqueued. Each `write_node` call attempts an opportunistic flush:
/// queued writes whose target nodes have zero readers are applied and
/// dequeued. When the queue reaches capacity, a **stop-the-world**
/// pause blocks new `SegmentGuard` acquisitions, waits for all
/// per-node readers to drain, flushes every pending write, then resumes.
pub struct NodeSlabBuffer {
    buffer: AlignedBoxWithSlice<u32>,
    /// Per-node reader count. `node_readers[i]` tracks active
    /// `SegmentGuard`s on node `i`.
    node_readers: Box<[AtomicUsize]>,
    /// Global flag: blocks new `SegmentGuard` acquisitions during
    /// stop-the-world flush.
    write_blocked: AtomicBool,
    /// Single bounded write queue shared across all nodes.
    write_queue: Mutex<VecDeque<NodeWrite>>,
    write_queue_capacity: usize,
    num_nodes: usize,
    stride_u32: usize,
}

/// Queued write for a single node.
struct NodeWrite {
    node: usize,
    /// Packed: `[compressed(0..compressed_len) | rest(compressed_len..)]`
    neighbors: Vec<u32>,
    compressed_len: usize,
}

// Safety: all shared state goes through atomics and Mutex.
unsafe impl Send for NodeSlabBuffer {}
unsafe impl Sync for NodeSlabBuffer {}

impl NodeSlabBuffer {
    /// Allocate a zeroed slab with `num_nodes` slots.
    pub fn new(num_nodes: usize, stride_u32: usize) -> Self {
        let total = num_nodes * stride_u32;
        let buffer =
            AlignedBoxWithSlice::<u32>::new(total, CACHE_LINE_BYTES).expect("NodeSlabBuffer alloc");
        let node_readers: Vec<AtomicUsize> = (0..num_nodes).map(|_| AtomicUsize::new(0)).collect();
        Self {
            buffer,
            node_readers: node_readers.into_boxed_slice(),
            write_blocked: AtomicBool::new(false),
            write_queue: Mutex::new(VecDeque::new()),
            write_queue_capacity: DEFAULT_WRITE_QUEUE_CAPACITY,
            num_nodes,
            stride_u32,
        }
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride_u32
    }

    // ── Direct read (zero overhead, no atomics) ──────────────────────────────

    /// Raw slice for node `i` (full slot including header + neighbors).
    #[inline]
    pub fn slot(&self, i: usize) -> &[u32] {
        let base = i * self.stride_u32;
        &self.buffer[base..base + self.stride_u32]
    }

    /// Pointer to start of node `i` (for prefetch).
    #[inline]
    pub fn slot_ptr(&self, i: usize) -> *const u32 {
        unsafe { self.buffer.as_ptr().add(i * self.stride_u32) }
    }

    /// Full buffer slice (for parallel read-only passes like compute_bidir).
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        self.buffer.as_slice()
    }

    // ── Guarded read (per-node reader tracking) ──────────────────────────────

    /// Acquire a per-node read guard. Spins during stop-the-world.
    #[inline]
    pub fn read_node(&self, i: usize) -> SegmentGuard<'_> {
        loop {
            if !self.write_blocked.load(Ordering::Acquire) {
                self.node_readers[i].fetch_add(1, Ordering::Acquire);
                if !self.write_blocked.load(Ordering::Acquire) {
                    return SegmentGuard {
                        slab: self,
                        node: i,
                    };
                }
                self.node_readers[i].fetch_sub(1, Ordering::Release);
            }
            std::hint::spin_loop();
        }
    }

    // ── Write interface (single bounded queue for all nodes) ─────────────────

    /// Enqueue a node update.
    ///
    /// - If node `node` has **no active readers**, the write is applied
    ///   immediately (after an opportunistic flush of other ready writes).
    /// - Otherwise the request is queued.
    /// - When the queue reaches capacity → **stop-the-world flush**.
    pub fn write_node(&self, node: usize, compressed: &[u32], rest: &[u32]) {
        let mut queue = self.write_queue.lock().unwrap();

        // Opportunistic: flush queued writes whose target nodes are free.
        self.try_flush_queue(&mut queue);

        // Try direct write for the current request.
        if self.node_readers[node].load(Ordering::Acquire) == 0 {
            self.apply_write(node, compressed, rest);
            return;
        }

        // Target node has readers — queue the request.
        let mut data = Vec::with_capacity(compressed.len() + rest.len());
        data.extend_from_slice(compressed);
        data.extend_from_slice(rest);
        queue.push_back(NodeWrite {
            node,
            neighbors: data,
            compressed_len: compressed.len(),
        });

        // Queue full → stop-the-world.
        if queue.len() >= self.write_queue_capacity {
            self.stop_the_world_flush(&mut queue);
        }
    }

    /// Flush all pending writes. Stop-the-world if any target nodes have readers.
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

    /// Build-phase direct write: no concurrent readers assumed.
    ///
    /// # Safety
    /// No concurrent readers may be active on node `node`.
    /// Different threads must target different nodes (cache-line alignment
    /// prevents false sharing).
    pub unsafe fn write_node_unchecked(&self, node: usize, compressed: &[u32], rest: &[u32]) {
        self.apply_write(node, compressed, rest);
    }

    /// Write raw u32 values at a specific offset within a node slot.
    /// Build-phase only (no concurrent readers).
    pub unsafe fn write_raw(&self, node: usize, offset: usize, values: &[u32]) {
        unsafe {
            let base = node * self.stride_u32 + offset;
            let ptr = self.buffer.as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr.add(base), values.len());
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    /// Apply a write directly to the buffer. Caller ensures no readers on `node`.
    fn apply_write(&self, node: usize, compressed: &[u32], rest: &[u32]) {
        let base = node * self.stride_u32;
        let total = compressed.len() + rest.len();
        let ptr = self.buffer.as_ptr() as *mut u32;
        unsafe {
            std::ptr::write(ptr.add(base), total as u32);
            std::ptr::write(ptr.add(base + 1), compressed.len() as u32);
            std::ptr::copy_nonoverlapping(
                compressed.as_ptr(),
                ptr.add(base + HEADER_U32),
                compressed.len(),
            );
            std::ptr::copy_nonoverlapping(
                rest.as_ptr(),
                ptr.add(base + HEADER_U32 + compressed.len()),
                rest.len(),
            );
        }
    }

    /// Flush queued writes whose target nodes have zero readers.
    fn try_flush_queue(&self, queue: &mut VecDeque<NodeWrite>) {
        let mut i = 0;
        while i < queue.len() {
            let node = queue[i].node;
            if self.node_readers[node].load(Ordering::Acquire) == 0 {
                let req = queue.remove(i).unwrap();
                let cd = req.compressed_len;
                self.apply_write(req.node, &req.neighbors[..cd], &req.neighbors[cd..]);
            } else {
                i += 1;
            }
        }
    }

    fn stop_the_world_flush(&self, queue: &mut VecDeque<NodeWrite>) {
        self.write_blocked.store(true, Ordering::Release);
        for reader in self.node_readers.iter() {
            while reader.load(Ordering::Acquire) > 0 {
                std::hint::spin_loop();
            }
        }
        while let Some(req) = queue.pop_front() {
            let cd = req.compressed_len;
            self.apply_write(req.node, &req.neighbors[..cd], &req.neighbors[cd..]);
        }
        self.write_blocked.store(false, Ordering::Release);
    }

    // ── IO ───────────────────────────────────────────────────────────────────

    /// Save buffer contents to a writer.
    pub fn save_to<W: Write>(&self, writer: &mut W) -> ANNResult<()> {
        let slice = self.buffer.as_slice();
        let bytes =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
        writer.write_all(bytes)?;
        Ok(())
    }

    /// Load buffer contents from a reader.
    pub fn load_from<R: Read>(
        reader: &mut R,
        num_nodes: usize,
        stride_u32: usize,
    ) -> ANNResult<Self> {
        let total = num_nodes * stride_u32;
        let slab = Self::new(num_nodes, stride_u32);
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(slab.buffer.as_ptr() as *mut u8, total * 4) };
        reader.read_exact(bytes)?;
        Ok(slab)
    }
}

impl Clone for NodeSlabBuffer {
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
            stride_u32: self.stride_u32,
        }
    }
}

impl std::fmt::Debug for NodeSlabBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeSlabBuffer")
            .field("num_nodes", &self.num_nodes)
            .field("stride_u32", &self.stride_u32)
            .finish()
    }
}

// ── SegmentGuard ─────────────────────────────────────────────────────────────

/// RAII guard for a single node slot. Decrements the per-node reader
/// count on drop, allowing queued writes to that node to proceed.
pub struct SegmentGuard<'a> {
    slab: &'a NodeSlabBuffer,
    node: usize,
}

impl<'a> SegmentGuard<'a> {
    #[inline]
    pub fn slot(&self) -> &[u32] {
        self.slab.slot(self.node)
    }
}

impl<'a> Drop for SegmentGuard<'a> {
    fn drop(&mut self) {
        self.slab.node_readers[self.node].fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_write_read() {
        let slab = NodeSlabBuffer::new(4, 16);
        unsafe {
            slab.write_node_unchecked(0, &[10], &[20]);
        }
        let s = slab.slot(0);
        assert_eq!(s[0], 2); // degree
        assert_eq!(s[1], 1); // compressed_degree
        assert_eq!(s[HEADER_U32], 10);
        assert_eq!(s[HEADER_U32 + 1], 20);
    }

    #[test]
    fn test_segment_guard() {
        let slab = NodeSlabBuffer::new(2, 16);
        unsafe {
            slab.write_node_unchecked(0, &[5], &[6, 7]);
        }
        let guard = slab.read_node(0);
        let s = guard.slot();
        assert_eq!(s[0], 3);
        assert_eq!(s[1], 1);
        drop(guard);
    }

    #[test]
    fn test_direct_write_no_readers() {
        let slab = NodeSlabBuffer::new(4, 16);
        unsafe {
            slab.write_node_unchecked(0, &[1, 2], &[3]);
        }
        slab.write_node(0, &[10], &[20]);
        assert_eq!(slab.slot(0)[0], 2);
        assert_eq!(slab.slot(0)[HEADER_U32], 10);
    }

    #[test]
    fn test_concurrent_segment_independence() {
        use std::sync::Arc;
        let slab = Arc::new(NodeSlabBuffer::new(4, 16));
        unsafe {
            slab.write_node_unchecked(0, &[1], &[2]);
            slab.write_node_unchecked(1, &[3], &[4]);
        }

        // Hold a guard on node 0.
        let guard = slab.read_node(0);

        // Write to node 1 succeeds immediately (different node, no contention).
        slab.write_node(1, &[30], &[40]);
        assert_eq!(slab.slot(1)[HEADER_U32], 30);

        drop(guard);
    }

    #[test]
    fn test_save_load() {
        let slab = NodeSlabBuffer::new(3, 16);
        unsafe {
            slab.write_node_unchecked(0, &[1, 2], &[3]);
            slab.write_node_unchecked(2, &[7], &[8, 9]);
        }
        let mut buf = Vec::new();
        slab.save_to(&mut buf).unwrap();

        let loaded = NodeSlabBuffer::load_from(&mut &buf[..], 3, 16).unwrap();
        assert_eq!(loaded.slot(0)[0], 3);
        assert_eq!(loaded.slot(2)[HEADER_U32], 7);
    }
}
