/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::sync::Arc;

use crossbeam::queue::ArrayQueue;
use diskann::model::NeighborPriorityQueue as DiskANNPQ;

use crate::algorithm::search::convergence::SearchConvergenceChecker;

/// Bitset-based visited tracker. For N nodes needs N/8 bytes.
/// 100K nodes = 12.5 KB (fits in L1), 1M nodes = 125 KB (fits in L2).
pub struct BitVecSeen {
    bits: Vec<u64>,
    num_words: usize,
}

impl BitVecSeen {
    pub fn new(num_nodes: usize) -> Self {
        let num_words = (num_nodes + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            num_words,
        }
    }

    /// Insert node. Returns true if the node was NOT previously seen (newly inserted).
    #[inline]
    pub fn insert(&mut self, id: u32) -> bool {
        let word = id as usize >> 6;
        let bit = 1u64 << (id & 63);
        if self.bits[word] & bit != 0 {
            false
        } else {
            self.bits[word] |= bit;
            true
        }
    }

    /// Clear all bits without deallocating.
    pub fn clear(&mut self) {
        // memset to zero — much faster than per-element clear for large bitsets.
        // For small bitsets (< L1) this is a single cache line write.
        unsafe {
            std::ptr::write_bytes(self.bits.as_mut_ptr(), 0, self.num_words);
        }
    }
}

/// Pre-allocated scratch for in-memory greedy search.
pub struct InMemSearchScratch {
    /// Sorted candidate queue; no internal dedup — dedup is handled by `seen`.
    pub pq: DiskANNPQ,
    /// Bitset-based visited tracker. O(1) insert + test, L1-friendly.
    pub seen: BitVecSeen,
    /// Staging buffer: unseen neighbor IDs collected before distance computation.
    pub id_scratch: Vec<u32>,
    /// Reusable convergence checker — avoids per-query allocation.
    pub dcc: SearchConvergenceChecker,
    /// Reusable early exit checker.
    pub early_exit: crate::algorithm::search::early_exit::EarlyExitChecker,
}

impl InMemSearchScratch {
    pub fn new(search_list_size: usize) -> Self {
        // Default capacity for bitset — will be resized on first prepare_for_query
        // if the actual num_nodes is larger.
        Self {
            pq: DiskANNPQ::with_capacity(search_list_size),
            seen: BitVecSeen::new(search_list_size * 20),
            id_scratch: Vec::with_capacity(64),
            dcc: SearchConvergenceChecker::new(5, 0.0),
            early_exit: crate::algorithm::search::early_exit::EarlyExitChecker::new(5),
        }
    }

    /// Reset for reuse. `num_nodes` sets the bitset capacity.
    pub fn prepare_for_query(&mut self, search_list_size: usize) {
        self.pq.clear();
        self.pq.reserve(search_list_size);
        self.pq.set_capacity(search_list_size);
        self.seen.clear();
        self.id_scratch.clear();
        self.dcc.reset();
        self.early_exit.reset();
    }

    /// Ensure the bitset covers `num_nodes` nodes.
    pub fn ensure_capacity(&mut self, num_nodes: usize) {
        let needed = (num_nodes + 63) / 64;
        if needed > self.seen.num_words {
            self.seen = BitVecSeen::new(num_nodes);
        }
    }
}

/// Pool of `InMemSearchScratch` objects for concurrent in-memory search.
///
/// Equivalent to DiskANN's `ArcConcurrentBoxedQueue<InMemQueryScratch>` but backed by
/// crossbeam's lock-free `ArrayQueue` instead of a mutex-based queue.
pub struct InMemScratchPool {
    pool: Arc<ArrayQueue<Box<InMemSearchScratch>>>,
}

impl InMemScratchPool {
    pub fn new(num_threads: usize, search_list_size: usize) -> Self {
        let pool = Arc::new(ArrayQueue::new(num_threads));
        for _ in 0..num_threads {
            pool.push(Box::new(InMemSearchScratch::new(search_list_size)))
                .ok();
        }
        Self { pool }
    }

    /// Checkout a scratch object, spinning until one is available.
    pub fn acquire(&self) -> InMemScratchGuard {
        loop {
            if let Some(scratch) = self.pool.pop() {
                return InMemScratchGuard {
                    scratch: Some(scratch),
                    pool: self.pool.clone(),
                };
            }
            std::hint::spin_loop();
        }
    }
}

/// RAII guard: returns scratch to pool on drop (without clearing — caller calls
/// `prepare_for_query` at checkout time instead).
pub struct InMemScratchGuard {
    scratch: Option<Box<InMemSearchScratch>>,
    pool: Arc<ArrayQueue<Box<InMemSearchScratch>>>,
}

impl InMemScratchGuard {
    pub fn scratch(&mut self) -> &mut InMemSearchScratch {
        self.scratch.as_deref_mut().unwrap()
    }
}

impl Drop for InMemScratchGuard {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            self.pool.push(scratch).ok();
        }
    }
}
