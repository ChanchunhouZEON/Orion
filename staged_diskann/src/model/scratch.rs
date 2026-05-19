/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::sync::Arc;

use crossbeam::queue::ArrayQueue;

use crate::algorithm::search::convergence::SearchConvergenceChecker;
use crate::model::Neighbor as DNeighbor;
use crate::model::NeighborPriorityQueue;
use crate::model::neighbor::neighbor_priority_queue::pad16;
use crate::model::visited_set::HashsetSeen;

/// Pre-allocated scratch for in-memory greedy search.
pub struct InMemSearchScratch {
    /// Sorted candidate queue; no internal dedup — dedup is handled by `seen`.
    /// Inserts go through `insert_unchecked` (no HashSet probe) since the
    /// external `seen` set already filters duplicates at neighbor-expansion
    /// time.
    pub pq: NeighborPriorityQueue,
    /// L1-resident approximate hash-set visited tracker (ParlayANN-style).
    pub seen: HashsetSeen,
    /// Staging buffer: unseen neighbor IDs collected before distance computation.
    pub id_scratch: Vec<u32>,
    /// Distance-computed candidates buffered per hop so they can be sorted
    /// once and batch-merged into `pq` in one pass (ParlayANN batching trick).
    ///
    /// **Capacity is sized once at scratch construction** to the
    /// worst-case staged volume across `FLUSH_INTERVAL` hops:
    /// `search_list_size + MAX_GRAPH_DEGREE × MAX_FLUSH_INTERVAL`.
    /// Per-hop `reserve(pre_kept)` calls were removed because they
    /// produced two pathologies: (1) the second-and-subsequent hops in
    /// a post-converged flush window inherited a `dist_buffer.len()`
    /// pinned by prior hops' admits, so each new hop's
    /// `as_mut_ptr().add(hop_start)` aimed at a position whose backing
    /// memory might have just moved (reserve realloc'd). (2) The
    /// growing visible `len` made `pq_worst_local` stale relative to
    /// the candidates already staged — `hop_admits` under-counted,
    /// `early_exit` could fire on a streak of bookkeeping zeros.
    /// With a fixed pre-allocated capacity, no reserve fires inside
    /// the search loop and the raw-pointer cmov-compact writes hit
    /// stable memory throughout the query.
    pub dist_buffer: Vec<DNeighbor>,
    /// Merge scratch for `NeighborPriorityQueue::batch_merge` — avoids
    /// per-call alloc when doing the linear-time set-union update.
    pub merge_scratch: Vec<DNeighbor>,

    /// **f32 query padded to a 32-B chunk multiple** (`ceil(N/8)·8`
    /// elements, trailing zero-pad). The Stage-2 truth-distance
    /// streaming kernel reads `chunks_per_vert · 32` bytes from query
    /// per vertex, so when `N % 8 != 0` the trailing partial chunk
    /// would otherwise read past the `[f32; N]` query end. Allocated
    /// once at scratch creation, refilled each query.
    pub q_query_f32_padded: Vec<f32>,

    /// Adaptive pre-filter threshold state (ParlayANN `beamSearch.h:136-145`):
    /// running average of the frontier's u8-distance tail-mean, used to
    /// tighten the quantized pre-filter cutoff as the search stabilizes.
    pub filter_threshold_sum: f32,
    pub filter_threshold_count: u32,
    pub last_worst_id: u32,
    pub filter_tail_mean: f32,
    /// Reusable convergence checker — avoids per-query allocation.
    pub dcc: SearchConvergenceChecker,
    /// Reusable early exit checker.
    pub early_exit: crate::algorithm::search::early_exit::EarlyExitChecker,
}

impl InMemSearchScratch {
    pub fn new(search_list_size: usize) -> Self {
        Self {
            pq: NeighborPriorityQueue::with_capacity(search_list_size),
            seen: HashsetSeen::new(search_list_size),
            id_scratch: Vec::with_capacity(64),
            // Capacity = MAX_GRAPH_DEGREE × MAX_FLUSH_INTERVAL = 100 × 4
            // = 400. Each hop stages up to `MAX_GRAPH_DEGREE` (= 100)
            // candidate distances; converged-phase accumulates across
            // up to FLUSH_INTERVAL[1] = 4 hops before flushing.
            dist_buffer: Vec::with_capacity(400),
            // Sized to match `pq.data` exactly via the same `pad8(L+1)`
            // formula `NeighborPriorityQueue::with_capacity` uses. The
            // two Vecs swap on every `batch_merge` / `batch_merge_gallop`
            // call, so once steady state is reached neither side has to
            // reallocate. The earlier `+128` slack was redundant — gallop
            // can briefly push past `cap`, but the `min(cap)` truncation
            // before swap restores the invariant.
            merge_scratch: Vec::with_capacity(pad16(search_list_size + 1)),
            // Sized to next multiple of 8 ≥ 1024 — covers up to
            // N=1024 dim before any reallocation. For typical
            // glove100 / SIFT (N ≤ 128) we use the first 16 elements
            // and the rest are unused but pre-allocated.
            q_query_f32_padded: Vec::with_capacity(1024),
            filter_threshold_sum: 0.0,
            filter_threshold_count: 0,
            last_worst_id: u32::MAX,
            filter_tail_mean: 0.0,
            dcc: SearchConvergenceChecker::new(5, 0.0),
            early_exit: crate::algorithm::search::early_exit::EarlyExitChecker::new(5),
        }
    }

    /// Reset for reuse. Resizes the hashset if `search_list_size` grew vs
    /// the previous query; clears it in place otherwise.
    pub fn prepare_for_query(&mut self, search_list_size: usize) {
        self.pq.clear();
        self.pq.reserve(search_list_size);
        self.pq.set_capacity(search_list_size);
        self.seen.resize_for(search_list_size);
        self.id_scratch.clear();
        self.merge_scratch.clear();
        self.filter_threshold_sum = 0.0;
        self.filter_threshold_count = 0;
        self.last_worst_id = u32::MAX;
        self.filter_tail_mean = 0.0;
        self.dcc.reset();
        self.early_exit.reset();
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
