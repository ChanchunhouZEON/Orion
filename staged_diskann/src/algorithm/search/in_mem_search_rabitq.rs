/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! `search_rabitq<N>` — **single-stage RaBitQ 1-bit-per-dim beam
//! search + post-hoc f32 rerank**, mirrors [`search_l2_u8_q`] but
//! replaces the u8-quantized Stage-1 with the rotated 1-bit RaBitQ
//! estimator.
//!
//! Pipeline (per query):
//!   1. Rotate the query (`P @ q`, kept f32 in a stack-local buffer)
//!      and precompute `||q||²` for the L2 expansion.
//!   2. Insert entry into PQ with the RaBitQ-estimated L2² distance.
//!   3. **Peel** 3 hops: each pops the closest unvisited, expands
//!      neighbours, computes RaBitQ estimates via
//!      [`RabitQDataset::estimate_l2_sq`], cmov-compacts admits (those
//!      beating the current PQ tail) into `dist_buffer`, sorts, and
//!      `batch_merge`s into PQ. Skips DCC / early-exit because the PQ
//!      isn't full yet.
//!   4. Main loop hops: same scalar RaBitQ estimator + cmov-compact +
//!      flush via 3-way routing every `FLUSH_INTERVAL[converged]`
//!      hops, with DCC convergence and early-exit countdown active.
//!   5. **Post-hoc f32 rerank**: take top `k · RERANK_FACTOR` from the
//!      RaBitQ PQ, recompute f32 truth via
//!      [`vector::DistanceStream<L2F32Distance, N>`] in one batched
//!      call, sort, emit top-k.
//!
//! PQ holds **estimated L2² in f32** throughout the search loop
//! (RaBitQ's estimator returns f32 directly). Memory bandwidth on the
//! per-hop Stage-1 read is `~N/8` bytes per vertex — for GIST D=960
//! that's 120 B per vertex, vs 960 B for u8-quantized — so the entire
//! quantized base fits in L1 for D ≥ ~512 on M2.
//!
//! ## Status
//!
//! Reference correctness path: Stage-1 uses
//! [`RabitQDataset::estimate_l2_sq`] which is **scalar** (no NEON yet).
//! The hand-tuned NEON 1-bit dot product is the next pass; once it
//! lands, this kernel slots in as a drop-in replacement of the
//! scalar inner loop.

use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::StagedDiskANN;
use crate::model::Neighbor as DNeighbor;
use crate::model::dataset::rabitq_dataset::RabitQDataset;
use crate::model::scratch::{InMemScratchPool, InMemSearchScratch};
use diskann::common::ANNResult;
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use super::in_mem_search::{
    AlignedQuery, FLUSH_INTERVAL, dstream_la_truth, insert_route_mul, linear_merge_mul,
};
use super::in_mem_search::{
    NDC_F32, NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, QUERY_COUNT,
    SETUP_NS, VISIT_COUNT,
};

/// Default rerank factor — top `k · RERANK_FACTOR` from the RaBitQ PQ
/// get re-evaluated with f32 truth. Overridable via env
/// `STAGED_RBQ_RERANK=<n>` for tuning. The basic estimator (no
/// per-vertex `s` correction) is noisier than the u8-quantized
/// baseline, especially at low D — wider rerank lets the f32 stage
/// rescue candidates the noisy Stage-1 mis-ranked. Default matches
/// `search_l2_u8_q` for apples-to-apples comparison; bump on SIFT-class
/// data where the per-vertex `s` upgrade isn't in yet.
const DEFAULT_RERANK_FACTOR: usize = 2;

#[inline]
fn rerank_factor() -> usize {
    std::env::var("STAGED_RBQ_RERANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_RERANK_FACTOR)
}

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// One peeled hop in the warm-up window (hops 0..=2). Mirrors
    /// [`Self::search_l2_u8_q`]'s peel but uses the scalar RaBitQ
    /// estimator for Stage-1 distances. The peel's role is the same:
    /// fill the PQ with close-by candidates before DCC/early-exit
    /// activate, exploiting that the PQ tail is `MAX` or only loosely
    /// constrained so every admit passes.
    #[inline(always)]
    fn expand_peeled_hop_rabitq(
        &self,
        q_ds: &RabitQDataset<N>,
        rotated_q: &[f32; N],
        q_norm_sq: f32,
        search_list_size: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Option<usize> {
        if !scratch.pq.has_notvisited_node() {
            return None;
        }
        let graph = &self.graph;

        let id = scratch.pq.closest_notvisited().id as usize;
        if let Some(next) = scratch.pq.peek_notvisited() {
            graph.prefetch_node(next.id as usize);
        }

        scratch.id_scratch.clear();
        for &nn in graph.neighbors(id) {
            if scratch.seen.insert(nn) {
                scratch.id_scratch.push(nn);
            }
        }
        let n = scratch.id_scratch.len();
        if n == 0 {
            return Some(0);
        }
        let pq_worst = if scratch.pq.size() >= search_list_size {
            scratch.pq[scratch.pq.size() - 1].distance
        } else {
            f32::MAX
        };

        let mut admitted = 0usize;
        for &nn in scratch.id_scratch.iter() {
            let qd = q_ds.estimate_l2_sq(rotated_q, q_norm_sq, nn);
            if qd < pq_worst {
                scratch.dist_buffer.push(DNeighbor::new(nn, qd));
                admitted += 1;
            }
        }

        NDC_I8.fetch_add(n as u64, Ordering::Relaxed);
        VISIT_COUNT.fetch_add(n as u64, Ordering::Relaxed);

        if admitted > 0 {
            scratch.dist_buffer.sort_unstable_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then_with(|| a.id.cmp(&b.id))
            });
            scratch
                .pq
                .batch_merge(&scratch.dist_buffer, &mut scratch.merge_scratch);
            scratch.dist_buffer.clear();
        }
        Some(admitted)
    }

    /// RaBitQ beam search: PQ holds RaBitQ-estimated L2² distances in
    /// f32; post-hoc f32 rerank of top `k · RERANK_FACTOR`. Targets
    /// high-dim datasets (GIST and above) where the 1-bit/dim code
    /// makes the quantized base fit in L1 cache.
    pub fn search_rabitq(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
    ) -> ANNResult<Vec<u32>> {
        let t_setup = Instant::now();
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let aligned = AlignedQuery(*query);

        // RaBitQ sidecar (lazily built / loaded from `.qrbq`).
        let q_ds = self.ensure_quantized_dataset_rabitq();

        // Per-query setup: rotate query into the same f32 space the
        // codes live in, precompute `||q||²` for the L2 expansion.
        let mut rotated_q = [0.0f32; N];
        q_ds.rotate_query(&aligned.0, &mut rotated_q);
        let q_norm_sq: f32 = aligned.0.iter().map(|v| v * v).sum();

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        // Entry distance via the same estimator — keeps the PQ tail
        // comparator apples-to-apples for the rest of the loop.
        scratch.seen.insert(entry);
        let entry_dist = q_ds.estimate_l2_sq(&rotated_q, q_norm_sq, entry);
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));
        SETUP_NS.fetch_add(t_setup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut pre_hops: u64 = 0;
        let mut post_hops: u64 = 0;
        let mut pre_admits: u64 = 0;
        let mut post_admits: u64 = 0;

        // Peel the first 3 hops. Same justification as `search_l2_u8_q`:
        // PQ transitions empty → partial → full, convergence can't
        // fire, the incoming batch is large enough that 3-way merge
        // routing always picks linear `batch_merge`.
        let mut prev_admitted: usize = 1;
        'peel: {
            match self.expand_peeled_hop_rabitq(
                q_ds,
                &rotated_q,
                q_norm_sq,
                search_list_size,
                scratch,
            ) {
                Some(n) => {
                    scratch.dcc.update(prev_admitted);
                    scratch.early_exit.should_exit(false, prev_admitted);
                    prev_admitted = n.max(1);
                    pre_hops += 1;
                    pre_admits += n as u64;
                }
                None => break 'peel,
            }
            match self.expand_peeled_hop_rabitq(
                q_ds,
                &rotated_q,
                q_norm_sq,
                search_list_size,
                scratch,
            ) {
                Some(n) => {
                    scratch.dcc.update(prev_admitted);
                    scratch.early_exit.should_exit(false, prev_admitted);
                    prev_admitted = n.max(1);
                    pre_hops += 1;
                    pre_admits += n as u64;
                }
                None => break 'peel,
            }
            match self.expand_peeled_hop_rabitq(
                q_ds,
                &rotated_q,
                q_norm_sq,
                search_list_size,
                scratch,
            ) {
                Some(n) => {
                    scratch.dcc.update(prev_admitted);
                    scratch.early_exit.should_exit(false, prev_admitted);
                    prev_admitted = n.max(1);
                    pre_hops += 1;
                    pre_admits += n as u64;
                }
                None => break 'peel,
            }
        }
        let mut hops_since_flush: usize = 0;

        while scratch.pq.has_notvisited_node() {
            let id = scratch.pq.closest_notvisited().id as usize;

            if let Some(next) = scratch.pq.peek_notvisited() {
                graph.prefetch_node(next.id as usize);
            }

            let converged = scratch.dcc.update(prev_admitted);

            scratch.id_scratch.clear();
            if !converged {
                for &nn in graph.neighbors(id) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }
            } else {
                let (local, extra) = graph.rerank_candidates(id);
                for &nn in local.iter().chain(extra.iter()) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }
            }

            let n_unseen = scratch.id_scratch.len();
            let pq_worst = if scratch.pq.size() >= search_list_size {
                scratch.pq[scratch.pq.size() - 1].distance
            } else {
                f32::MAX
            };

            // Stage-1: scalar RaBitQ estimator on every unseen neighbour.
            // Admits (qd < pq_worst) get pushed to `dist_buffer` for the
            // flush phase to batch-merge into PQ. The NEON kernel will
            // replace this loop with a vectorized 1-bit dot product;
            // until then the per-cmp cost is ~N float fmas vs the u8
            // path's ~N/16 vector fmas — slower but bandwidth-bound on
            // the same scale, so the absolute throughput drop should be
            // modest while we wait for the NEON kernel.
            let hop_start = scratch.dist_buffer.len();
            let mut hop_admits: usize = 0;
            for &nn in scratch.id_scratch.iter() {
                let qd = q_ds.estimate_l2_sq(&rotated_q, q_norm_sq, nn);
                if qd < pq_worst {
                    scratch.dist_buffer.push(DNeighbor::new(nn, qd));
                    hop_admits += 1;
                }
            }
            // hop_start is the boundary between earlier-hop admits
            // (still in dist_buffer awaiting the next flush) and this
            // hop's admits — keep the assertion to surface accidental
            // buffer truncation in future refactors.
            debug_assert!(scratch.dist_buffer.len() == hop_start + hop_admits);

            prev_admitted = hop_admits;

            if converged {
                post_hops += 1;
                post_admits += hop_admits as u64;
            } else {
                pre_hops += 1;
                pre_admits += hop_admits as u64;
            }

            VISIT_COUNT.fetch_add(n_unseen as u64, Ordering::Relaxed);
            NDC_I8.fetch_add(n_unseen as u64, Ordering::Relaxed);

            let should_exit = scratch.early_exit.should_exit(converged, hop_admits);
            let will_stop = should_exit | !scratch.pq.has_notvisited_node();

            hops_since_flush += 1;
            let flush_interval = FLUSH_INTERVAL[converged as usize];
            let must_flush = (hops_since_flush >= flush_interval) | will_stop;

            if must_flush {
                let cnt = scratch.dist_buffer.len();
                if cnt > 0 {
                    if insert_route_mul(cnt) < search_list_size {
                        for c in scratch.dist_buffer.drain(..) {
                            scratch.pq.insert(c);
                        }
                    } else {
                        scratch.dist_buffer.sort_unstable_by(|a, b| {
                            a.distance
                                .total_cmp(&b.distance)
                                .then_with(|| a.id.cmp(&b.id))
                        });
                        if linear_merge_mul(cnt) > search_list_size {
                            scratch
                                .pq
                                .batch_merge(&scratch.dist_buffer, &mut scratch.merge_scratch);
                        } else {
                            scratch.pq.batch_merge_gallop(
                                &scratch.dist_buffer,
                                &mut scratch.merge_scratch,
                            );
                        }
                        scratch.dist_buffer.clear();
                    }
                }
                hops_since_flush = 0;
            }

            if should_exit {
                break;
            }
        }

        // Publish counters for the benchmarker.
        QUERY_COUNT.fetch_add(1, Ordering::Relaxed);
        PRE_CONV_HOPS.fetch_add(pre_hops, Ordering::Relaxed);
        POST_CONV_HOPS.fetch_add(post_hops, Ordering::Relaxed);
        PRE_CONV_ADMITS.fetch_add(pre_admits, Ordering::Relaxed);
        POST_CONV_ADMITS.fetch_add(post_admits, Ordering::Relaxed);

        // ── Post-hoc f32 rerank ───────────────────────────────────────
        // Top `k · RERANK_FACTOR` from the RaBitQ PQ → f32 truth via
        // `DistanceStream<L2F32Distance, N>` in one batched call → sort
        // → emit top-k. Same shape as `search_l2_u8_q`.
        let beam_n = scratch.pq.size();
        let num_check = (k * rerank_factor()).min(beam_n);
        if num_check == 0 {
            return Ok(Vec::new());
        }

        let f32_stride_bytes = N * 4;
        let f32_compute_bytes = f32_stride_bytes.div_ceil(32) * 32;
        let f32_padded_lanes = f32_compute_bytes / 4;
        scratch.q_query_f32_padded.clear();
        scratch.q_query_f32_padded.extend_from_slice(&aligned.0);
        scratch.q_query_f32_padded.resize(f32_padded_lanes, 0.0);

        let dataset_base_ptr = dataset.get_data().as_ptr() as *const u8;
        scratch.id_scratch.clear();
        for i in 0..num_check {
            scratch.id_scratch.push(scratch.pq[i].id);
        }

        let mut rerank_buf: Vec<(u32, f32)> = Vec::with_capacity(num_check);
        unsafe {
            let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), num_check);
            vector::DistanceStream::<vector::L2F32Distance, N>::new(
                dataset_base_ptr,
                f32_stride_bytes,
                f32_compute_bytes,
                scratch.q_query_f32_padded.as_ptr() as *const u8,
                id_in,
                dstream_la_truth(),
            )
            .run(|id, dist| {
                rerank_buf.push((id, dist));
            });
        }
        NDC_F32.fetch_add(num_check as u64, Ordering::Relaxed);
        rerank_buf.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let n_out = k.min(rerank_buf.len());
        let mut out: Vec<u32> = Vec::with_capacity(n_out);
        for i in 0..n_out {
            out.push(rerank_buf[i].0);
        }
        Ok(out)
    }

    /// Parallel batch wrapper for [`Self::search_rabitq`]. Same
    /// L-adaptive `par_chunks(BATCH)` shape as `search_batch_l2_u8_q`
    /// / `search_batch_mips_q` so every search path shares the rayon
    /// dispatch heuristic. `STAGED_BATCH=<n>` env override forces a
    /// fixed batch size for profiling.
    pub fn search_batch_rabitq(
        &self,
        queries: &[[f32; N]],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
    ) -> ANNResult<Vec<Vec<u32>>> {
        self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let batch: usize = match std::env::var("STAGED_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(b) if b > 0 => b,
            _ => super::in_mem_search::search_batch_size(search_list_size),
        };
        let n = queries.len();
        let mut results: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        results
            .par_chunks_mut(batch)
            .zip(queries.par_chunks(batch))
            .for_each(|(out_chunk, q_chunk)| {
                for (out, query) in out_chunk.iter_mut().zip(q_chunk.iter()) {
                    *out = self
                        .search_rabitq(
                            query,
                            k,
                            search_list_size,
                            window_size,
                            epsilon,
                            early_exit_limit,
                        )
                        .unwrap_or_default();
                }
            });
        Ok(results)
    }
}
