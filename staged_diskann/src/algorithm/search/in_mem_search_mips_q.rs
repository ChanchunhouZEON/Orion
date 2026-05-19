/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! `search_mips_q<Q, N>` — **single-stage quantized MIPS beam search +
//! post-hoc f32 rerank**, the GloVe-100 / mid-recall-band path.
//!
//! Pipeline (per hop):
//!   1. Pop closest unvisited from PQ.
//!   2. Expand neighbours (local + remote / local + extra after DCC
//!      converges).
//!   3. Stream **quantized** distances over unseen neighbours via
//!      [`vector::DistanceStream`] driven by `Q::QuantDistanceFn`,
//!      cmov-compact admits (`qd < pq_worst`) into `dist_buffer`.
//!   4. Flush `dist_buffer` into PQ via 3-way routing (per-insert /
//!      gallop / linear) every `FLUSH_INTERVAL[converged]` hops or
//!      on early-exit.
//!
//! Post-loop:
//!   * Take the top `k · RERANK_FACTOR` PQ candidates, recompute f32
//!     distance via [`vector::DistanceStream`] driven by
//!     `Q::TruthDistanceFn`, sort, emit top-k IDs.
//!
//! This matches PA's GloVe recipe (`-quantize_bits 16 -quantize_mode 1
//! -rerank_factor 2`) but with our MSHR-aware DistanceStream prefetch
//! pipeline + 4-wide ILP unroll.

use std::sync::atomic::Ordering;

use crate::StagedDiskANN;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::InMemScratchPool;
use diskann::common::ANNResult;
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use super::in_mem_search::{
    AlignedQuery, FLUSH_INTERVAL, dstream_la_q, dstream_la_truth, insert_route_mul,
    linear_merge_mul,
};

/// PA's `-rerank_factor 2` for `mode=1` / GloVe-family recipes.
const RERANK_FACTOR: usize = 2;

// Counters (`VISIT_COUNT`, `QUERY_COUNT`, `NDC_I8`, `NDC_F32`,
// `SETUP_NS`, hop/admit splits) live in
// [`super::in_mem_search`](crate::algorithm::search::in_mem_search) so
// every search path (L2, plain MIPS, MIPS-Q) increments the same
// atomics. Re-export them here for back-compat with existing call
// sites that imported them from this module.
pub use super::in_mem_search::{
    NDC_F32, NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, QUERY_COUNT,
    SETUP_NS, VISIT_COUNT,
};

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Single-stage quantized MIPS beam search + post-hoc f32 rerank.
    /// `Q` selects i8 / i16 storage via [`crate::model::QuantSpec`].
    pub fn search_mips_q<Q: crate::model::QuantSpec>(
        &self,
        query: &[f32; N],
        q_ds: &crate::model::QuantizedDataset<Q, N>,
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
    ) -> ANNResult<Vec<u32>> {
        let t_setup = std::time::Instant::now();
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;

        // MIPS path: normalise query so the quantized base (built from
        // already-normalised f32) and the query agree on the unit-sphere
        // assumption. (Q::NORMALIZE_QUERY is true for MipsI8/MipsI16.)
        let aligned = if Q::NORMALIZE_QUERY {
            let mut q_norm: [f32; N] = *query;
            vector::l2_normalize_f32_inplace(&mut q_norm);
            AlignedQuery(q_norm)
        } else {
            AlignedQuery(*query)
        };

        // Short query (= N elements) used by `q_ds.qdist` for the
        // initial entry distance.
        let q_query = q_ds.quantize_query(&aligned.0);
        // Padded query (= STRIDE elements, trailing zeros) for the
        // DistanceStream main loop kernel — chunk loop reads exactly
        // STRIDE bytes per vertex on both base and query.
        let q_query_padded = q_ds.quantize_query_padded(&aligned.0);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        // PQ holds **quantized** distances throughout the search loop.
        // Entry must be Q::distance (via q_ds.qdist), not f32 truth,
        // so it sits on the same scale as every later PQ entry.
        scratch.seen.insert(entry);
        let entry_dist = unsafe { q_ds.qdist(entry, &q_query) };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));
        SETUP_NS.fetch_add(t_setup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut prev_admitted: usize = 1;
        let mut hops_since_flush: usize = 0;
        let mut visits: u64 = 0;
        let mut pre_hops: u64 = 0;
        let mut post_hops: u64 = 0;
        let mut pre_admits: u64 = 0;
        let mut post_admits: u64 = 0;
        let mut ndc_i8_local: u64 = 0;

        while scratch.pq.has_notvisited_node() {
            let id = scratch.pq.closest_notvisited().id as usize;
            visits += 1;

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
            ndc_i8_local += n_unseen as u64;
            let pq_worst = if scratch.pq.size() >= search_list_size {
                scratch.pq[scratch.pq.size() - 1].distance
            } else {
                f32::MAX
            };

            // No queue construction — DistanceStream resolves prfm
            // addresses inline.
            let stride_bytes =
                crate::model::QuantizedDataset::<Q, N>::STRIDE * std::mem::size_of::<Q::Storage>();
            let lookahead_lines = dstream_la_q();
            let base_ptr = q_ds.data.as_slice().as_ptr() as *const u8;

            let hop_start = scratch.dist_buffer.len();
            // dist_buffer is pre-sized at scratch construction to fit
            // `search_list_size + MAX_GRAPH_DEGREE × MAX_FLUSH_INTERVAL`
            // staged admits — no per-hop reserve needed.
            let pq_worst_local = pq_worst;
            // SAFETY: cmov-compact sink only ever advances `w` by 0
            // or 1; writes never go past `n_unseen` (= upper bound).
            let hop_admits: usize = unsafe {
                let base_out = scratch.dist_buffer.as_mut_ptr().add(hop_start);
                let mut w = 0usize;
                let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n_unseen);
                vector::DistanceStream::<Q::QuantDistanceFn, N>::new(
                    base_ptr,
                    stride_bytes,
                    stride_bytes,
                    q_query_padded.as_ptr() as *const u8,
                    id_in,
                    lookahead_lines,
                )
                .run(|id, qd| {
                    base_out.add(w).write(DNeighbor::new(id, qd));
                    w += (qd < pq_worst_local) as usize;
                });
                scratch.dist_buffer.set_len(hop_start + w);
                w
            };
            prev_admitted = hop_admits;

            // Per-phase counters: split this hop's admits + visit by
            // converged state for diagnostic printing.
            if converged {
                post_hops += 1;
                post_admits += hop_admits as u64;
            } else {
                pre_hops += 1;
                pre_admits += hop_admits as u64;
            }

            // Legacy convergence-aware early exit — DCC converged +
            // `early_exit_limit` consecutive zero-admit hops.
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
        VISIT_COUNT.fetch_add(visits, Ordering::Relaxed);
        QUERY_COUNT.fetch_add(1, Ordering::Relaxed);
        PRE_CONV_HOPS.fetch_add(pre_hops, Ordering::Relaxed);
        POST_CONV_HOPS.fetch_add(post_hops, Ordering::Relaxed);
        PRE_CONV_ADMITS.fetch_add(pre_admits, Ordering::Relaxed);
        POST_CONV_ADMITS.fetch_add(post_admits, Ordering::Relaxed);
        NDC_I8.fetch_add(ndc_i8_local, Ordering::Relaxed);

        // ── Post-hoc f32 rerank ───────────────────────────────────────
        // Top `k · RERANK_FACTOR` from quantized PQ → f32 truth via
        // DistanceStream<Q::TruthDistanceFn, N> in one batched call →
        // sort → emit top-k.
        let beam_n = scratch.pq.size();
        let num_check = (k * RERANK_FACTOR).min(beam_n);
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
            vector::DistanceStream::<Q::TruthDistanceFn, N>::new(
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
        unsafe {
            let dst = out.as_mut_ptr();
            let src = rerank_buf.as_ptr();
            let full_chunks = n_out / 10;
            for c in 0..full_chunks {
                let base = c * 10;
                *dst.add(base) = (*src.add(base)).0;
                *dst.add(base + 1) = (*src.add(base + 1)).0;
                *dst.add(base + 2) = (*src.add(base + 2)).0;
                *dst.add(base + 3) = (*src.add(base + 3)).0;
                *dst.add(base + 4) = (*src.add(base + 4)).0;
                *dst.add(base + 5) = (*src.add(base + 5)).0;
                *dst.add(base + 6) = (*src.add(base + 6)).0;
                *dst.add(base + 7) = (*src.add(base + 7)).0;
                *dst.add(base + 8) = (*src.add(base + 8)).0;
                *dst.add(base + 9) = (*src.add(base + 9)).0;
            }
            for i in (full_chunks * 10)..n_out {
                *dst.add(i) = (*src.add(i)).0;
            }
            out.set_len(n_out);
        }
        Ok(out)
    }

    /// Parallel batch wrapper for [`Self::search_mips_q`].
    ///
    /// Uses `par_chunks(BATCH)` instead of `par_iter()` so each rayon
    /// worker processes a contiguous block of queries serially. At
    /// low L (queries finish in ~60 µs each), per-task overhead in
    /// rayon's work-stealing scheduler dominates if every query
    /// becomes its own task; batching amortises scheduling cost
    /// across `BATCH` queries, keeps thread-local scratch hot, and
    /// reduces low-L QPS variance materially.
    ///
    /// `BATCH` is L-adaptive. Per-query wall scales roughly linearly
    /// with L (each L step adds ~1 visit's worth of i8 dotp + scratch
    /// maintenance), so a single fixed BATCH leaves per-chunk wall
    /// time spanning an order of magnitude across the sweep — at
    /// large beam the last thread finishes ~one chunk after the
    /// others, and a smaller BATCH lets work-stealing rebalance that
    /// tail; at very narrow beam the per-task overhead dominates so a
    /// moderate BATCH amortizes it. The schedule below is from a 5×7
    /// (BATCH × L) probe on glove100 (1.18M, M2, 8 threads):
    ///
    ///   L < 24        → BATCH = 32    (narrow beam, dispatch dominates)
    ///   24 ≤ L < 64   → BATCH = 32    (dispatch-bound, even per-thread work)
    ///   64 ≤ L < 256  → BATCH = 16
    ///   256 ≤ L < 768 → BATCH = 8
    ///   L ≥ 768       → BATCH = 4     (long queries, wants finest grain)
    pub fn search_batch_mips_q<Q: crate::model::QuantSpec>(
        &self,
        queries: &[[f32; N]],
        q_ds: &crate::model::QuantizedDataset<Q, N>,
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
    ) -> ANNResult<Vec<Vec<u32>>> {
        self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        // STAGED_BATCH=<n> forces a fixed batch size (used by the
        // batch-vs-L probe). Otherwise use the shared L-adaptive
        // default from `super::in_mem_search::search_batch_size`.
        let batch: usize = match std::env::var("STAGED_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(b) if b > 0 => b,
            _ => super::in_mem_search::search_batch_size(search_list_size),
        };
        let n = queries.len();
        let mut results: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();

        // SAFETY: each chunk writes to a disjoint range of `results`;
        // rayon guarantees no two workers see the same chunk. We split
        // by index so the result slot indexing matches the input order.
        results
            .par_chunks_mut(batch)
            .zip(queries.par_chunks(batch))
            .for_each(|(out_chunk, q_chunk)| {
                for (out, query) in out_chunk.iter_mut().zip(q_chunk.iter()) {
                    *out = self
                        .search_mips_q::<Q>(
                            query,
                            q_ds,
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
