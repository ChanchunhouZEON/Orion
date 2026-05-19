/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! `search_l2_u8_q<N>` — **single-stage u8 L2 beam search + post-hoc
//! f32 rerank**, mirrors [`search_mips_q`] for the SIFT-family L2
//! path with the L2 path's 3-hop loop peel preserved.
//!
//! Pipeline (per query):
//!   1. Insert entry into PQ with **u8 quantized** L2 distance.
//!   2. **Peel** 3 hops: each pops the closest unvisited, expands
//!      neighbours, computes u8 distances via
//!      [`vector::DistanceStream<L2U8Distance, N>`], cmov-compacts
//!      admits (those beating the current PQ tail) into `dist_buffer`,
//!      sorts, and `batch_merge`s into PQ. Skips DCC / early-exit
//!      because the PQ isn't full yet (loosely-constrained pq_worst
//!      makes the prefilter degenerate).
//!   3. Main loop hops: same single-stage u8 stream + cmov-compact +
//!      flush via 3-way routing every `FLUSH_INTERVAL[converged]`
//!      hops, with DCC convergence and early-exit countdown active.
//!   4. **Post-hoc rerank**: take top `k · RERANK_FACTOR` from the
//!      quantized PQ, recompute f32 truth via
//!      `DistanceStream<L2F32Distance, N>` in one batched call, sort,
//!      emit top-k.
//!
//! PQ holds **u8 quantized** L2 distances throughout the search loop —
//! same shape as `search_mips_q` but with `L2U8` instead of `MipsI8`.
//! That gives ~4× memory bandwidth savings on the per-hop distance
//! reads vs the mixed Stage-1+Stage-2 shape in `search_l2_u8`, at the
//! cost of search-quality drift from quantization noise (the rerank
//! at the end recovers the top-k).

use std::sync::atomic::Ordering;

use crate::StagedDiskANN;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::{InMemScratchPool, InMemSearchScratch};
use diskann::common::ANNResult;
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use super::in_mem_search::{
    AlignedQuery, FLUSH_INTERVAL, dstream_la_q, dstream_la_truth, insert_route_mul,
    linear_merge_mul, use_filter,
};

use super::in_mem_search::{
    NDC_F32, NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, QUERY_COUNT,
    SETUP_NS, VISIT_COUNT,
};

/// Match `search_mips_q`'s rerank factor (PA's `rerank_factor=2` for
/// `quantize_mode=1`).
const RERANK_FACTOR: usize = 2;

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// One peeled hop in the warm-up window (hops 0..=2). Mirrors
    /// `expand_peeled_hop_l2` but uses the **u8 quantized** L2 stream
    /// + writes u8 distances to the PQ. The peel's role is the same:
    /// fill the PQ with close-by candidates before DCC/early-exit
    /// activate, exploiting that the PQ tail is `MAX` or only loosely
    /// constrained so every admit passes.
    ///
    /// Returns `Some(admitted)` on a normal hop, `None` if the PQ has
    /// no unvisited node left (peel breaks early).
    #[inline(always)]
    fn expand_peeled_hop_l2_q(
        &self,
        q_ds: &crate::model::QuantizedDataset<crate::model::L2U8, N>,
        q_query_padded_ptr: *const u8,
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

        let q_stride_bytes = crate::model::QuantizedDataset::<crate::model::L2U8, N>::STRIDE
            * std::mem::size_of::<u8>();
        let q_base_ptr = q_ds.data.as_slice().as_ptr() as *const u8;

        let admitted: usize = unsafe {
            let base_out = scratch.dist_buffer.as_mut_ptr();
            let mut w = 0usize;
            let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n);
            vector::DistanceStream::<vector::L2U8Distance, N>::new(
                q_base_ptr,
                q_stride_bytes,
                q_stride_bytes,
                q_query_padded_ptr,
                id_in,
                dstream_la_q(),
            )
            .run(|id, qd| {
                base_out.add(w).write(DNeighbor::new(id, qd));
                w += (qd < pq_worst) as usize;
            });
            scratch.dist_buffer.set_len(w);
            w
        };
        // Counters: every peel cmp is a u8 compute; sink writes the
        // admit count directly to PQ via batch_merge below.
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

    /// L2-Q beam search: PQ holds **u8 quantized** distances, with the
    /// L2 path's 3-hop peel preserved + `search_mips_q`-style post-hoc
    /// f32 rerank. Targets SIFT where the u8 quantization noise is
    /// small enough that PA-style "search in u8, rerank top
    /// `k·RERANK_FACTOR` in f32" delivers within recall of the
    /// per-hop-rerank `search_l2_u8` shape but at materially lower
    /// per-hop bandwidth (no f32 reads in the steady state).
    pub fn search_l2_u8_q(
        &self,
        query: &[f32; N],
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
        let aligned = AlignedQuery(*query);

        // Quantized dataset (lazily built / loaded from `.qds` sidecar).
        let q_ds = self.ensure_quantized_dataset();
        let q_query_padded = q_ds.quantize_query_padded(&aligned.0);
        // Short query (`[u8; N]`) used for the entry distance.
        let q_query = q_ds.quantize_query(&aligned.0);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        // Entry distance must match the per-hop scale (u8), so the PQ
        // tail comparator stays apples-to-apples throughout the loop.
        scratch.seen.insert(entry);
        let entry_dist = unsafe { q_ds.qdist(entry, &q_query) };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));
        SETUP_NS.fetch_add(t_setup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut pre_hops: u64 = 0;
        let mut post_hops: u64 = 0;
        let mut pre_admits: u64 = 0;
        let mut post_admits: u64 = 0;

        // Precompute u8 stride / base / lookahead — invariant across
        // peel + main loop, and used by every DistanceStream::new.
        let q_stride_bytes = crate::model::QuantizedDataset::<crate::model::L2U8, N>::STRIDE
            * std::mem::size_of::<u8>();
        let q_base_ptr = q_ds.data.as_slice().as_ptr() as *const u8;
        let lookahead_lines = dstream_la_q();
        let q_query_padded_ptr = q_query_padded.as_ptr() as *const u8;

        // Peel the first 3 hops. Same justification as `search_l2_u8`:
        // PQ transitions empty → partial → full, convergence can't
        // fire, the incoming batch is large enough that 3-way merge
        // routing always picks linear `batch_merge`. Every peel hop
        // updates DCC/early-exit *state* (so when the main loop kicks
        // in the windows are pre-populated) but doesn't gate on them.
        let mut prev_admitted: usize = 1;
        'peel: {
            match self.expand_peeled_hop_l2_q(
                q_ds,
                q_query_padded_ptr,
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
            match self.expand_peeled_hop_l2_q(
                q_ds,
                q_query_padded_ptr,
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
            match self.expand_peeled_hop_l2_q(
                q_ds,
                q_query_padded_ptr,
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

            // ── Optional running-mean filter (USE_FILTER=1) ──
            // Mirrors `search_l2_u8`'s PA-style filter, but cheaper here:
            // L2-Q's PQ already holds u8 quantized distances, so the
            // frontier mean is `sum(pq[i].distance) / pq.size()` with
            // no extra qdist() recomputation. The mean tightens
            // admission below `pq_worst` (mean ≤ worst), reducing
            // marginal admits that would just fall out of the rerank
            // pool anyway. Only active once the PQ would actually
            // overflow this hop — the same `pq_can_overflow` gate as
            // the L2 path.
            let hop_start = scratch.dist_buffer.len();
            // SAFETY: cmov-compact sink only ever advances `w` by 0
            // or 1; writes never go past `n_unseen` (= upper bound).
            let hop_admits: usize = unsafe {
                let base_out = scratch.dist_buffer.as_mut_ptr().add(hop_start);
                let mut w = 0usize;
                let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n_unseen);
                vector::DistanceStream::<vector::L2U8Distance, N>::new(
                    q_base_ptr,
                    q_stride_bytes,
                    q_stride_bytes,
                    q_query_padded_ptr,
                    id_in,
                    lookahead_lines,
                )
                .run(|id, qd| {
                    base_out.add(w).write(DNeighbor::new(id, qd));
                    w += (qd < pq_worst) as usize;
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
        // Top `k · RERANK_FACTOR` from the u8 PQ → f32 truth via
        // DistanceStream<L2F32Distance, N> in one batched call → sort
        // → emit top-k.
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

    /// Parallel batch wrapper for [`Self::search_l2_u8_q`]. Same
    /// L-adaptive `par_chunks(BATCH)` shape as `search_batch_l2_u8` /
    /// `search_batch_mips_q` so every search path shares the rayon
    /// dispatch heuristic.
    pub fn search_batch_l2_u8_q(
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
                        .search_l2_u8_q(
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
