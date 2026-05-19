/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! `search_l2_u8<N>` — **L2 squared-Euclidean beam search with i8
//! prefilter + per-hop f32 rerank**, the SIFT-family path.
//!
//! Pipeline (per hop):
//!   1. Pop closest unvisited from PQ.
//!   2. Expand neighbours (local + remote / local + extra after DCC
//!      converges).
//!   3. **Stage-1**: stream u8 quantized squared-L2 distances over
//!      unseen neighbours via [`vector::DistanceStream`] driven by
//!      `L2U8::QuantDistanceFn`, cmov-compact survivors (those whose
//!      `qd ≤ pq_worst · slope² · Q_SLACK`) back into `id_scratch`
//!      in-place.
//!   4. **Stage-2**: stream f32 truth squared-L2 over survivors via
//!      `DistanceStream<L2F32Distance>`, cmov-compact admits
//!      (`dist < pq_worst`) into `dist_buffer`.
//!   5. Flush via 3-way routing.
//!
//! PQ holds **f32** distances throughout (not quantized — L2 path
//! has historically used per-hop rerank since SIFT's quantization
//! noise is small and adding rerank is essentially free).
//!
//! Adapted from the legacy `.bak` `search` (L2) but with both stages
//! routed through `DistanceStream` for uniform prefetch + 4-wide ILP.

use crate::StagedDiskANN;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::{InMemScratchPool, InMemSearchScratch};
use diskann::common::ANNResult;
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use super::in_mem_search::{
    AlignedQuery, FLUSH_INTERVAL, NDC_F32, NDC_I8, QUERY_COUNT, VISIT_COUNT, dstream_la_q,
    dstream_la_truth, insert_route_mul, linear_merge_mul, search_batch_size_l2, use_filter,
};
use crate::algorithm::in_mem_search::{
    POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, SETUP_NS,
};
use std::sync::atomic::Ordering;

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// One peeled hop in the warm-up window (hops 0..=2). During this
    /// window the PQ transitions empty → partial → full; convergence
    /// can't fire and the incoming batch size is almost always >= L/2,
    /// so 3-way merge routing deterministically picks linear
    /// `batch_merge`. Skipping Stage-1 u8 prefilter here cuts a
    /// nearly-useless filter pass — `pq_worst` is either MAX (100%
    /// pass) or only loosely constrained, so the filter rejects almost
    /// nothing.
    ///
    /// Mirrors the legacy `expand_peeled_hop_l2` shape but routes f32
    /// truth through `DistanceStream<L2F32Distance>` for the same
    /// queue-driven prefetch the main loop uses.
    ///
    /// Returns `Some(admitted)` on a normal hop, `None` if the PQ has
    /// no unvisited node left (peel breaks early).
    #[inline(always)]
    fn expand_peeled_hop_l2(
        &self,
        q_query_padded_ptr: *const u8,
        f32_stride_bytes: usize,
        f32_compute_bytes: usize,
        search_list_size: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Option<usize> {
        if !scratch.pq.has_notvisited_node() {
            return None;
        }
        let dataset = &self.dataset;
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

        // No queue construction — DistanceStream resolves prfm
        // addresses inline from (base_ptr, ids[v], stride, line).
        let dataset_base_ptr = dataset.get_data().as_ptr() as *const u8;

        // No reserve — `dist_buffer` is pre-sized at scratch construction
        // to `search_list_size + MAX_GRAPH_DEGREE × MAX_FLUSH_INTERVAL`,
        // which dominates `n` (≤ MAX_GRAPH_DEGREE) for any peel hop.
        let admitted: usize = unsafe {
            let base_out = scratch.dist_buffer.as_mut_ptr();
            let mut w = 0usize;
            let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n);
            vector::DistanceStream::<vector::L2F32Distance, N>::new(
                dataset_base_ptr,
                f32_stride_bytes,
                f32_compute_bytes,
                q_query_padded_ptr,
                id_in,
                dstream_la_truth(),
            )
            .run(|id, dist| {
                base_out.add(w).write(DNeighbor::new(id, dist));
                w += (dist < pq_worst) as usize;
            });
            scratch.dist_buffer.set_len(w);
            w
        };
        // No Stage-1 ran on peel: every unseen f32 cmp = visit.
        NDC_F32.fetch_add(n as u64, Ordering::Relaxed);
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

    /// L2 squared-Euclidean beam search with u8 prefilter + per-hop
    /// f32 rerank. Targets SIFT-family datasets where `slope²` is
    /// small enough that the prefilter cutoff `pq_worst · slope² ·
    /// Q_SLACK` rejects most non-admits while letting the f32 rerank
    /// catch the few quantization-noise misses.
    pub fn search_l2_u8(
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

        // L2 path doesn't normalise the query.
        let q_ds = self.ensure_quantized_dataset();
        let q_query_padded = q_ds.quantize_query_padded(&aligned.0);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        // Build padded f32 query once per search (Stage-2 needs zero-pad).
        let f32_stride_bytes = N * 4;
        let f32_compute_bytes = f32_stride_bytes.div_ceil(32) * 32;
        let f32_padded_lanes = f32_compute_bytes / 4;
        scratch.q_query_f32_padded.clear();
        scratch.q_query_f32_padded.extend_from_slice(&aligned.0);
        scratch.q_query_f32_padded.resize(f32_padded_lanes, 0.0);

        scratch.seen.insert(entry);
        let entry_dist = unsafe {
            let v_arr = dataset.get_vertex_unchecked(entry);
            vector::distance_l2_vector_f32::<N>(&aligned.0, v_arr)
        };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));

        SETUP_NS.fetch_add(t_setup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut pre_hops: u64 = 0;
        let mut post_hops: u64 = 0;
        let mut pre_admits: u64 = 0;
        let mut post_admits: u64 = 0;

        // Peel the first 3 hops. PQ transitions empty → partial → full
        // during this window; convergence is impossible and the 3-way
        // merge routing always resolves to linear `batch_merge`.
        // Skipping Stage-1 u8 prefilter here cuts a nearly-useless
        // filter pass (pq_worst is MAX or only loosely constrained).
        // Take a stable raw pointer to the padded query bytes — the
        // Vec backing them isn't mutated for the rest of the search,
        // so passing the ptr to the helper avoids a borrow conflict
        // with the `&mut scratch` argument.
        let q_query_padded_ptr = scratch.q_query_f32_padded.as_ptr() as *const u8;
        let mut prev_admitted: usize = 1;
        'peel: {
            match self.expand_peeled_hop_l2(
                q_query_padded_ptr,
                f32_stride_bytes,
                f32_compute_bytes,
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
            match self.expand_peeled_hop_l2(
                q_query_padded_ptr,
                f32_stride_bytes,
                f32_compute_bytes,
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
            match self.expand_peeled_hop_l2(
                q_query_padded_ptr,
                f32_stride_bytes,
                f32_compute_bytes,
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

            // ── Stage 1: optional u8 prefilter via DistanceStream ────
            //
            // When `STAGED_USE_FILTER=1`, run the u8 prefilter with
            // PA's running-mean threshold (mirrors `beamSearch.h:
            // 130–145`): every hop where the frontier is full,
            // recompute the **mean of the frontier's u8 distances**
            // whenever the frontier-worst id changes, accumulate
            // into a running sum, and use the sum/count as the
            // per-hop u8 cutoff. Survivors get cmov-compacted in
            // place into `id_scratch[..pre_kept]`.
            //
            // When the env is unset (default): **skip the u8
            // prefilter entirely**. All `n_unseen` candidates flow
            // straight into Stage-2 f32 rerank — equivalent to PA's
            // default config (`use_filtering=false`, no quantized
            // prefilter). Saves the u8 DistanceStream pass + the
            // running-mean compute, at the cost of more f32 cmps.

            // Fast path: skip Stage-1 prefilter when admitting every
            // unseen candidate would still leave the PQ below
            // capacity — i.e. `n_unseen + pq.size() < L`. In that
            // window `pq_worst` is `f32::MAX`, the prefilter
            // threshold degenerates to MAX, no candidate gets
            // rejected, and the filter state isn't even updated
            // (the running-mean branch only runs when frontier_full).
            // Running Stage-1 here is pure waste.
            //
            // Slightly stricter than `pq.size() < L` (the equivalent
            // "PQ not yet full" test): this version also keeps the
            // prefilter active on the *boundary hop* where the PQ
            // crosses capacity mid-admission, paying one hop of
            // throwaway u8 work but starting the running-mean state
            // accumulation at the same hop the threshold becomes
            // meaningful.
            let pq_can_overflow = n_unseen + scratch.pq.size() >= search_list_size;
            let pre_kept: usize = if use_filter() && pq_can_overflow {
                // PA-style running-mean threshold: `width = frontier.size()`.
                // Recompute the frontier's mean u8 distance only when the
                // worst-id changes (cache invalidation), then accumulate
                // into a running sum and use sum/count as the per-hop
                // u8 cutoff.
                let q_threshold: f32 = {
                    let pq_size = scratch.pq.size();
                    let cur_worst_id = scratch.pq[pq_size - 1].id;
                    if scratch.last_worst_id != cur_worst_id {
                        let mut sum: f32 = 0.0;
                        for i in 0..pq_size {
                            let id = scratch.pq[i].id;
                            // SAFETY: id < num_nodes by pq invariant.
                            let qd =
                                unsafe { q_ds.qdist(id, &q_query_padded[..N].try_into().unwrap()) };
                            sum += qd;
                        }
                        scratch.filter_tail_mean = sum / pq_size as f32;
                        scratch.last_worst_id = cur_worst_id;
                    }
                    scratch.filter_threshold_sum += scratch.filter_tail_mean;
                    scratch.filter_threshold_count += 1;
                    scratch.filter_threshold_sum / scratch.filter_threshold_count as f32
                };

                let q_stride_bytes = crate::model::QuantizedDataset::<crate::model::L2U8, N>::STRIDE
                    * std::mem::size_of::<u8>();
                let lookahead_lines = dstream_la_q();
                let q_base_ptr = q_ds.data.as_slice().as_ptr() as *const u8;

                let mut pre_kept: usize = 0;
                // SAFETY: cmov-compact rule — id_out cursor never
                // advances past read cursor, so the in-place
                // mutation is monotonic.
                unsafe {
                    let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n_unseen);
                    let id_out = scratch.id_scratch.as_mut_ptr();
                    vector::DistanceStream::<vector::L2U8Distance, N>::new(
                        q_base_ptr,
                        q_stride_bytes,
                        q_stride_bytes,
                        q_query_padded.as_ptr() as *const u8,
                        id_in,
                        lookahead_lines,
                    )
                    .run(|id, qd| {
                        if qd <= q_threshold {
                            *id_out.add(pre_kept) = id;
                            pre_kept += 1;
                        }
                    });
                }
                // NDC: every Stage-1 visit is one u8 compute;
                // cmov-compact survivors go to Stage-2 f32 rerank.
                NDC_I8.fetch_add(n_unseen as u64, Ordering::Relaxed);
                pre_kept
            } else {
                // No prefilter — pass all `n_unseen` candidates to
                // Stage-2. id_scratch[..n_unseen] is already in the
                // original neighbor-expansion order (no cmov
                // rewrite happened).
                n_unseen
            };
            VISIT_COUNT.fetch_add(n_unseen as u64, Ordering::Relaxed);
            NDC_F32.fetch_add(pre_kept as u64, Ordering::Relaxed);

            // ── Stage 2: f32 truth via DistanceStream ─────────────────
            // No queue construction — DistanceStream resolves prfm
            // addresses inline.
            let dataset_base_ptr = dataset.get_data().as_ptr() as *const u8;

            let hop_start = scratch.dist_buffer.len();
            // No reserve — capacity is fixed at scratch construction
            // (see `InMemSearchScratch::new` for the sizing rationale).
            // `hop_start + pre_kept` always fits within the pre-allocated
            // capacity because `pre_kept ≤ n_unseen ≤ MAX_GRAPH_DEGREE`
            // and `hop_start ≤ MAX_GRAPH_DEGREE × MAX_FLUSH_INTERVAL`.
            let pq_worst_local = pq_worst;
            let hop_admits: usize = unsafe {
                let base_out = scratch.dist_buffer.as_mut_ptr().add(hop_start);
                let mut w = 0usize;
                let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), pre_kept);
                vector::DistanceStream::<vector::L2F32Distance, N>::new(
                    dataset_base_ptr,
                    f32_stride_bytes,
                    f32_compute_bytes,
                    scratch.q_query_f32_padded.as_ptr() as *const u8,
                    id_in,
                    dstream_la_truth(),
                )
                .run(|id, dist| {
                    base_out.add(w).write(DNeighbor::new(id, dist));
                    w += (dist < pq_worst_local) as usize;
                });

                scratch.dist_buffer.set_len(hop_start + w);
                w
            };

            // Per-phase counters: split this hop's admits + visit by
            // converged state for diagnostic printing.
            if converged {
                post_hops += 1;
                post_admits += hop_admits as u64;
            } else {
                pre_hops += 1;
                pre_admits += hop_admits as u64;
            }

            prev_admitted = hop_admits;

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

        // PQ already holds f32 distances — no rerank needed. Emit top-k.
        let n = scratch.pq.size().min(k);
        let mut out: Vec<u32> = Vec::with_capacity(k);
        unsafe {
            let dst = out.as_mut_ptr();
            let full_chunks = n / 10;
            for c in 0..full_chunks {
                let base = c * 10;
                *dst.add(base) = scratch.pq[base].id;
                *dst.add(base + 1) = scratch.pq[base + 1].id;
                *dst.add(base + 2) = scratch.pq[base + 2].id;
                *dst.add(base + 3) = scratch.pq[base + 3].id;
                *dst.add(base + 4) = scratch.pq[base + 4].id;
                *dst.add(base + 5) = scratch.pq[base + 5].id;
                *dst.add(base + 6) = scratch.pq[base + 6].id;
                *dst.add(base + 7) = scratch.pq[base + 7].id;
                *dst.add(base + 8) = scratch.pq[base + 8].id;
                *dst.add(base + 9) = scratch.pq[base + 9].id;
            }
            for i in (full_chunks * 10)..n {
                *dst.add(i) = scratch.pq[i].id;
            }
            out.set_len(n);
        }
        Ok(out)
    }

    /// Parallel batch wrapper for [`Self::search_l2_u8`].
    ///
    /// Uses the same L-adaptive `par_chunks` BATCH as MIPS-Q
    /// (see [`super::in_mem_search::search_batch_size`]) to amortize
    /// rayon dispatch overhead at low L while keeping work-stealing
    /// granularity fine at high L. `STAGED_BATCH=<n>` env override
    /// forces a fixed batch size for sweeps.
    pub fn search_batch_l2_u8(
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
            _ => search_batch_size_l2(search_list_size),
        };

        let n = queries.len();
        let mut results: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        results
            .par_chunks_mut(batch)
            .zip(queries.par_chunks(batch))
            .for_each(|(out_chunk, q_chunk)| {
                for (out, query) in out_chunk.iter_mut().zip(q_chunk.iter()) {
                    *out = self
                        .search_l2_u8(
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
