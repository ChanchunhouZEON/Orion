/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! `search_mips<N>` — **single-stage f32 MIPS beam search**, no
//! quantization, no per-hop rerank. Targets datasets where
//! quantization adds more overhead than it saves: low-dim (~25-50)
//! data where vertex bytes already fit in 1-2 cache lines and the
//! f32 IP kernel itself is the bottleneck.
//!
//! Adapted from the legacy `.bak` `search_mips` function but with
//! per-hop f32 distance compute routed through
//! [`vector::DistanceStream`] driven by [`vector::IpF32Distance`],
//! so the same MSHR-aware software-pipelined prefetch + 4-wide ILP
//! unroll the quantized path uses also covers this single-stage path.

use std::sync::atomic::Ordering;

use crate::StagedDiskANN;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::InMemScratchPool;
use diskann::common::ANNResult;
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use super::in_mem_search::{
    AlignedQuery, FLUSH_INTERVAL, NDC_F32, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS,
    PRE_CONV_HOPS, QUERY_COUNT, SETUP_NS, VISIT_COUNT, dstream_la_truth, insert_route_mul,
    linear_merge_mul,
};

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Single-stage f32 MIPS beam search (no quantization).
    pub fn search_mips(
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

        // Normalise query in-place into an aligned buffer.
        let mut q_norm: [f32; N] = *query;
        vector::l2_normalize_f32_inplace(&mut q_norm);
        let aligned = AlignedQuery(q_norm);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        // Build padded f32 query once per search — the IpF32Distance
        // kernel reads `chunks_per_vert · 32` bytes per vertex, so
        // trailing slots past N must be zero (over-reads on
        // `N % 8 != 0` datasets get multiplied by zero in MIPS).
        let f32_stride_bytes = N * 4;
        let f32_compute_bytes = f32_stride_bytes.div_ceil(32) * 32;
        let f32_padded_lanes = f32_compute_bytes / 4;
        scratch.q_query_f32_padded.clear();
        scratch.q_query_f32_padded.extend_from_slice(&aligned.0);
        scratch.q_query_f32_padded.resize(f32_padded_lanes, 0.0);

        scratch.seen.insert(entry);
        let entry_dist = unsafe {
            let v_arr = dataset.get_vertex_unchecked(entry);
            vector::distance_ip_vector_f32::<N>(&aligned.0, v_arr)
        };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));
        SETUP_NS.fetch_add(t_setup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut prev_admitted: usize = 1;
        let mut hops_since_flush: usize = 0;
        let mut visits: u64 = 0;
        let mut pre_hops: u64 = 0;
        let mut post_hops: u64 = 0;
        let mut pre_admits: u64 = 0;
        let mut post_admits: u64 = 0;
        let mut ndc_f32_local: u64 = 1; // entry-point distance counted above

        while scratch.pq.has_notvisited_node() {
            let id = scratch.pq.closest_notvisited().id as usize;
            visits += 1;

            if let Some(next) = scratch.pq.peek_notvisited() {
                graph.prefetch_node(next.id as usize);
                dataset.prefetch_vector(next.id);
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
            ndc_f32_local += n_unseen as u64;
            let pq_worst = if scratch.pq.size() >= search_list_size {
                scratch.pq[scratch.pq.size() - 1].distance
            } else {
                f32::MAX
            };

            let dataset_base_ptr = dataset.get_data().as_ptr() as *const u8;

            let hop_start = scratch.dist_buffer.len();
            scratch.dist_buffer.reserve(n_unseen);
            let pq_worst_local = pq_worst;
            // SAFETY: cmov-compact sink only ever advances `w` by 0
            // or 1; writes never go past `n_unseen` (= upper bound).
            let hop_admits: usize = unsafe {
                let base_out = scratch.dist_buffer.as_mut_ptr().add(hop_start);
                let mut w = 0usize;
                let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n_unseen);
                vector::DistanceStream::<vector::IpF32Distance, N>::new(
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
            prev_admitted = hop_admits;

            // Per-phase counters: split this hop's admits + visit by
            // converged state for diagnostic printing (mirrors the
            // mips-q path so the staged_sweep printout is uniform).
            if converged {
                post_hops += 1;
                post_admits += hop_admits as u64;
            } else {
                pre_hops += 1;
                pre_admits += hop_admits as u64;
            }

            // Legacy convergence-aware early exit.
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

        // Publish counters for the benchmarker. Plain f32 MIPS has no
        // i8 prefilter, so NDC_I8 stays untouched — every cmp goes to
        // NDC_F32 and the staged_sweep print line shows i8=0 here.
        VISIT_COUNT.fetch_add(visits, Ordering::Relaxed);
        QUERY_COUNT.fetch_add(1, Ordering::Relaxed);
        PRE_CONV_HOPS.fetch_add(pre_hops, Ordering::Relaxed);
        POST_CONV_HOPS.fetch_add(post_hops, Ordering::Relaxed);
        PRE_CONV_ADMITS.fetch_add(pre_admits, Ordering::Relaxed);
        POST_CONV_ADMITS.fetch_add(post_admits, Ordering::Relaxed);
        NDC_F32.fetch_add(ndc_f32_local, Ordering::Relaxed);

        // Emit top-k IDs (PQ already holds f32 distances, no rerank).
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

    /// Parallel batch wrapper for [`Self::search_mips`].
    pub fn search_batch_mips(
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

        let results: Vec<Vec<u32>> = queries
            .par_iter()
            .map(|query| {
                self.search_mips(
                    query,
                    k,
                    search_list_size,
                    window_size,
                    epsilon,
                    early_exit_limit,
                )
                .unwrap_or_default()
            })
            .collect();
        Ok(results)
    }
}
