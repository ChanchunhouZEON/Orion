/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! `search_rabitq_b4<N>` — Extended RaBitQ at **B=4 bits/dim**.
//!
//! Same overall pipeline as [`search_rabitq`] (peel + DCC + early
//! exit + post-hoc f32 rerank), but Stage-1 uses
//! [`RabitQ4Dataset::estimate_l2_sq`] which decodes 4-bit signed
//! values per dimension and scales by the per-vertex `tau_x`. The
//! tighter quantizer lifts the recall ceiling that 1-bit RaBitQ hits
//! at ~0.74 (SIFT) / 0.94 (GIST) — the paper's Theorem 5 puts the
//! ceiling at B=4 well above 0.99 on both.

use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::StagedDiskANN;
use crate::model::Neighbor as DNeighbor;
use crate::model::dataset::rabitq_b4_dataset::RabitQ4Dataset;
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
    /// One peeled hop with the B=4 estimator. Twin of
    /// [`Self::expand_peeled_hop_rabitq`] — only difference is the
    /// estimator call.
    #[inline(always)]
    fn expand_peeled_hop_rabitq_b4(
        &self,
        q_ds: &RabitQ4Dataset<N>,
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

    /// B=4 RaBitQ beam search. PQ holds RaBitQ-B4-estimated L2²
    /// distances in f32; post-hoc f32 rerank of top `k · RERANK_FACTOR`.
    pub fn search_rabitq_b4(
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

        let q_ds = self.ensure_quantized_dataset_rabitq_b4();

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

        scratch.seen.insert(entry);
        let entry_dist = q_ds.estimate_l2_sq(&rotated_q, q_norm_sq, entry);
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));
        SETUP_NS.fetch_add(t_setup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let mut pre_hops: u64 = 0;
        let mut post_hops: u64 = 0;
        let mut pre_admits: u64 = 0;
        let mut post_admits: u64 = 0;

        let mut prev_admitted: usize = 1;
        'peel: {
            match self.expand_peeled_hop_rabitq_b4(
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
            match self.expand_peeled_hop_rabitq_b4(
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
            match self.expand_peeled_hop_rabitq_b4(
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

            let hop_start = scratch.dist_buffer.len();
            let mut hop_admits: usize = 0;
            for &nn in scratch.id_scratch.iter() {
                let qd = q_ds.estimate_l2_sq(&rotated_q, q_norm_sq, nn);
                if qd < pq_worst {
                    scratch.dist_buffer.push(DNeighbor::new(nn, qd));
                    hop_admits += 1;
                }
            }
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

        QUERY_COUNT.fetch_add(1, Ordering::Relaxed);
        PRE_CONV_HOPS.fetch_add(pre_hops, Ordering::Relaxed);
        POST_CONV_HOPS.fetch_add(post_hops, Ordering::Relaxed);
        PRE_CONV_ADMITS.fetch_add(pre_admits, Ordering::Relaxed);
        POST_CONV_ADMITS.fetch_add(post_admits, Ordering::Relaxed);

        // Post-hoc f32 rerank — identical to `search_rabitq`.
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

    pub fn search_batch_rabitq_b4(
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
                        .search_rabitq_b4(
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
