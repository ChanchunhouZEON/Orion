/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Unified cascade beam search
//!
//! Single beam loop parameterised over the three-stage cascade
//! ([`PrefilterStage`], [`AdmissionStage`], [`RerankStage`]) —
//! every recipe `Cascade::default_for_dataset` ships routes
//! through this one function. See `super::utils` for the
//! shared per-thread counter machinery, search-loop tuning
//! constants, and diagnostic search variants (`search_diag`,
//! `search_profile`).

use crate::Orion;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::{InMemScratchPool, InMemSearchScratch};
use diskann::common::ANNResult;
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use super::stage::{
    AdmissionSession, AdmissionStage, PrefilterSession, PrefilterStage, RerankStage,
};
use super::utils::{
    AlignedQuery, FLUSH_INTERVAL, dstream_la_q, insert_route_mul, linear_merge_mul,
};
use super::{
    NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, QUERY_COUNT,
    RAW_VISIT_COUNT, SETUP_NS, VISIT_COUNT,
};

/// PA's `-rerank_factor 2` shape — top `k · 2` PQ entries get the
/// truth pass at end of beam. Constant here (rather than per-recipe)
/// because every concrete recipe we ship uses the same factor.
const RERANK_FACTOR: usize = 2;

/// Peel hops at search start — fixed warm-up window that bypasses
/// DCC, early-exit, and the prefilter. Default `3` matches the
/// legacy `expand_peeled_hop_l2_q` count. Set via
/// `ORION_PEEL_HOPS=0` to disable (useful on low-D recipes where
/// the search converges in tens of hops and the peel is overhead
/// relative to the main loop).
#[inline]
fn peel_hops() -> usize {
    use std::sync::OnceLock;
    static PEEL: OnceLock<usize> = OnceLock::new();
    *PEEL.get_or_init(|| {
        std::env::var("ORION_PEEL_HOPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3)
    })
}

/// Multiplicative slack on the prefilter threshold (mean over the PQ
/// of per-entry prefilter distances). Mirrors `jl_slack()` in the
/// legacy `in_mem_search_l2_q`; resolved once per process via
/// `OnceLock` so the env-var lookup stays off the inner loop.
///
/// Default `1.05` is the calibration sweep optimum for JL Sparse on
/// GIST 1M (0.1-1.4 pp recall loss, +35-53% iso-recall QPS). Other
/// prefilter implementations (e.g. RaBitQ-as-prefilter) may want
/// different defaults — when they land, refactor this into a per-
/// recipe constant.
#[inline]
fn prefilter_slack() -> f32 {
    use std::sync::OnceLock;
    static SLACK: OnceLock<f32> = OnceLock::new();
    *SLACK.get_or_init(|| {
        std::env::var("ORION_JL_SLACK")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|&v| v > 0.0)
            .unwrap_or(1.05)
    })
}

impl<const N: usize> Orion<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// **One peeled hop**: bypass prefilter / DCC / early-exit and
    /// run the admission stream directly. Used in the search-start
    /// warm-up window (hops 0..PEEL_HOPS); the PQ transitions
    /// empty → partial → full here, the prefilter threshold isn't
    /// yet meaningful (small PQ, mean dominated by outliers), and
    /// DCC / early-exit could fire on noisy admit-count signals.
    ///
    /// Returns `Some((admitted, n_unseen))` for the per-phase counter
    /// update + DCC state propagation, or `None` if there are no
    /// unvisited PQ entries (peel breaks early).
    ///
    /// Takes the per-query [`AdmissionSession`] (already opened by
    /// the caller); the session owns the padded query / norm-sq /
    /// short form, so no per-hop quantization happens here.
    #[inline]
    fn peel_hop_unified(
        &self,
        a_session: &dyn AdmissionSession,
        search_list_size: usize,
        scratch: &mut InMemSearchScratch,
        lookahead_lines: usize,
    ) -> Option<(usize, usize)> {
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
            return Some((0, 0));
        }

        let pq_worst = if scratch.pq.size() >= search_list_size {
            scratch.pq[scratch.pq.size() - 1].distance
        } else {
            f32::MAX
        };

        let admitted: usize = unsafe {
            let base_out = scratch.dist_buffer.as_mut_ptr();
            let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n);
            let w = a_session.admit_stream(id_in, base_out, pq_worst, lookahead_lines);
            scratch.dist_buffer.set_len(w);
            w
        };

        // The legacy peel sorts and batch_merges admits directly into
        // the PQ each hop — no per-hop flush deferral, no 3-way
        // routing decision. Since `n` is large (full graph degree)
        // and admits are all close-by, `batch_merge` (linear) is the
        // right move every time.
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
        Some((admitted, n))
    }

    /// Unified cascade beam search.
    ///
    /// Drives a single beam loop through the three stage traits.
    /// `prefilter` is optional (`None` = no rejection tier, e.g. on
    /// SIFT-class workloads where the admission kernel is already
    /// cheap enough). `admission` is mandatory (it defines the PQ-
    /// ranked distance metric). `rerank` is mandatory but can be
    /// [`NoRerank`](crate::algorithm::search::stage::rerank::NoRerank)
    /// if the admission tier already ranks at sufficient precision.
    ///
    /// # Stage interaction contract
    ///
    /// * **Peel** (PEEL_HOPS = 3) runs at search start to fill the
    ///   PQ before any threshold-state machinery activates.
    ///
    /// * **Prefilter threshold** is a multiplicative slack over the
    ///   running mean of prefilter distances on PQ entries. The mean
    ///   is incrementally re-computed only when `pq_back_id` changes
    ///   (mirrors PA `beamSearch.h:142`).
    ///
    /// * **Admission cutoff** is the PQ tail distance (`pq_worst`)
    ///   when the PQ is full, otherwise `f32::MAX`. The admission
    ///   stream cmov-compacts admits into the per-hop staging buffer.
    ///
    /// * **Rerank** runs once at end of beam over the top
    ///   `k · RERANK_FACTOR` PQ entries.
    ///
    /// # Generic monomorphisation
    ///
    /// The function body inlines every trait method call. With ~5-7
    /// distinct recipes in the dispatcher we get ~5-7 specialised
    /// hot loops — bloat budget ~20 KiB compiled text.
    #[allow(clippy::too_many_arguments)]
    pub fn search_unified<P, A, R>(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        prefilter: Option<&P>,
        admission: &A,
        rerank: &R,
    ) -> ANNResult<Vec<u32>>
    where
        P: ?Sized + PrefilterStage<N>,
        A: ?Sized + AdmissionStage<N>,
        R: ?Sized + RerankStage<N>,
    {
        let t_setup = std::time::Instant::now();
        let entry = self.entry;
        let graph = &self.graph;
        let aligned = AlignedQuery(*query);

        // ── Per-query stage state (sessions) ─────────────────────
        // Prefilter is optional: `None` ⇒ no `open` call, no
        // threshold recompute, no `filter_compact`. The
        // `if let Some(_) = p_session` branches below are loop-
        // invariant; LLVM hoists them.
        let p_session: Option<Box<dyn PrefilterSession + '_>> =
            prefilter.map(|p| p.open(&aligned.0));
        let a_session: Box<dyn AdmissionSession + '_> = admission.open(&aligned.0);

        // ── Scratch pool ─────────────────────────────────────────
        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });
        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        // ── Entry distance + insert ──────────────────────────────
        scratch.seen.insert(entry);
        let entry_dist = a_session.entry_distance(entry);
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));
        SETUP_NS.add(t_setup.elapsed().as_nanos() as u64);

        // ── Stack-local counter accumulators ──────────────────────
        // Same shape `search_mips_q` and the post-hoist
        // `search_l2_u8_q` use: bump locals per hop, fetch_add once
        // per query at the end. Sharded counters under the hood, so
        // even the per-query fetch_add is a non-contended u64 add
        // into the calling worker's slot.
        let mut pre_hops: u64 = 0;
        let mut post_hops: u64 = 0;
        let mut pre_admits: u64 = 0;
        let mut post_admits: u64 = 0;
        let mut visits: u64 = 0;
        let mut raw_visits: u64 = 0;
        let mut ndc_i8_local: u64 = 0;

        let lookahead_lines = dstream_la_q();

        // ── Peel hops ─────────────────────────────────────────────
        // Generic across recipes — every cascade benefits from
        // populating the PQ before its prefilter threshold becomes
        // meaningful. Even recipes with `NoPrefilter` benefit (DCC
        // and early-exit windows are pre-populated with sensible
        // admit counts before the main loop kicks in).
        let mut prev_admitted: usize = 1;
        let peel_count = peel_hops();
        'peel: {
            for _ in 0..peel_count {
                match self.peel_hop_unified(
                    a_session.as_ref(),
                    search_list_size,
                    scratch,
                    lookahead_lines,
                ) {
                    Some((admitted, n_unseen)) => {
                        scratch.dcc.update(prev_admitted);
                        scratch.early_exit.should_exit(false, prev_admitted);
                        prev_admitted = admitted.max(1);
                        pre_hops += 1;
                        pre_admits += admitted as u64;
                        visits += n_unseen as u64;
                        // Peel hops bypass the prefilter — raw and
                        // survivor counts are identical.
                        raw_visits += n_unseen as u64;
                        ndc_i8_local += n_unseen as u64;
                    }
                    None => break 'peel,
                }
            }
        }

        let mut hops_since_flush: usize = 0;

        // ── Main beam loop ────────────────────────────────────────
        while scratch.pq.has_notvisited_node() {
            let id = scratch.pq.closest_notvisited().id as usize;

            if let Some(next) = scratch.pq.peek_notvisited() {
                graph.prefetch_node(next.id as usize);
            }

            let converged = scratch.dcc.update(prev_admitted);

            // Expand unseen neighbours from the graph. Converged
            // hops use `rerank_candidates` (local + extra), which
            // pulls in the off-graph extras that lift recall in the
            // tail of the search.
            scratch.id_scratch.clear();
            if !converged {
                for &nn in graph.neighbors(id) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }
            } else if crate::algorithm::search::include_extras() {
                let (local, extra) = graph.rerank_candidates(id);
                for &nn in local.iter().chain(extra.iter()) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }
            } else {
                // Ablation toggle: with `INCLUDE_EXTRAS=false`, the
                // post-convergence branch falls back to the same
                // `local + remote` walk used pre-convergence — i.e.
                // the rerank-mode "substitute remote with extras"
                // never fires. Without this fallback, disabling
                // extras would collapse the converged branch to
                // `local` only, which drops the long-range remote
                // shortcuts and degrades worse than just "no extras."
                for &nn in graph.neighbors(id) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }
            }

            let pq_worst = if scratch.pq.size() >= search_list_size {
                scratch.pq[scratch.pq.size() - 1].distance
            } else {
                f32::MAX
            };

            // Capture the raw graph-expansion count BEFORE the
            // prefilter compacts it. If no prefilter runs this hop,
            // `raw_unseen == n_unseen` after the block below.
            let raw_unseen = scratch.id_scratch.len() as u64;

            // ── Optional prefilter ────────────────────────────────
            // Whole-PQ mean-of-prefilter-distance × slack threshold.
            // Recomputed only when `pq_back_id` changes — in steady
            // state the back turns over rarely so most hops do zero
            // recomputation work.
            let frontier_full = scratch.pq.size() >= search_list_size;
            if let Some(ps) = p_session.as_deref() {
                if frontier_full {
                    let pq_size = scratch.pq.size();
                    let pq_back_id = scratch.pq[pq_size - 1].id;
                    if scratch.jl_threshold_count == 0 || scratch.jl_last_worst_id != pq_back_id {
                        let mut tail_sum = 0.0f32;
                        for i in 0..pq_size {
                            tail_sum += ps.distance(scratch.pq[i].id);
                        }
                        scratch.jl_tail_mean = tail_sum / (pq_size as f32);
                        scratch.jl_last_worst_id = pq_back_id;
                    }
                    scratch.jl_threshold_sum += scratch.jl_tail_mean;
                    scratch.jl_threshold_count += 1;
                    let threshold = scratch.jl_threshold_sum / (scratch.jl_threshold_count as f32)
                        * prefilter_slack();

                    unsafe {
                        ps.filter_compact(&mut scratch.id_scratch, threshold, lookahead_lines);
                    }
                }
            }

            // ── Admission ─────────────────────────────────────────
            let n_unseen = scratch.id_scratch.len();
            let admit_cutoff = pq_worst;

            let hop_start = scratch.dist_buffer.len();
            let hop_admits: usize = unsafe {
                let base_out = scratch.dist_buffer.as_mut_ptr().add(hop_start);
                let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), n_unseen);
                let w = a_session.admit_stream(id_in, base_out, admit_cutoff, lookahead_lines);
                scratch.dist_buffer.set_len(hop_start + w);
                w
            };
            prev_admitted = hop_admits;

            // Per-phase counter split for diagnostic printing.
            if converged {
                post_hops += 1;
                post_admits += hop_admits as u64;
            } else {
                pre_hops += 1;
                pre_admits += hop_admits as u64;
            }
            visits += n_unseen as u64;
            raw_visits += raw_unseen;
            ndc_i8_local += n_unseen as u64;

            let should_exit = scratch.early_exit.should_exit(converged, hop_admits);
            let will_stop = should_exit | !scratch.pq.has_notvisited_node();

            // ── Flush ─────────────────────────────────────────────
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

        // ── Publish counters ─────────────────────────────────────
        QUERY_COUNT.add(1);
        PRE_CONV_HOPS.add(pre_hops);
        POST_CONV_HOPS.add(post_hops);
        PRE_CONV_ADMITS.add(pre_admits);
        POST_CONV_ADMITS.add(post_admits);
        VISIT_COUNT.add(visits);
        RAW_VISIT_COUNT.add(raw_visits);
        NDC_I8.add(ndc_i8_local);

        // ── Rerank ───────────────────────────────────────────────
        // `RerankStage::rerank` accepts a slice of PQ entries for
        // future flexibility, but current impls read directly from
        // `scratch.pq` (the unified loop guarantees `scratch.pq`
        // holds the post-search entries here).
        Ok(rerank.rerank(query, &[], k, RERANK_FACTOR, scratch))
    }

    /// Parallel batch wrapper. Mirrors the `search_batch_*` shape on
    /// every other search variant — rayon `par_iter_mut().zip(...)`
    /// with no manual chunking (empirically tied with PA's parlay
    /// scheduler at our work granularity).
    #[allow(clippy::too_many_arguments)]
    pub fn search_batch_unified<P, A, R>(
        &self,
        queries: &[[f32; N]],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        prefilter: Option<&P>,
        admission: &A,
        rerank: &R,
    ) -> ANNResult<Vec<Vec<u32>>>
    where
        P: ?Sized + PrefilterStage<N> + Sync,
        A: ?Sized + AdmissionStage<N> + Sync,
        R: ?Sized + RerankStage<N> + Sync,
    {
        self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let n = queries.len();
        let mut results: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        results
            .par_iter_mut()
            .zip(queries.par_iter())
            .for_each(|(out, query)| {
                *out = self
                    .search_unified(
                        query,
                        k,
                        search_list_size,
                        window_size,
                        epsilon,
                        early_exit_limit,
                        prefilter,
                        admission,
                        rerank,
                    )
                    .unwrap_or_default();
            });
        Ok(results)
    }
}
