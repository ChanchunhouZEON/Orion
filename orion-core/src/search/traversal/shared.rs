/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! The single computation body for immediate and suspendable graph access.
//! Only acquiring a node may suspend. Sessions and scratch stay query-owned;
//! distance streams, convergence, neighbor order and flush policy are unchanged.
use super::*;
use crate::model::graph::graph_access::{GraphAccess, NodeView};

/// Expands one initial beam node without filtering, awaiting its graph row before scoring unseen neighbors.
#[inline(always)]
async fn peel_hop<G: GraphAccess, O: SearchObserver>(
    graph: &G,
    a_session: &dyn AdmissionSession,
    search_list_size: usize,
    scratch: &mut InMemSearchScratch,
    lookahead_lines: usize,
    observer: &mut O,
) -> Result<Option<(usize, usize)>> {
    if !scratch.pq.has_notvisited_node() {
        return Ok(None);
    }

    let id = scratch.pq.closest_notvisited().id as usize;
    if let Some(next) = scratch.pq.peek_notvisited() {
        graph.prefetch_node(next.id as usize);
    }

    let node = graph.node(id).await?;
    scratch.id_scratch.clear();
    for &nn in node.neighbors() {
        if scratch.seen.insert(nn) {
            scratch.id_scratch.push(nn);
        }
    }
    let n = scratch.id_scratch.len();
    observer.expansion_started(id as u32, false, true);
    observer.expansion(false, &scratch.id_scratch);
    observer.prefilter_outcome(
        None,
        (scratch.pq.size() >= search_list_size).then(|| scratch.pq[scratch.pq.size() - 1].distance),
        0,
        0,
    );
    if n == 0 {
        return Ok(Some((0, 0)));
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

    observer.admission(n, &scratch.dist_buffer);

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
    observer.beam_updated(scratch.pq.neighbors());
    Ok(Some((admitted, n)))
}

/// Runs the shared adaptive beam walk, preserving query state across awaited graph reads.
///
/// Only graph access suspends here; admission and rerank sessions still consume their
/// prepared resident or mapped storage synchronously. Scratch remains exclusively
/// borrowed until completion or cancellation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn traverse<G: GraphAccess, O: SearchObserver, R>(
    graph: &G,
    entry: u32,
    scratch: &mut InMemSearchScratch,
    k: usize,
    search_list_size: usize,
    window_size: usize,
    epsilon: f32,
    early_exit_limit: usize,
    prefilter: &dyn PrefilterSession,
    a_session: &dyn AdmissionSession,
    mode: NeighborMode,
    observer: &mut O,
    t_setup: crate::search::diagnostics::metrics::SetupTimer,
    finish: impl FnOnce(&mut InMemSearchScratch, usize) -> R,
) -> Result<R> {
    scratch.prepare_for_query(search_list_size);
    scratch.scc.reconfigure(window_size, epsilon);
    scratch.early_exit.reconfigure(early_exit_limit);

    // ── Entry distance + insert ──────────────────────────────
    scratch.seen.insert(entry);
    let entry_dist = a_session.entry_distance(entry);
    scratch.pq.insert(DNeighbor::new(entry, entry_dist));
    observer.entry(entry);
    observer.scored_entry(entry, entry_dist);
    t_setup.record();

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
            match peel_hop(
                graph,
                a_session,
                search_list_size,
                scratch,
                lookahead_lines,
                observer,
            )
            .await?
            {
                Some((admitted, n_unseen)) => {
                    scratch.scc.update(prev_admitted);
                    scratch.early_exit.should_exit(false, prev_admitted);
                    prev_admitted = admitted;
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

        let node = graph.node(id).await?;
        let converged = scratch.scc.update(prev_admitted);
        observer.expansion_started(id as u32, converged, false);
        observer.before_expansion(
            id as u32,
            converged,
            scratch,
            prev_admitted,
            hops_since_flush,
        );

        // Expand unseen neighbours from the graph. Converged
        // hops use `rerank_candidates` (local + extra), which
        // pulls in the off-graph extras that lift recall in the
        // tail of the search.
        scratch.id_scratch.clear();
        if !converged || mode == NeighborMode::FullNeighbor {
            for &nn in node.neighbors() {
                if scratch.seen.insert(nn) {
                    scratch.id_scratch.push(nn);
                }
            }
        } else {
            let (local, extra) = node.rerank_candidates();
            let extra = if mode == NeighborMode::LocalExtra {
                extra
            } else {
                &[]
            };
            for &nn in local.iter().chain(extra.iter()) {
                if scratch.seen.insert(nn) {
                    scratch.id_scratch.push(nn);
                }
            }
        }
        observer.expansion(converged, &scratch.id_scratch);

        let pq_worst = if scratch.pq.size() >= search_list_size {
            scratch.pq[scratch.pq.size() - 1].distance
        } else {
            f32::MAX
        };

        // Capture the raw graph-expansion count BEFORE the
        // prefilter compacts it. If no prefilter runs this hop,
        // `raw_unseen == n_unseen` after the block below.
        let raw_unseen = scratch.id_scratch.len() as u64;

        // IDs come from the validated graph. Each stage owns its full
        // operation; NoPrefilter returns without reading or changing scratch.
        let filter_outcome = unsafe { prefilter.apply(scratch, search_list_size, lookahead_lines) };
        filter_outcome.observe(
            observer,
            (scratch.pq.size() >= search_list_size).then_some(pq_worst),
        );

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
        observer.admission(n_unseen, &scratch.dist_buffer[hop_start..]);
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
                        scratch
                            .pq
                            .batch_merge_gallop(&scratch.dist_buffer, &mut scratch.merge_scratch);
                    }
                    scratch.dist_buffer.clear();
                }
            }
            hops_since_flush = 0;
            observer.after_flush(scratch.pq.neighbors());
            observer.beam_updated(scratch.pq.neighbors());
        }

        if should_exit {
            observer.early_exit();
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
    let factor = rerank_factor();
    observer.rerank(scratch.pq.size().min(k.saturating_mul(factor)));
    Ok(finish(scratch, factor))
}
