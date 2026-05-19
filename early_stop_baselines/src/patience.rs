/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Patience-based early-stop baseline.
//!
//! Single-phase DiskANN-style greedy beam search that terminates after
//! `patience_k` consecutive steps with zero new top-L admissions. This is the
//! simple LAET/AdaptNN-style baseline: stopping is driven purely by "no recent
//! progress", with no learned model and no explicit convergence threshold.

use diskann::common::ANNResult;
use diskann::index::InmemIndex;
use diskann::model::{Neighbor, Vertex};
use rayon::prelude::*;
use vector::FullPrecisionDistance;

use crate::scratch::BaselineScratch;

/// State for the patience rule.
pub struct PatienceChecker {
    k: usize,
    consecutive_no_admit: usize,
}

impl PatienceChecker {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            consecutive_no_admit: 0,
        }
    }

    pub fn reset(&mut self) {
        self.consecutive_no_admit = 0;
    }

    /// Update after an expansion step; returns `true` if search should stop.
    #[inline]
    pub fn should_exit(&mut self, num_admitted: usize) -> bool {
        if num_admitted == 0 {
            self.consecutive_no_admit += 1;
            self.consecutive_no_admit >= self.k
        } else {
            self.consecutive_no_admit = 0;
            false
        }
    }
}

/// Single-query greedy beam search with patience-based early stop.
///
/// - `patience_k == 0` disables the rule: the search runs until the priority
///   queue is exhausted (matches "no early stop" ground truth).
/// - Expands every graph neighbor at every step — no two-phase switch, no
///   rerank-candidate set; this matches the DiskANN search loop that LAET and
///   AdaptNN plug into.
pub fn search_patience<T, const N: usize>(
    index: &InmemIndex<T, N>,
    query: &[T; N],
    k: usize,
    search_list_size: usize,
    patience_k: usize,
    scratch: &mut BaselineScratch,
) -> ANNResult<Vec<u32>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    scratch.prepare(search_list_size);

    let query_vertex = Vertex::<T, N>::new(query, u32::MAX);
    let metric = index.configuration.dist_metric;

    // Seed with the index entry point.
    let entry = index.start;
    let entry_vertex = index.dataset.get_vertex(entry)?;
    let entry_dist = query_vertex.compare(&entry_vertex, metric);
    scratch.seed_entry(entry, entry_dist);

    let mut patience = PatienceChecker::new(patience_k.max(1));

    while scratch.best_candidates.has_notvisited_node() {
        let closest = scratch.best_candidates.closest_notvisited();

        // Collect unseen neighbors into the staging buffer. Bind the RwLock
        // read-guard to a local so its borrow outlives the neighbor iteration.
        scratch.id_buffer.clear();
        let guard = index.final_graph.read_vertex_and_neighbors(closest.id)?;
        let max_vertex_id = index.configuration.max_points + index.configuration.num_frozen_pts;
        for &id in guard.get_neighbors() {
            if (id as usize) >= max_vertex_id {
                continue;
            }
            if scratch.visited.insert(id) {
                scratch.id_buffer.push(id);
            }
        }
        drop(guard);

        // Capture PQ worst before this batch so we can count admissions in
        // a single distance-compute pass (matching the staged_diskann loop).
        let pq_worst_before = if scratch.best_candidates.size() >= search_list_size {
            scratch.best_candidates[scratch.best_candidates.size() - 1].distance
        } else {
            f32::MAX
        };
        let mut admitted = 0usize;

        if let Some(&first) = scratch.id_buffer.first() {
            index.dataset.prefetch_vector(first);
        }
        for m in 0..scratch.id_buffer.len() {
            if m + 1 < scratch.id_buffer.len() {
                index.dataset.prefetch_vector(scratch.id_buffer[m + 1]);
            }
            let id = scratch.id_buffer[m];
            let vertex = index.dataset.get_vertex(id)?;
            let dist = query_vertex.compare(&vertex, metric);
            if scratch.best_candidates.size() < search_list_size || dist < pq_worst_before {
                admitted += 1;
            }
            scratch.best_candidates.insert(Neighbor::new(id, dist));
        }

        if patience_k > 0 && patience.should_exit(admitted) {
            break;
        }
    }

    // Harvest top-k result IDs.
    let take = scratch.best_candidates.size().min(k);
    let mut ids = Vec::with_capacity(take);
    for i in 0..take {
        ids.push(scratch.best_candidates[i].id);
    }
    Ok(ids)
}

/// Parallel batch variant of [`search_patience`]. Each query is given its own
/// `BaselineScratch` owned by the calling closure so that rayon can fan out
/// freely; no shared pool, no locks.
pub fn search_patience_batch<T, const N: usize>(
    index: &InmemIndex<T, N>,
    queries: &[[T; N]],
    k: usize,
    search_list_size: usize,
    patience_k: usize,
) -> Vec<Vec<u32>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let num_nodes = index.configuration.max_points + index.configuration.num_frozen_pts;
    queries
        .par_iter()
        .map_init(
            || BaselineScratch::new(search_list_size, num_nodes),
            |scratch, query| {
                search_patience(index, query, k, search_list_size, patience_k, scratch)
                    .unwrap_or_default()
            },
        )
        .collect()
}
