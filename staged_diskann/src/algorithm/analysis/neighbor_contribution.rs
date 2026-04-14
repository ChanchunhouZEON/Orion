/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! **Neighbor Contribution Analysis** for Vamana graphs.
//!
//! Measures which neighbors (by distance-sorted position) actually contribute
//! to greedy search convergence. Two main analyses:
//!
//! 1. **Truncated-Degree Recall Sweep** — artificially cap each node's
//!    neighbor list at position *t* (distance-sorted), measure recall.
//! 2. **Per-Position Admission Rate** — during search, for each position in
//!    the distance-sorted neighbor list, track how often that neighbor is
//!    admitted to the priority queue (i.e., actually improves the candidate set).

use crate::model::PhasedGraph;

// ── Precompute distance-sorted neighbor lists ───────────────────────────────

/// For every node, sort its graph neighbors by distance (ascending) and return
/// the sorted adjacency list.
pub fn sort_neighbors_by_distance(
    graph: &PhasedGraph,
    flat_data: &[f32],
    dim: usize,
) -> Vec<Vec<u32>> {
    let n = graph.num_nodes();
    let mut sorted_adj = Vec::with_capacity(n);

    for node in 0..n {
        let pa = &flat_data[node * dim..(node + 1) * dim];
        let neighbors = graph.neighbors(node);
        let mut nbrs_with_dist: Vec<(u32, f32)> = neighbors
            .iter()
            .map(|&nbr| {
                let pb = &flat_data[nbr as usize * dim..(nbr as usize + 1) * dim];
                let d: f32 = pa.iter().zip(pb).map(|(a, b)| (a - b) * (a - b)).sum();
                (nbr, d)
            })
            .collect();
        nbrs_with_dist.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted_adj.push(nbrs_with_dist.into_iter().map(|(id, _)| id).collect());
    }

    sorted_adj
}

// ── Standalone greedy search (no StagedDiskANN dependency) ──────────────────

/// Minimal greedy beam search on a pre-sorted adjacency list.
///
/// `max_nbrs`: only use the first `max_nbrs` neighbors per node (truncation).
/// Set to `usize::MAX` for no truncation.
///
/// Returns the top-k result IDs.
pub fn greedy_search_truncated(
    sorted_adj: &[Vec<u32>],
    flat_data: &[f32],
    dim: usize,
    entry: u32,
    query: &[f32],
    k: usize,
    search_list_size: usize,
    max_nbrs: usize,
) -> Vec<u32> {
    let n = sorted_adj.len();
    let l = search_list_size;

    // Simple sorted-array priority queue + visited bitset.
    let mut pq: Vec<(u32, f32, bool)> = Vec::with_capacity(l + 1); // (id, dist, visited)
    let mut seen = vec![false; n];

    let dist_fn = |a: &[f32], b_idx: usize| -> f32 {
        let b = &flat_data[b_idx * dim..(b_idx + 1) * dim];
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    };

    // Insert entry.
    seen[entry as usize] = true;
    let ed = dist_fn(query, entry as usize);
    pq.push((entry, ed, false));

    loop {
        // Find closest not-visited.
        let pos = pq.iter().position(|e| !e.2);
        let Some(cur_pos) = pos else { break };
        pq[cur_pos].2 = true;
        let cur_id = pq[cur_pos].0;

        // Expand neighbors (truncated).
        let nbrs = &sorted_adj[cur_id as usize];
        let limit = nbrs.len().min(max_nbrs);
        for &nn in &nbrs[..limit] {
            if seen[nn as usize] {
                continue;
            }
            seen[nn as usize] = true;

            let d = dist_fn(query, nn as usize);

            // Insert if PQ not full or better than worst.
            if pq.len() < l || d < pq.last().unwrap().1 {
                // Binary search insertion.
                let ins = pq
                    .binary_search_by(|e| e.1.partial_cmp(&d).unwrap())
                    .unwrap_or_else(|x| x);
                pq.insert(ins, (nn, d, false));
                if pq.len() > l {
                    pq.pop();
                }
            }
        }
    }

    pq.iter().take(k).map(|e| e.0).collect()
}

// ── Per-position contribution tracking ──────────────────────────────────────

/// Accumulated per-position statistics across many queries.
#[derive(Debug)]
pub struct PositionContributionStats {
    /// Number of queries profiled.
    pub num_queries: usize,
    /// Total node expansions (across all queries).
    pub total_expansions: usize,
    /// For each position i in the sorted neighbor list:
    /// - `visit_count[i]`: how many times position i was encountered during expansions
    /// - `unseen_count[i]`: how many times position i was unseen (not already visited)
    /// - `admitted_count[i]`: how many times position i was admitted to the PQ
    ///   (i.e., distance was good enough to enter the candidate list)
    /// - `last_useful[i]`: how many expansions had their last-admitted position at i
    pub visit_count: Vec<usize>,
    pub unseen_count: Vec<usize>,
    pub admitted_count: Vec<usize>,
    pub last_useful_pos: Vec<usize>,
    /// Histogram of "last useful position" across all expansions.
    /// `last_useful_hist[i]` = number of expansions whose last PQ-admitted neighbor
    /// was at position i.
    pub last_useful_hist: Vec<usize>,
}

/// Run instrumented greedy search, tracking per-position neighbor contribution.
///
/// For each node expansion, we iterate through its distance-sorted neighbors
/// and record which positions lead to PQ admission.
pub fn search_with_contribution(
    sorted_adj: &[Vec<u32>],
    flat_data: &[f32],
    dim: usize,
    entry: u32,
    query: &[f32],
    k: usize,
    search_list_size: usize,
    max_degree: usize,
    stats: &mut PositionContributionStats,
) -> Vec<u32> {
    let n = sorted_adj.len();
    let l = search_list_size;

    let mut pq: Vec<(u32, f32, bool)> = Vec::with_capacity(l + 1);
    let mut seen = vec![false; n];

    let dist_fn = |a: &[f32], b_idx: usize| -> f32 {
        let b = &flat_data[b_idx * dim..(b_idx + 1) * dim];
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    };

    seen[entry as usize] = true;
    let ed = dist_fn(query, entry as usize);
    pq.push((entry, ed, false));

    loop {
        let pos = pq.iter().position(|e| !e.2);
        let Some(cur_pos) = pos else { break };
        pq[cur_pos].2 = true;
        let cur_id = pq[cur_pos].0;

        let nbrs = &sorted_adj[cur_id as usize];
        stats.total_expansions += 1;

        let worst_dist = if pq.len() >= l {
            pq.last().unwrap().1
        } else {
            f32::INFINITY
        };

        let mut last_useful = None;

        for (pos_i, &nn) in nbrs.iter().enumerate() {
            if pos_i >= max_degree {
                break;
            }
            // Ensure stats vectors are large enough.
            if pos_i >= stats.visit_count.len() {
                let new_len = pos_i + 1;
                stats.visit_count.resize(new_len, 0);
                stats.unseen_count.resize(new_len, 0);
                stats.admitted_count.resize(new_len, 0);
                stats.last_useful_pos.resize(new_len, 0);
                stats.last_useful_hist.resize(new_len, 0);
            }
            stats.visit_count[pos_i] += 1;

            if seen[nn as usize] {
                continue;
            }
            seen[nn as usize] = true;
            stats.unseen_count[pos_i] += 1;

            let d = dist_fn(query, nn as usize);

            let admitted = pq.len() < l || d < worst_dist;
            if admitted {
                stats.admitted_count[pos_i] += 1;
                last_useful = Some(pos_i);

                let ins = pq
                    .binary_search_by(|e| e.1.partial_cmp(&d).unwrap())
                    .unwrap_or_else(|x| x);
                pq.insert(ins, (nn, d, false));
                if pq.len() > l {
                    pq.pop();
                }
            }
        }

        // Record last-useful position for this expansion.
        if let Some(lp) = last_useful {
            if lp >= stats.last_useful_hist.len() {
                stats.last_useful_hist.resize(lp + 1, 0);
            }
            stats.last_useful_hist[lp] += 1;
        }
    }

    stats.num_queries += 1;
    pq.iter().take(k).map(|e| e.0).collect()
}

impl PositionContributionStats {
    pub fn new() -> Self {
        Self {
            num_queries: 0,
            total_expansions: 0,
            visit_count: Vec::new(),
            unseen_count: Vec::new(),
            admitted_count: Vec::new(),
            last_useful_pos: Vec::new(),
            last_useful_hist: Vec::new(),
        }
    }
}

impl std::fmt::Display for PositionContributionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Neighbor Position Contribution ({} queries, {} expansions)",
            self.num_queries, self.total_expansions
        )?;
        writeln!(
            f,
            "\n  {:<6} {:<10} {:<10} {:<10} {:<12} {:<12} {:<14}",
            "Pos", "Visited", "Unseen", "Admitted", "Unseen%", "Admit%", "Admit/Unseen%"
        )?;
        writeln!(f, "  {}", "─".repeat(78))?;
        for i in 0..self.visit_count.len() {
            let v = self.visit_count[i];
            let u = self.unseen_count[i];
            let a = self.admitted_count[i];
            if v == 0 {
                continue;
            }
            let unseen_pct = u as f64 / v as f64 * 100.0;
            let admit_pct = a as f64 / v as f64 * 100.0;
            let admit_unseen_pct = if u > 0 {
                a as f64 / u as f64 * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "  {:<6} {:<10} {:<10} {:<10} {:<12.1} {:<12.1} {:<14.1}",
                i, v, u, a, unseen_pct, admit_pct, admit_unseen_pct
            )?;
        }

        writeln!(f, "\n  Last-useful position distribution:")?;
        let total_with_useful: usize = self.last_useful_hist.iter().sum();
        let mut cumulative = 0usize;
        for (i, &cnt) in self.last_useful_hist.iter().enumerate() {
            if cnt > 0 {
                cumulative += cnt;
                writeln!(
                    f,
                    "    pos {:<3}: {} ({:.1}%, cumul {:.1}%)",
                    i,
                    cnt,
                    cnt as f64 / total_with_useful.max(1) as f64 * 100.0,
                    cumulative as f64 / total_with_useful.max(1) as f64 * 100.0,
                )?;
            }
        }

        Ok(())
    }
}
