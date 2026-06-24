/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Auto-calibration of convergence threshold and early exit limit.

use crate::Orion;
use crate::algorithm::search::utils::AlignedQuery;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::InMemSearchScratch;
use diskann::common::ANNResult;
use diskann::model::Vertex;
use vector::{FullPrecisionDistance, Metric};

/// Calibrated search parameters.
#[derive(Debug, Clone, Copy)]
pub struct CalibratedParams {
    pub threshold: f32,
    pub early_exit_limit: usize,
}

impl<const N: usize> Orion<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Calibrate convergence and early exit parameters from warmup queries.
    ///
    /// Runs full-graph search (no convergence, no early exit) and tracks:
    /// 1. Per-step admission rate → derive `threshold` from the second
    ///    inflection point of the rate curve.
    /// 2. The step at which each final top-k result was admitted →
    ///    derive `early_exit_limit` from P90 of remaining steps after
    ///    the last useful admission.
    pub fn calibrate(
        &self,
        warmup_queries: &[[f32; N]],
        search_list_size: usize,
        window_size: usize,
    ) -> ANNResult<CalibratedParams> {
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let k = 10usize;

        let mut scratch = InMemSearchScratch::new(search_list_size);

        let max_steps = search_list_size * 4;
        let mut pos_admit_count = vec![0u64; max_steps];
        let mut pos_total_count = vec![0u64; max_steps];
        // Per-hop admit counts across ALL queries × hops. Used to derive
        // the batch_merge-vs-per-insert admit threshold from the actual
        // distribution (bugfix: previous `admitted: bool` only counted
        // whether any admission happened, losing the per-hop count).
        let mut useful_gaps: Vec<usize> = Vec::new();
        let mut tail_gaps: Vec<usize> = Vec::new();

        for query in warmup_queries {
            let aligned = AlignedQuery(*query);
            let query_vertex = Vertex::new(&aligned.0, 0);
            scratch.prepare_for_query(search_list_size);

            scratch.seen.insert(entry);
            let entry_dist = {
                let v = dataset.get_vertex(entry)?;
                v.compare(&query_vertex, Metric::L2)
            };
            scratch.pq.insert(DNeighbor::new(entry, entry_dist));

            // Track: for each node ID, which step it was admitted.
            let mut admit_step: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::new();
            admit_step.insert(entry, 0);

            let mut step = 0usize;

            while scratch.pq.has_notvisited_node() {
                let neighbor = scratch.pq.closest_notvisited();
                let id = neighbor.id as usize;

                scratch.id_scratch.clear();
                for &nn in graph.neighbors(id) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }

                let n_unseen = scratch.id_scratch.len();
                let pq_worst = if scratch.pq.size() >= search_list_size {
                    scratch.pq[scratch.pq.size() - 1].distance
                } else {
                    f32::MAX
                };
                let mut admitted = false;

                for m in 0..n_unseen {
                    let nn = scratch.id_scratch[m];
                    let v = dataset.get_vertex(nn)?;
                    let dist = query_vertex.compare(&v, Metric::L2);
                    if dist < pq_worst || scratch.pq.size() < search_list_size {
                        admitted = true;
                        admit_step.insert(nn, step);
                    }
                    scratch.pq.insert(DNeighbor::new(nn, dist));
                }

                if step < max_steps {
                    pos_total_count[step] += 1;
                    if admitted {
                        pos_admit_count[step] += 1;
                    }
                }
                step += 1;
            }

            let total_steps = step;

            // Collect steps where top-k candidates were admitted, sorted.
            let top_k: Vec<u32> = (0..scratch.pq.size().min(k))
                .map(|i| scratch.pq[i].id)
                .collect();

            let mut useful_steps: Vec<usize> = top_k
                .iter()
                .filter_map(|id| admit_step.get(id))
                .copied()
                .collect();
            useful_steps.sort_unstable();
            useful_steps.dedup();

            // Gaps between consecutive useful admissions.
            for w in useful_steps.windows(2) {
                useful_gaps.push(w[1] - w[0]);
            }

            // Tail gap: from last useful admission to search end.
            if let Some(&last) = useful_steps.last() {
                tail_gaps.push(total_steps.saturating_sub(last));
            }
        }

        // ── Compute gap statistics ──
        useful_gaps.sort_unstable();
        tail_gaps.sort_unstable();

        let gap_p50 = percentile(&useful_gaps, 50);
        let gap_p90 = percentile(&useful_gaps, 90);
        let gap_p95 = percentile(&useful_gaps, 95);
        let tail_p50 = percentile(&tail_gaps, 50);

        // ── Derive threshold from per-step admission rate curve ──
        // Compute per-step admission rate, find the inflection point
        // where the rate stabilizes (second derivative ≈ 0).
        let valid_steps = pos_total_count
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(max_steps);
        let rates: Vec<f32> = (0..valid_steps)
            .map(|i| {
                if pos_total_count[i] > 0 {
                    pos_admit_count[i] as f32 / pos_total_count[i] as f32
                } else {
                    0.0
                }
            })
            .collect();

        // Smooth rates with sliding window.
        let sw = window_size.max(3);
        let smoothed: Vec<f32> = (0..rates.len())
            .map(|i| {
                let s = i.saturating_sub(sw / 2);
                let e = (i + sw / 2 + 1).min(rates.len());
                rates[s..e].iter().sum::<f32>() / (e - s) as f32
            })
            .collect();

        // Find second inflection: scan smoothed curve for the first position
        // where the first derivative changes from negative to near-zero
        // (rate stops decreasing = plateau reached).
        let mut threshold = 0.15f32;
        if smoothed.len() > sw * 3 {
            let derivs: Vec<f32> = smoothed.windows(2).map(|w| w[1] - w[0]).collect();
            // Skip the initial steep drop, find where derivative stabilizes near 0.
            let skip = sw * 2;
            for i in skip..derivs.len() {
                if derivs[i].abs() < 0.003 && smoothed[i] < 0.25 {
                    threshold = smoothed[i];
                    break;
                }
            }
        }
        threshold = threshold.max(0.05).min(0.25);

        log::info!(
            "calibrate: {} queries, {} gaps, gap_p50={}, gap_p90={}, gap_p95={}, tail_p50={}",
            warmup_queries.len(),
            useful_gaps.len(),
            gap_p50,
            gap_p90,
            gap_p95,
            tail_p50,
        );

        // Log gap histogram.
        let buckets: &[(usize, usize, &str)] = &[
            (0, 1, "0-1"),
            (2, 2, "2"),
            (3, 4, "3-4"),
            (5, 7, "5-7"),
            (8, 15, "8-15"),
            (16, 31, "16-31"),
            (32, usize::MAX, "32+"),
        ];
        let total_g = useful_gaps.len().max(1) as f64;
        for &(lo, hi, label) in buckets {
            let cnt = useful_gaps.iter().filter(|&&g| g >= lo && g <= hi).count();
            if cnt > 0 {
                log::info!(
                    "  gap {:<8}: {} ({:.1}%)",
                    label,
                    cnt,
                    cnt as f64 / total_g * 100.0
                );
            }
        }

        // Use gap_p95: only 5% of inter-useful gaps exceed this value.
        // If we wait this long without a useful admission, very likely
        // no more useful ones are coming.
        let early_exit_limit = gap_p95.max(3);

        log::info!(
            "calibrate result: threshold={:.2}, early_exit_limit={} (gap_p95)",
            threshold,
            early_exit_limit,
        );

        Ok(CalibratedParams {
            threshold,
            early_exit_limit,
        })
    }
}

/// Raw diagnostic data from calibration for visualization.
#[derive(Debug, Clone)]
pub struct CalibrationDiagnostics {
    /// Per-step admission rate (smoothed).
    pub admission_rates: Vec<f32>,
    /// Gaps between consecutive top-k admissions (ee is P95 of this).
    pub useful_gaps: Vec<usize>,
    /// Tail gap distribution: steps from last useful admission to search end.
    pub tail_gaps: Vec<usize>,
    /// Per-step cumulative fraction of final top-k results found (averaged over queries).
    pub topk_coverage_by_step: Vec<f32>,
    /// Calibrated parameters.
    pub params: CalibratedParams,
}

impl<const N: usize> Orion<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Calibrate with full diagnostics for visualization.
    pub fn calibrate_with_diagnostics(
        &self,
        warmup_queries: &[[f32; N]],
        search_list_size: usize,
        window_size: usize,
    ) -> ANNResult<CalibrationDiagnostics> {
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let k = 10usize;

        let mut scratch = InMemSearchScratch::new(search_list_size);

        let max_steps = search_list_size * 4;
        let mut pos_admit_count = vec![0u64; max_steps];
        let mut pos_total_count = vec![0u64; max_steps];
        let mut tail_gaps: Vec<usize> = Vec::new();
        let mut useful_gaps: Vec<usize> = Vec::new();

        for query in warmup_queries {
            let aligned = AlignedQuery(*query);
            let query_vertex = Vertex::new(&aligned.0, 0);
            scratch.prepare_for_query(search_list_size);

            scratch.seen.insert(entry);
            let entry_dist = {
                let v = dataset.get_vertex(entry)?;
                v.compare(&query_vertex, Metric::L2)
            };
            scratch.pq.insert(DNeighbor::new(entry, entry_dist));

            let mut admit_step: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::new();
            admit_step.insert(entry, 0);
            let mut step = 0usize;

            while scratch.pq.has_notvisited_node() {
                let neighbor = scratch.pq.closest_notvisited();
                let id = neighbor.id as usize;

                scratch.id_scratch.clear();
                for &nn in graph.neighbors(id) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }

                let n_unseen = scratch.id_scratch.len();
                let pq_worst = if scratch.pq.size() >= search_list_size {
                    scratch.pq[scratch.pq.size() - 1].distance
                } else {
                    f32::MAX
                };
                let mut admitted = false;

                for m in 0..n_unseen {
                    let nn = scratch.id_scratch[m];
                    let v = dataset.get_vertex(nn)?;
                    let dist = query_vertex.compare(&v, Metric::L2);
                    if dist < pq_worst || scratch.pq.size() < search_list_size {
                        admitted = true;
                        admit_step.insert(nn, step);
                    }
                    scratch.pq.insert(DNeighbor::new(nn, dist));
                }

                if step < max_steps {
                    pos_total_count[step] += 1;
                    if admitted {
                        pos_admit_count[step] += 1;
                    }
                }
                step += 1;
            }

            let total_steps = step;
            let top_k: Vec<u32> = (0..scratch.pq.size().min(k))
                .map(|i| scratch.pq[i].id)
                .collect();
            let mut useful_steps: Vec<usize> = top_k
                .iter()
                .filter_map(|id| admit_step.get(id))
                .copied()
                .collect();
            useful_steps.sort_unstable();
            useful_steps.dedup();

            for w in useful_steps.windows(2) {
                useful_gaps.push(w[1] - w[0]);
            }
            if let Some(&last) = useful_steps.last() {
                tail_gaps.push(total_steps.saturating_sub(last));
            }
        }

        // Compute admission rate curve
        let valid_steps = pos_total_count
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(max_steps);
        let rates: Vec<f32> = (0..valid_steps)
            .map(|i| {
                if pos_total_count[i] > 0 {
                    pos_admit_count[i] as f32 / pos_total_count[i] as f32
                } else {
                    0.0
                }
            })
            .collect();

        let sw = window_size.max(3);
        let smoothed: Vec<f32> = (0..rates.len())
            .map(|i| {
                let s = i.saturating_sub(sw / 2);
                let e = (i + sw / 2 + 1).min(rates.len());
                rates[s..e].iter().sum::<f32>() / (e - s) as f32
            })
            .collect();

        // Derive params (same logic as calibrate)
        let mut threshold = 0.15f32;
        if smoothed.len() > sw * 3 {
            let derivs: Vec<f32> = smoothed.windows(2).map(|w| w[1] - w[0]).collect();
            let skip = sw * 2;
            for i in skip..derivs.len() {
                if derivs[i].abs() < 0.003 && smoothed[i] < 0.25 {
                    threshold = smoothed[i];
                    break;
                }
            }
        }
        threshold = threshold.max(0.05).min(0.25);

        useful_gaps.sort_unstable();
        tail_gaps.sort_unstable();
        let early_exit_limit = percentile(&useful_gaps, 95).max(3);

        // Compute per-step top-k coverage: for each step s, what fraction of
        // final top-k results have been admitted at step <= s (averaged over queries).
        let max_steps_seen = pos_total_count
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(max_steps);
        let mut topk_coverage_by_step = vec![0.0f32; max_steps_seen];
        // Re-run to collect per-step coverage (reuse data from above)
        // We already collected per-query useful_steps; reconstruct coverage from admit_step data.
        // Actually, we need to re-collect. Use a second pass.
        {
            let mut per_step_coverage_sum = vec![0.0f64; max_steps_seen];
            let mut n_queries = 0usize;

            for query in warmup_queries {
                let aligned = AlignedQuery(*query);
                let query_vertex = Vertex::new(&aligned.0, 0);
                scratch.prepare_for_query(search_list_size);

                scratch.seen.insert(entry);
                let entry_dist = {
                    let v = dataset.get_vertex(entry)?;
                    v.compare(&query_vertex, Metric::L2)
                };
                scratch.pq.insert(DNeighbor::new(entry, entry_dist));

                let mut admit_step_map: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::new();
                admit_step_map.insert(entry, 0);
                let mut step = 0usize;

                while scratch.pq.has_notvisited_node() {
                    let neighbor = scratch.pq.closest_notvisited();
                    let id = neighbor.id as usize;
                    scratch.id_scratch.clear();
                    for &nn in graph.neighbors(id) {
                        if scratch.seen.insert(nn) {
                            scratch.id_scratch.push(nn);
                        }
                    }
                    let pq_worst = if scratch.pq.size() >= search_list_size {
                        scratch.pq[scratch.pq.size() - 1].distance
                    } else {
                        f32::MAX
                    };
                    for m in 0..scratch.id_scratch.len() {
                        let nn = scratch.id_scratch[m];
                        let v = dataset.get_vertex(nn)?;
                        let dist = query_vertex.compare(&v, Metric::L2);
                        if dist < pq_worst || scratch.pq.size() < search_list_size {
                            admit_step_map.insert(nn, step);
                        }
                        scratch.pq.insert(DNeighbor::new(nn, dist));
                    }
                    step += 1;
                }

                // Get final top-k
                let top_k: Vec<u32> = (0..scratch.pq.size().min(k))
                    .map(|i| scratch.pq[i].id)
                    .collect();
                let k_actual = top_k.len() as f64;

                // Build per-step cumulative coverage
                for s in 0..max_steps_seen.min(step) {
                    let found = top_k
                        .iter()
                        .filter(|id| admit_step_map.get(id).map_or(false, |&as_| as_ <= s))
                        .count();
                    per_step_coverage_sum[s] += found as f64 / k_actual;
                }
                // Fill remaining steps with 1.0 (all found)
                for s in step..max_steps_seen {
                    per_step_coverage_sum[s] += 1.0;
                }
                n_queries += 1;
            }

            if n_queries > 0 {
                for s in 0..max_steps_seen {
                    topk_coverage_by_step[s] = (per_step_coverage_sum[s] / n_queries as f64) as f32;
                }
            }
        }

        Ok(CalibrationDiagnostics {
            admission_rates: smoothed,
            useful_gaps,
            tail_gaps,
            topk_coverage_by_step,
            params: CalibratedParams {
                threshold,
                early_exit_limit,
            },
        })
    }
}

fn percentile(sorted: &[usize], p: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
    sorted[idx]
}
