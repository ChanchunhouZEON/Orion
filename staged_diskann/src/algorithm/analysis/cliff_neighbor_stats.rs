/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! **Cliff Neighbor Analysis** for Vamana graphs.
//!
//! A *cliff neighbor* is a position in a node's distance-sorted neighbor list
//! where the distance jumps disproportionately — e.g., one neighbor is a true
//! Top-10 nearest neighbor while the very next is only Top-50.
//!
//! For each node we:
//! 1. Sort its graph neighbors by distance from the origin node.
//! 2. Compute consecutive distance ratios: `d[i+1] / d[i]`.
//! 3. Identify the *cliff position* (argmax of the ratio vector).
//!
//! Optionally, with brute-force KNN ranks, we also measure how many neighbors
//! before/after the cliff belong to the true Top-K.

use crate::model::PhasedGraph;

// ── Per-node stats ──────────────────────────────────────────────────────────

/// Distance-sorted neighbor entry for a single node.
#[derive(Debug, Clone)]
pub struct SortedNeighbor {
    /// Neighbor node ID.
    pub id: u32,
    /// L2 distance from the origin node.
    pub distance: f32,
    /// Brute-force KNN rank (0-based). `None` when brute-force is not computed.
    pub bf_rank: Option<u32>,
}

/// Per-node cliff analysis result.
#[derive(Debug, Clone)]
pub struct NodeCliffStats {
    pub node: u32,
    /// Degree (number of neighbors).
    pub degree: usize,
    /// Neighbors sorted by distance ascending.
    pub sorted_neighbors: Vec<SortedNeighbor>,
    /// Consecutive distance ratios: `d[i+1] / d[i]` for i in 0..degree-1.
    pub gap_ratios: Vec<f32>,
    /// Index into `gap_ratios` with the maximum value (the primary cliff).
    /// This means the cliff sits *between* sorted_neighbors[cliff_pos] and
    /// sorted_neighbors[cliff_pos + 1].
    pub cliff_pos: usize,
    /// The maximum gap ratio value.
    pub cliff_ratio: f32,
    /// Distance of the nearest neighbor.
    pub min_distance: f32,
    /// Distance of the farthest neighbor.
    pub max_distance: f32,
}

// ── Core computation ────────────────────────────────────────────────────────

/// Compute cliff statistics for every node in the graph.
///
/// `flat_data` is the row-major dataset with stride `dim`.
/// Distance metric: squared L2 (consistent with Vamana build).
pub fn compute_cliff_stats(
    graph: &PhasedGraph,
    flat_data: &[f32],
    dim: usize,
) -> Vec<NodeCliffStats> {
    let n = graph.num_nodes();
    let mut stats = Vec::with_capacity(n);

    for node in 0..n {
        let neighbors = graph.neighbors(node);
        if neighbors.is_empty() {
            stats.push(NodeCliffStats {
                node: node as u32,
                degree: 0,
                sorted_neighbors: vec![],
                gap_ratios: vec![],
                cliff_pos: 0,
                cliff_ratio: 1.0,
                min_distance: 0.0,
                max_distance: 0.0,
            });
            continue;
        }

        // Compute distances and sort.
        let pa = &flat_data[node * dim..(node + 1) * dim];
        let mut sorted: Vec<SortedNeighbor> = neighbors
            .iter()
            .map(|&nbr| {
                let pb = &flat_data[nbr as usize * dim..(nbr as usize + 1) * dim];
                let dist: f32 = pa.iter().zip(pb).map(|(a, b)| (a - b) * (a - b)).sum();
                SortedNeighbor {
                    id: nbr,
                    distance: dist,
                    bf_rank: None,
                }
            })
            .collect();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Consecutive gap ratios.
        let mut gap_ratios = Vec::with_capacity(sorted.len().saturating_sub(1));
        let mut cliff_pos = 0usize;
        let mut cliff_ratio = 0.0f32;

        for i in 0..sorted.len().saturating_sub(1) {
            let ratio = if sorted[i].distance > 0.0 {
                sorted[i + 1].distance / sorted[i].distance
            } else {
                1.0
            };
            if ratio > cliff_ratio {
                cliff_ratio = ratio;
                cliff_pos = i;
            }
            gap_ratios.push(ratio);
        }

        let min_d = sorted.first().map(|s| s.distance).unwrap_or(0.0);
        let max_d = sorted.last().map(|s| s.distance).unwrap_or(0.0);

        stats.push(NodeCliffStats {
            node: node as u32,
            degree: sorted.len(),
            sorted_neighbors: sorted,
            gap_ratios,
            cliff_pos,
            cliff_ratio,
            min_distance: min_d,
            max_distance: max_d,
        });
    }

    stats
}

// ── Brute-force rank annotation ─────────────────────────────────────────────

/// For sampled nodes, annotate each neighbor with its brute-force KNN rank.
///
/// `bf_max_rank` caps the brute-force search depth (e.g. 200). Neighbors
/// whose rank exceeds `bf_max_rank` get `bf_rank = None`.
pub fn annotate_bf_ranks(
    stats: &mut [NodeCliffStats],
    flat_data: &[f32],
    dim: usize,
    num_points: usize,
    bf_max_rank: usize,
) {
    for s in stats.iter_mut() {
        let node = s.node as usize;
        let pa = &flat_data[node * dim..(node + 1) * dim];

        // Brute-force: compute distances to all points, partial sort to bf_max_rank.
        let mut dists: Vec<(u32, f32)> = (0..num_points)
            .filter(|&i| i != node)
            .map(|i| {
                let pb = &flat_data[i * dim..(i + 1) * dim];
                let d: f32 = pa.iter().zip(pb).map(|(a, b)| (a - b) * (a - b)).sum();
                (i as u32, d)
            })
            .collect();

        // Partial sort: we only need the top bf_max_rank ranks.
        // Full sort is simpler and fine for profiling.
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Build rank lookup for the top bf_max_rank.
        let rank_cap = bf_max_rank.min(dists.len());
        let mut rank_map = std::collections::HashMap::with_capacity(rank_cap);
        for (rank, &(id, _)) in dists[..rank_cap].iter().enumerate() {
            rank_map.insert(id, rank as u32);
        }

        for nbr in &mut s.sorted_neighbors {
            nbr.bf_rank = rank_map.get(&nbr.id).copied();
        }
    }
}

// ── Aggregate summary ───────────────────────────────────────────────────────

/// Summary statistics across all nodes.
#[derive(Debug)]
pub struct CliffSummary {
    pub num_nodes: usize,
    pub avg_degree: f64,
    /// Cliff position histogram: index = cliff_pos, value = count of nodes.
    pub cliff_pos_histogram: Vec<usize>,
    /// Percentiles (p10, p25, p50, p75, p90, p95, p99) of cliff_ratio.
    pub cliff_ratio_percentiles: [f64; 7],
    /// Mean cliff ratio.
    pub mean_cliff_ratio: f64,
    /// Mean cliff position.
    pub mean_cliff_pos: f64,
    /// Fraction of nodes whose cliff is in the first 25% of the neighbor list.
    pub cliff_in_first_quarter: f64,
    /// Mean distance span ratio: max_distance / min_distance.
    pub mean_distance_span: f64,
}

/// Compute aggregate summary from per-node cliff stats.
pub fn summarize_cliff_stats(stats: &[NodeCliffStats]) -> CliffSummary {
    let n = stats.len();
    if n == 0 {
        return CliffSummary {
            num_nodes: 0,
            avg_degree: 0.0,
            cliff_pos_histogram: vec![],
            cliff_ratio_percentiles: [0.0; 7],
            mean_cliff_ratio: 0.0,
            mean_cliff_pos: 0.0,
            cliff_in_first_quarter: 0.0,
            mean_distance_span: 0.0,
        };
    }

    let total_degree: usize = stats.iter().map(|s| s.degree).sum();
    let avg_degree = total_degree as f64 / n as f64;

    // Cliff position histogram.
    let max_pos = stats.iter().map(|s| s.cliff_pos).max().unwrap_or(0);
    let mut pos_hist = vec![0usize; max_pos + 1];
    for s in stats {
        if s.degree > 1 {
            pos_hist[s.cliff_pos] += 1;
        }
    }

    // Cliff ratio percentiles.
    let mut ratios: Vec<f64> = stats
        .iter()
        .filter(|s| s.degree > 1)
        .map(|s| s.cliff_ratio as f64)
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |p: f64| -> f64 {
        if ratios.is_empty() {
            return 0.0;
        }
        let idx = ((p / 100.0) * (ratios.len() - 1) as f64).round() as usize;
        ratios[idx.min(ratios.len() - 1)]
    };
    let cliff_ratio_percentiles = [
        percentile(10.0),
        percentile(25.0),
        percentile(50.0),
        percentile(75.0),
        percentile(90.0),
        percentile(95.0),
        percentile(99.0),
    ];

    let mean_ratio: f64 = ratios.iter().sum::<f64>() / ratios.len().max(1) as f64;

    // Mean cliff position.
    let nodes_with_cliff = stats.iter().filter(|s| s.degree > 1).count();
    let mean_pos: f64 = stats
        .iter()
        .filter(|s| s.degree > 1)
        .map(|s| s.cliff_pos as f64)
        .sum::<f64>()
        / nodes_with_cliff.max(1) as f64;

    // Cliff in first quarter.
    let first_quarter = stats
        .iter()
        .filter(|s| s.degree > 1)
        .filter(|s| (s.cliff_pos as f64) < (s.degree as f64 * 0.25))
        .count();
    let cliff_in_first_quarter = first_quarter as f64 / nodes_with_cliff.max(1) as f64;

    // Mean distance span.
    let mean_span: f64 = stats
        .iter()
        .filter(|s| s.min_distance > 0.0)
        .map(|s| s.max_distance as f64 / s.min_distance as f64)
        .sum::<f64>()
        / stats.iter().filter(|s| s.min_distance > 0.0).count().max(1) as f64;

    CliffSummary {
        num_nodes: n,
        avg_degree,
        cliff_pos_histogram: pos_hist,
        cliff_ratio_percentiles,
        mean_cliff_ratio: mean_ratio,
        mean_cliff_pos: mean_pos,
        cliff_in_first_quarter,
        mean_distance_span: mean_span,
    }
}

impl std::fmt::Display for CliffSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cliff Neighbor Summary ({} nodes)", self.num_nodes)?;
        writeln!(f, "  Avg degree:              {:.1}", self.avg_degree)?;
        writeln!(f, "  Mean cliff position:     {:.2}", self.mean_cliff_pos)?;
        writeln!(
            f,
            "  Cliff in first 25%:      {:.1}%",
            self.cliff_in_first_quarter * 100.0
        )?;
        writeln!(f, "  Mean cliff ratio:        {:.3}", self.mean_cliff_ratio)?;
        writeln!(
            f,
            "  Cliff ratio percentiles: p10={:.2} p25={:.2} p50={:.2} p75={:.2} p90={:.2} p95={:.2} p99={:.2}",
            self.cliff_ratio_percentiles[0],
            self.cliff_ratio_percentiles[1],
            self.cliff_ratio_percentiles[2],
            self.cliff_ratio_percentiles[3],
            self.cliff_ratio_percentiles[4],
            self.cliff_ratio_percentiles[5],
            self.cliff_ratio_percentiles[6],
        )?;
        writeln!(
            f,
            "  Mean distance span:      {:.2}x",
            self.mean_distance_span
        )?;
        writeln!(f, "  Cliff position distribution:")?;
        for (i, &count) in self.cliff_pos_histogram.iter().enumerate() {
            if count > 0 {
                writeln!(
                    f,
                    "    pos {:<3}: {} nodes ({:.1}%)",
                    i,
                    count,
                    count as f64 / self.num_nodes as f64 * 100.0
                )?;
            }
        }
        Ok(())
    }
}

// ── Brute-force rank summary ────────────────────────────────────────────────

/// Summary of brute-force rank distribution around cliff positions.
#[derive(Debug)]
pub struct CliffRankSummary {
    pub num_sampled: usize,
    /// Average BF rank of neighbors *before* the cliff (positions 0..cliff_pos inclusive).
    pub avg_rank_before_cliff: f64,
    /// Average BF rank of neighbors *after* the cliff (positions cliff_pos+1..).
    pub avg_rank_after_cliff: f64,
    /// For each KNN threshold K, what fraction of nodes have their cliff
    /// separating Top-K from the rest?
    /// Entries: (K, fraction).
    pub cliff_separates_topk: Vec<(usize, f64)>,
    /// Distribution of BF ranks at cliff_pos (the last "close" neighbor).
    pub rank_at_cliff_percentiles: [f64; 5], // p10, p25, p50, p75, p90
    /// Distribution of BF ranks at cliff_pos+1 (the first "far" neighbor).
    pub rank_after_cliff_percentiles: [f64; 5],
}

/// Compute rank-based summary for nodes that have bf_rank annotations.
pub fn summarize_cliff_ranks(
    stats: &[NodeCliffStats],
    topk_thresholds: &[usize],
) -> CliffRankSummary {
    let annotated: Vec<&NodeCliffStats> = stats
        .iter()
        .filter(|s| s.degree > 1 && s.sorted_neighbors.iter().any(|n| n.bf_rank.is_some()))
        .collect();

    let num = annotated.len();
    if num == 0 {
        return CliffRankSummary {
            num_sampled: 0,
            avg_rank_before_cliff: 0.0,
            avg_rank_after_cliff: 0.0,
            cliff_separates_topk: topk_thresholds.iter().map(|&k| (k, 0.0)).collect(),
            rank_at_cliff_percentiles: [0.0; 5],
            rank_after_cliff_percentiles: [0.0; 5],
        };
    }

    let mut sum_before = 0.0f64;
    let mut cnt_before = 0usize;
    let mut sum_after = 0.0f64;
    let mut cnt_after = 0usize;
    let mut ranks_at_cliff = Vec::with_capacity(num);
    let mut ranks_after_cliff = Vec::with_capacity(num);

    // For topk separation: count how many nodes have ALL pre-cliff neighbors
    // in Top-K AND the post-cliff neighbor outside Top-K.
    let mut topk_sep_counts = vec![0usize; topk_thresholds.len()];

    for s in &annotated {
        let cp = s.cliff_pos;

        // Ranks before cliff (inclusive).
        for nbr in &s.sorted_neighbors[..=cp] {
            if let Some(r) = nbr.bf_rank {
                sum_before += r as f64;
                cnt_before += 1;
            }
        }
        // Ranks after cliff.
        for nbr in &s.sorted_neighbors[cp + 1..] {
            if let Some(r) = nbr.bf_rank {
                sum_after += r as f64;
                cnt_after += 1;
            }
        }

        // Rank at cliff boundary.
        if let Some(r) = s.sorted_neighbors[cp].bf_rank {
            ranks_at_cliff.push(r as f64);
        }
        if cp + 1 < s.sorted_neighbors.len() {
            if let Some(r) = s.sorted_neighbors[cp + 1].bf_rank {
                ranks_after_cliff.push(r as f64);
            }
        }

        // Top-K separation check.
        for (ti, &k) in topk_thresholds.iter().enumerate() {
            let all_before_in_topk = s.sorted_neighbors[..=cp]
                .iter()
                .all(|n| n.bf_rank.map_or(false, |r| (r as usize) < k));
            let first_after_outside = if cp + 1 < s.sorted_neighbors.len() {
                s.sorted_neighbors[cp + 1]
                    .bf_rank
                    .map_or(true, |r| (r as usize) >= k)
            } else {
                true
            };
            if all_before_in_topk && first_after_outside {
                topk_sep_counts[ti] += 1;
            }
        }
    }

    let pct = |v: &mut Vec<f64>, p: f64| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
        v[idx.min(v.len() - 1)]
    };

    let mut rc = ranks_at_cliff;
    let mut ra = ranks_after_cliff;

    CliffRankSummary {
        num_sampled: num,
        avg_rank_before_cliff: if cnt_before > 0 {
            sum_before / cnt_before as f64
        } else {
            0.0
        },
        avg_rank_after_cliff: if cnt_after > 0 {
            sum_after / cnt_after as f64
        } else {
            0.0
        },
        cliff_separates_topk: topk_thresholds
            .iter()
            .enumerate()
            .map(|(ti, &k)| (k, topk_sep_counts[ti] as f64 / num as f64))
            .collect(),
        rank_at_cliff_percentiles: [
            pct(&mut rc, 10.0),
            pct(&mut rc, 25.0),
            pct(&mut rc, 50.0),
            pct(&mut rc, 75.0),
            pct(&mut rc, 90.0),
        ],
        rank_after_cliff_percentiles: [
            pct(&mut ra, 10.0),
            pct(&mut ra, 25.0),
            pct(&mut ra, 50.0),
            pct(&mut ra, 75.0),
            pct(&mut ra, 90.0),
        ],
    }
}

impl std::fmt::Display for CliffRankSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Cliff Rank Analysis ({} sampled nodes)",
            self.num_sampled
        )?;
        writeln!(
            f,
            "  Avg BF rank before cliff: {:.1}",
            self.avg_rank_before_cliff
        )?;
        writeln!(
            f,
            "  Avg BF rank after cliff:  {:.1}",
            self.avg_rank_after_cliff
        )?;
        writeln!(
            f,
            "  Rank at cliff (p10/p25/p50/p75/p90): {:.0} / {:.0} / {:.0} / {:.0} / {:.0}",
            self.rank_at_cliff_percentiles[0],
            self.rank_at_cliff_percentiles[1],
            self.rank_at_cliff_percentiles[2],
            self.rank_at_cliff_percentiles[3],
            self.rank_at_cliff_percentiles[4],
        )?;
        writeln!(
            f,
            "  Rank after cliff (p10/p25/p50/p75/p90): {:.0} / {:.0} / {:.0} / {:.0} / {:.0}",
            self.rank_after_cliff_percentiles[0],
            self.rank_after_cliff_percentiles[1],
            self.rank_after_cliff_percentiles[2],
            self.rank_after_cliff_percentiles[3],
            self.rank_after_cliff_percentiles[4],
        )?;
        writeln!(f, "  Cliff separates Top-K:")?;
        for &(k, frac) in &self.cliff_separates_topk {
            writeln!(f, "    Top-{:<4}: {:.1}% of nodes", k, frac * 100.0)?;
        }
        Ok(())
    }
}

// ── Per-node gap profile (for detailed output) ─────────────────────────────

/// Print a detailed gap profile for a single node (for debugging / visualization).
pub fn print_node_cliff_detail(s: &NodeCliffStats) {
    println!("Node {} (degree={})", s.node, s.degree);
    println!(
        "  Cliff at position {} (ratio={:.3})",
        s.cliff_pos, s.cliff_ratio
    );
    println!(
        "  Distance range: {:.4} .. {:.4} (span {:.2}x)",
        s.min_distance,
        s.max_distance,
        if s.min_distance > 0.0 {
            s.max_distance / s.min_distance
        } else {
            0.0
        }
    );
    for (i, nbr) in s.sorted_neighbors.iter().enumerate() {
        let cliff_marker = if i == s.cliff_pos { " <-- CLIFF" } else { "" };
        let rank_str = match nbr.bf_rank {
            Some(r) => format!("rank={}", r),
            None => "rank=?".to_string(),
        };
        let gap_str = if i < s.gap_ratios.len() {
            format!("gap={:.3}", s.gap_ratios[i])
        } else {
            String::new()
        };
        println!(
            "  [{:>2}] id={:<7} dist={:.4}  {}  {}{}",
            i, nbr.id, nbr.distance, rank_str, gap_str, cliff_marker
        );
    }
}

// ── Rank-precision cliff detection ──────────────────────────────────────────

/// Per-node rank-precision profile.
///
/// For position k in the distance-sorted neighbor list, `precision[k]` =
/// `|graph_neighbors[0..=k] ∩ true_top_(k+1)| / (k+1)`.
///
/// The cliff point is where precision drops most sharply.
#[derive(Debug, Clone)]
pub struct NodeRankPrecisionProfile {
    pub node: u32,
    pub degree: usize,
    /// precision[k] for k = 0..degree-1.
    pub precision: Vec<f64>,
    /// Position of maximum precision drop (cliff).
    pub cliff_pos: usize,
    /// The drop magnitude at cliff_pos: precision[cliff_pos] - precision[cliff_pos+1].
    pub cliff_drop: f64,
}

/// Compute rank-precision profiles for sampled nodes.
///
/// `sorted_nbrs[node]` = distance-sorted neighbor IDs.
/// `bf_knn[si]` = brute-force top-K IDs for sample_nodes[si].
pub fn compute_rank_precision_profiles(
    sorted_nbrs: &[Vec<u32>],
    sample_nodes: &[usize],
    bf_knn: &[Vec<u32>],
) -> Vec<NodeRankPrecisionProfile> {
    sample_nodes
        .iter()
        .zip(bf_knn.iter())
        .map(|(&node, bf_top)| {
            let nbrs = &sorted_nbrs[node];
            let degree = nbrs.len();
            let mut precision = Vec::with_capacity(degree);

            let mut hits = 0usize;
            for (k, &nbr_id) in nbrs.iter().enumerate() {
                // Is this neighbor in the true top-(k+1)?
                if bf_top.len() > k && bf_top[..=k].contains(&nbr_id) {
                    hits += 1;
                }
                precision.push(hits as f64 / (k + 1) as f64);
            }

            // Find cliff: position of maximum drop.
            let mut cliff_pos = 0;
            let mut cliff_drop = 0.0f64;
            for i in 0..precision.len().saturating_sub(1) {
                let drop = precision[i] - precision[i + 1];
                if drop > cliff_drop {
                    cliff_drop = drop;
                    cliff_pos = i;
                }
            }

            NodeRankPrecisionProfile {
                node: node as u32,
                degree,
                precision,
                cliff_pos,
                cliff_drop,
            }
        })
        .collect()
}

/// Aggregate rank-precision summary.
#[derive(Debug)]
pub struct RankPrecisionSummary {
    pub num_sampled: usize,
    /// Average precision@k across all sampled nodes, for each position k.
    pub avg_precision_by_pos: Vec<f64>,
    /// Average cliff position.
    pub mean_cliff_pos: f64,
    /// Cliff drop percentiles (p10, p25, p50, p75, p90).
    pub cliff_drop_percentiles: [f64; 5],
    /// Cliff position histogram.
    pub cliff_pos_histogram: Vec<usize>,
}

pub fn summarize_rank_precision(profiles: &[NodeRankPrecisionProfile]) -> RankPrecisionSummary {
    let n = profiles.len();
    if n == 0 {
        return RankPrecisionSummary {
            num_sampled: 0,
            avg_precision_by_pos: vec![],
            mean_cliff_pos: 0.0,
            cliff_drop_percentiles: [0.0; 5],
            cliff_pos_histogram: vec![],
        };
    }

    let max_deg = profiles.iter().map(|p| p.degree).max().unwrap_or(0);
    let mut pos_sum = vec![0.0f64; max_deg];
    let mut pos_cnt = vec![0usize; max_deg];
    for p in profiles {
        for (k, &prec) in p.precision.iter().enumerate() {
            pos_sum[k] += prec;
            pos_cnt[k] += 1;
        }
    }
    let avg_precision_by_pos: Vec<f64> = pos_sum
        .iter()
        .zip(pos_cnt.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    let mean_cliff = profiles.iter().map(|p| p.cliff_pos as f64).sum::<f64>() / n as f64;

    let mut drops: Vec<f64> = profiles.iter().map(|p| p.cliff_drop).collect();
    drops.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (drops.len() - 1) as f64).round() as usize;
        drops[idx.min(drops.len() - 1)]
    };

    let max_pos = profiles.iter().map(|p| p.cliff_pos).max().unwrap_or(0);
    let mut hist = vec![0usize; max_pos + 1];
    for p in profiles {
        hist[p.cliff_pos] += 1;
    }

    RankPrecisionSummary {
        num_sampled: n,
        avg_precision_by_pos,
        mean_cliff_pos: mean_cliff,
        cliff_drop_percentiles: [pct(10.0), pct(25.0), pct(50.0), pct(75.0), pct(90.0)],
        cliff_pos_histogram: hist,
    }
}

impl std::fmt::Display for RankPrecisionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Rank-Precision Cliff Summary ({} nodes)",
            self.num_sampled
        )?;
        writeln!(f, "  Mean cliff position:   {:.2}", self.mean_cliff_pos)?;
        writeln!(
            f,
            "  Cliff drop (p10/p25/p50/p75/p90): {:.3} / {:.3} / {:.3} / {:.3} / {:.3}",
            self.cliff_drop_percentiles[0],
            self.cliff_drop_percentiles[1],
            self.cliff_drop_percentiles[2],
            self.cliff_drop_percentiles[3],
            self.cliff_drop_percentiles[4],
        )?;
        writeln!(f, "\n  Avg precision@k by position:")?;
        writeln!(f, "  {:<6} {:<12}", "Pos", "Precision@k")?;
        writeln!(f, "  {}", "─".repeat(20))?;
        for (k, &p) in self.avg_precision_by_pos.iter().enumerate() {
            if k < 40 || k == self.avg_precision_by_pos.len() - 1 {
                writeln!(f, "  {:<6} {:<12.4}", k, p)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph_and_data() -> (PhasedGraph, Vec<f32>, usize) {
        let dim = 2;
        let flat = vec![
            0.0, 0.0, // node 0
            1.0, 0.0, // node 1
            3.0, 0.0, // node 2
            10.0, 0.0, // node 3
        ];
        let partitions = vec![
            (vec![1, 2, 3], vec![], vec![]),
            (vec![0, 2], vec![], vec![]),
            (vec![0, 1, 3], vec![], vec![]),
            (vec![0, 2], vec![], vec![]),
        ];
        let pg = PhasedGraph::build_from_partitions(&partitions, 4, 4);
        (pg, flat, dim)
    }

    #[test]
    fn test_cliff_detection() {
        let (graph, flat, dim) = make_test_graph_and_data();
        let stats = compute_cliff_stats(&graph, &flat, dim);

        // Node 0: neighbors sorted by dist = [1(d=1), 2(d=9), 3(d=100)]
        // gap_ratios = [9/1=9.0, 100/9=11.11]
        // cliff_pos = 1 (11.11 > 9.0)
        let s0 = &stats[0];
        assert_eq!(s0.degree, 3);
        assert_eq!(s0.sorted_neighbors[0].id, 1);
        assert_eq!(s0.sorted_neighbors[1].id, 2);
        assert_eq!(s0.sorted_neighbors[2].id, 3);
        assert_eq!(s0.cliff_pos, 1);
        assert!((s0.cliff_ratio - 100.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let (graph, flat, dim) = make_test_graph_and_data();
        let stats = compute_cliff_stats(&graph, &flat, dim);
        let summary = summarize_cliff_stats(&stats);
        assert_eq!(summary.num_nodes, 4);
        assert!(summary.mean_cliff_ratio > 1.0);
        println!("{summary}");
    }
}
