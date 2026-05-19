/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::StagedDiskANN;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::InMemScratchPool;
use diskann::common::ANNResult;
use diskann::model::Vertex;
use rayon::prelude::*;
use std::sync::atomic::AtomicU64;
use vector::{FullPrecisionDistance, Metric};

// ── Per-phase instrumentation counters (shared across search paths) ──
// All atomic with `Relaxed`; counter accuracy across threads is
// approximate but ordering doesn't matter for averaging. Mirrors PA's
// `average visited` / `average cmps` fields for direct A/B comparison.
//
// Used by `staged_sweep` to print per-L stats. Reset between trials
// via the same Ordering. Live here (not in `in_mem_search_mips_q`)
// so the L2, MIPS, and MIPS-Q paths can all increment the same set
// without cross-module dependencies.
pub static VISIT_COUNT: AtomicU64 = AtomicU64::new(0);
pub static QUERY_COUNT: AtomicU64 = AtomicU64::new(0);
pub static PRE_CONV_HOPS: AtomicU64 = AtomicU64::new(0);
pub static POST_CONV_HOPS: AtomicU64 = AtomicU64::new(0);
pub static PRE_CONV_ADMITS: AtomicU64 = AtomicU64::new(0);
pub static POST_CONV_ADMITS: AtomicU64 = AtomicU64::new(0);
/// Total distance computations per query — counts every Stage-1
/// quantized distance (one per unseen neighbour per hop) + every
/// Stage-2 f32 truth/rerank distance. Mirrors PA's `average cmps`.
pub static NDC_I8: AtomicU64 = AtomicU64::new(0);
pub static NDC_F32: AtomicU64 = AtomicU64::new(0);
/// Total nanoseconds spent in per-query setup (normalise + quantize
/// query + scratch acquire/prepare/reconfigure + entry insert),
/// summed across all threads. Divide by `QUERY_COUNT` for per-query
/// setup cost; useful for diagnosing whether the low-recall QPS
/// plateau is dominated by setup overhead rather than search work.
pub static SETUP_NS: AtomicU64 = AtomicU64::new(0);

/// **PA-style adaptive prefilter toggle**. When `STAGED_USE_FILTER=1`,
/// the L2 search path replaces the multiplicative `pq_worst · slope² ·
/// Q_SLACK` cutoff with PA's running-mean filter: every hop where
/// the frontier is full, take the mean of the frontier-tail's u8
/// distances, accumulate into a running average, and use that as the
/// per-hop u8-prefilter threshold (PA `beamSearch.h:130–145`).
///
/// Off by default — the multiplicative cutoff is simpler and works
/// well on glove100. Turn on for SIFT1M where the multiplicative
/// cutoff over-admits to the f32 rerank stage (causing the 1.5×
/// NDC-vs-PA gap measured at recall 0.91–0.99).
pub(super) fn use_filter() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("STAGED_USE_FILTER")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// L-adaptive `par_chunks` BATCH size for per-query parallelism in
/// `search_batch_*`. Picked by an LA × L probe on glove100 i8: low L
/// favors slightly larger batches (amortise rayon dispatch over
/// short queries), high L favors smaller batches (work-stealing tail
/// rebalances across threads when the per-query wall is long).
///
/// Used by the **MIPS-Q** path (i8 / i16 quantized beam, glove100).
#[inline]
pub(super) fn search_batch_size(search_list_size: usize) -> usize {
    if search_list_size < 64 {
        32
    } else if search_list_size < 256 {
        16
    } else if search_list_size < 768 {
        8
    } else {
        4
    }
}

/// L-adaptive `par_chunks` BATCH size for the **L2** path (SIFT-family).
/// Half the MIPS-Q schedule per L band — measured empirically on
/// SIFT1M to be the sweet spot. The L2 path's two-stage pipeline
/// (u8 prefilter + f32 rerank, with cmov-compact between) makes
/// each query slower than MIPS-Q's single-stage flow, so smaller
/// batches give work-stealing more opportunities to rebalance the
/// late-finishing tail across threads.
///
/// Schedule:
///   L < 64        → BATCH = 16
///   64 ≤ L < 256  → BATCH = 8
///   256 ≤ L < 768 → BATCH = 4
///   L ≥ 768       → BATCH = 2
#[inline]
pub(super) fn search_batch_size_l2(search_list_size: usize) -> usize {
    (search_batch_size(search_list_size) / 2).max(1)
}

// ── Search-loop tuning constants ──────────────────────────────────────────
/// Cache-line lookahead for the software-pipelined `DistanceStream`
/// **Stage-1 quantized prefilter**. Stage-1 vert sizes:
/// glove100 i8 = **1 line/vert** (LA=12 ⇒ 12 verts ahead),
/// glove100 i16 = 2 lines/vert (LA=12 ⇒ 6 verts ahead),
/// SIFT u8 = 1 line/vert. Steady-state issues `LA_Q` prfm per outer
/// iter; M2 MSHR ≈ 12. Override via `STAGED_DSTREAM_LA_Q=<N>`
/// (back-compat: `STAGED_DSTREAM_LOOKAHEAD` still honored).
pub(super) fn dstream_la_q() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("STAGED_DSTREAM_LA_Q")
            .ok()
            .or_else(|| std::env::var("STAGED_DSTREAM_LOOKAHEAD").ok())
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 128)
            // 64 chosen via LA × L sweep on glove100 i8 with the
            // sliding-window continuous-stride prefetch design.
            // LA=64 wins low/mid band (L=16=204k, L=32=126k, L=56=86k)
            // and ties LA=48/128 at high L. Yields 1.34× geomean PA
            // across 7 recall-aligned anchor points. Larger LA
            // (96/128) over-prefetches at narrow beam (L=16); smaller
            // LA (8/16) under-fills the MSHR queue.
            .unwrap_or(10)
    })
}

/// Cache-line lookahead for the software-pipelined `DistanceStream`
/// **Stage-2 f32 truth**. f32 vert sizes are larger:
/// glove100 f32 = 4 lines/vert (LA=12 ⇒ **3 verts ahead**),
/// SIFT f32 = 4 lines/vert, GIST f32 = **30 lines/vert** (LA=12 ⇒
/// 0 verts — LA is sub-vert here, prologue still primes lines for
/// the in-progress vert). Stage-2's per-vert compute (~25 ns @ N=100,
/// 4-acc kernel) is ~4× longer than Stage-1's, so the same line
/// budget translates to similar runway in nanoseconds.
/// Override via `STAGED_DSTREAM_LA_TRUTH=<N>`.
pub(super) fn dstream_la_truth() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("STAGED_DSTREAM_LA_TRUTH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 256)
            // 64 — same default as `dstream_la_q`. Sliding-window
            // continuous-stride prefetch in `DistanceStream::run`
            // works the same way for Stage-1 quantized and Stage-2
            // f32 truth: one prfm per outer iter at distance
            // `lookahead_lines` ahead, keeping MSHR at steady state
            // without burst eviction. Larger LA gives a deeper
            // runway to hide DRAM latency on the f32 truth read,
            // which has the same per-line miss cost as Stage-1.
            .unwrap_or(6)
    })
}
/// Slack multiplier on the u8 pre-filter cutoff: quantized distance is
/// noisy, so keep candidates within this factor of the scale-converted
/// `pq_worst`.
#[allow(dead_code)]
pub(super) const Q_SLACK: f32 = 1.3;
/// Flush cadence by phase: `FLUSH_INTERVAL[converged as usize]` hops.
/// Pre-converged flushes every hop to keep pq_worst tight; converged
/// accumulates across 4 hops to amortize sort+merge over sparse admits
/// without letting pq_worst drift enough to defeat `early_exit`.
pub(super) const FLUSH_INTERVAL: [usize; 2] = [1, 4];
/// 3-way merge routing divisors re-fit from `pq_merge_bench` under
/// the pad16 PQ layout:
///   K * 8   < L → per-insert   (K < L / 8,    small-batch regime)
///   K * 1.5 > L → linear merge (K > L · 0.67, near-or-over-capacity)
///   otherwise   → gallop merge (log + bulk-memcpy wins in the middle)
///
/// The bench shows gallop owns the entire `K/L ∈ [0.125, 0.67]` band
/// for L ≥ 64 — its "binary-search insertion + extend_from_slice run
/// of cache-line-sized memcpys" pattern beats both per-insert
/// (O(K·L)) and linear merge (3-way branchy set-union) by 10-30 ns
/// per call. Per-insert wins below the 1/8 line because its constant
/// factor is just one binary-search + one copy_within with no scratch
/// swap. Linear merge wins above the 2/3 line because gallop's
/// `partition_point` degenerates to length-1 runs when admits are
/// near-uniform across the full PQ range.
///
/// Encoded as pure shifts + adds — both comparisons lower to one or
/// two shifts + an add + a compare, no integer multiply:
///   K * 8   = K << 3
///   K * 1.5 = K + (K >> 1)
const INSERT_ROUTE_SHIFT: u32 = 3; // << 3 = × 8

#[inline(always)]
pub(super) const fn insert_route_mul(k: usize) -> usize {
    k << INSERT_ROUTE_SHIFT
}

#[inline(always)]
pub(super) const fn linear_merge_mul(k: usize) -> usize {
    k + (k >> 1)
}

/// 16-byte aligned query buffer for efficient NEON loads.
/// Query is copied once at search entry, then reused for all distance computations.
#[repr(C, align(16))]
pub struct AlignedQuery<const N: usize>(pub [f32; N]);

pub const DEFAULT_SEARCH_LIST_SIZE: usize = 48;
pub const DEFAULT_WINDOW_SIZE: usize = 5;
pub const DEFAULT_EPSILON: f32 = 0.0;
pub const DEFAULT_EARLY_EXIT_LIMIT: usize = 7;

/// Per-operation timing breakdown accumulated across queries.
#[derive(Default)]
pub struct SearchProfileStats {
    pub queries: u64,
    pub iterations: u64,
    /// Time in distance computations (get_vertex + compare).
    pub distance_ns: u64,
    pub distance_count: u64,
    /// Time in PQ operations (closest_notvisited + insert).
    pub pq_ops_ns: u64,
    /// Time reading graph neighbors.
    pub graph_read_ns: u64,
    pub graph_read_count: u64,
    /// Time in seen-set insert + test.
    pub seen_ns: u64,
    /// Time in convergence checker.
    pub convergence_ns: u64,
}

impl SearchProfileStats {
    pub fn total_ns(&self) -> u64 {
        self.distance_ns + self.pq_ops_ns + self.graph_read_ns + self.seen_ns + self.convergence_ns
    }

    pub fn print_report(&self) {
        let total = self.total_ns() as f64;
        let q = self.queries as f64;
        println!(
            "\n─── Search Profile ({} queries, {:.0} iterations/query) ───",
            self.queries,
            self.iterations as f64 / q
        );
        println!(
            "  {:<22} {:>10} {:>8} {:>12}",
            "Operation", "Total (ms)", "% time", "Per-query (µs)"
        );
        println!("  {}", "─".repeat(56));
        let rows = [
            ("Distance compute", self.distance_ns, self.distance_count),
            ("PQ ops", self.pq_ops_ns, 0),
            ("Graph read", self.graph_read_ns, self.graph_read_count),
            ("Seen-set ops", self.seen_ns, 0),
            ("Convergence check", self.convergence_ns, 0),
        ];
        for (name, ns, count) in rows {
            let ms = ns as f64 / 1_000_000.0;
            let pct = ns as f64 / total * 100.0;
            let per_q = ns as f64 / q / 1000.0;
            if count > 0 {
                println!(
                    "  {:<22} {:>10.1} {:>7.1}% {:>12.1}   ({} calls, {:.0} ns/call)",
                    name,
                    ms,
                    pct,
                    per_q,
                    count,
                    ns as f64 / count as f64
                );
            } else {
                println!("  {:<22} {:>10.1} {:>7.1}% {:>12.1}", name, ms, pct, per_q);
            }
        }
        let total_ms = total / 1_000_000.0;
        let per_q_us = total / q / 1000.0;
        println!("  {}", "─".repeat(56));
        println!(
            "  {:<22} {:>10.1} {:>7}  {:>12.1}",
            "TOTAL", total_ms, "100%", per_q_us
        );
        println!(
            "  Estimated QPS (single-thread): {:.0}",
            1_000_000_000.0 / (total / q)
        );
    }
}

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Convenience wrapper around [`Self::search_l2_u8`] with the
    /// crate's default L / window / epsilon / early-exit knobs.
    pub fn search_default(&self, query: &[f32; N], k: usize) -> ANNResult<Vec<u32>> {
        self.search_l2_u8(
            query,
            k,
            DEFAULT_SEARCH_LIST_SIZE,
            DEFAULT_WINDOW_SIZE,
            DEFAULT_EPSILON,
            DEFAULT_WINDOW_SIZE << 1,
        )
    }

    /// Two-phase search with ADSampling distance computations.
    ///
    /// Identical control flow to [`Self::search`] — convergence-driven phase
    /// switch, reranking candidates, τ/ee early exit — but every neighbor
    /// distance uses the scaled-partial ADSampling kernel instead of plain
    /// `compare_with_bound`. **Callers must rotate the dataset and the query
    /// with the same orthogonal matrix** before invoking this function;
    /// otherwise the scaled-partial correctness guarantee breaks.
    pub fn search_adsampling(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        ads_epsilon: f32,
    ) -> ANNResult<Vec<u32>> {
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let aligned = AlignedQuery(*query);
        let query_vertex = Vertex::new(&aligned.0, 0);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        scratch.seen.insert(entry);
        let entry_dist = {
            let v = dataset.get_vertex(entry)?;
            v.compare(&query_vertex, Metric::L2)
        };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));

        let mut prev_admitted: usize = 1;

        while scratch.pq.has_notvisited_node() {
            let neighbor = scratch.pq.closest_notvisited();
            let id = neighbor.id as usize;

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
            let pq_worst = if scratch.pq.size() >= search_list_size {
                scratch.pq[scratch.pq.size() - 1].distance
            } else {
                f32::MAX
            };
            let mut admitted = 0usize;

            if n_unseen > 0 {
                dataset.prefetch_vector(scratch.id_scratch[0]);
            }
            for m in 0..n_unseen {
                if m + 1 < n_unseen {
                    dataset.prefetch_vector(scratch.id_scratch[m + 1]);
                }
                let nn = scratch.id_scratch[m];
                let v = dataset.get_vertex(nn)?;
                let dist = query_vertex.compare_adsampling(&v, pq_worst, ads_epsilon);
                if dist >= 0.0 {
                    if scratch.pq.size() < search_list_size || dist < pq_worst {
                        admitted += 1;
                    }
                    scratch.pq.insert(DNeighbor::new(nn, dist));
                }
            }

            prev_admitted = admitted;

            if scratch.early_exit.should_exit(converged, admitted) {
                break;
            }
        }

        Ok((0..scratch.pq.size().min(k))
            .map(|i| scratch.pq[i].id)
            .collect())
    }

    /// Parallel batch version of [`Self::search_adsampling`].
    pub fn search_adsampling_batch(
        &self,
        queries: &[[f32; N]],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        ads_epsilon: f32,
    ) -> ANNResult<Vec<Vec<u32>>> {
        self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let results: Vec<Vec<u32>> = queries
            .par_iter()
            .map(|query| {
                self.search_adsampling(
                    query,
                    k,
                    search_list_size,
                    window_size,
                    epsilon,
                    early_exit_limit,
                    ads_epsilon,
                )
                .unwrap_or_default()
            })
            .collect();
        Ok(results)
    }

    /// Diagnostic search: returns (results, converge_step, total_steps, phase1_ndc, phase2_ndc).
    pub fn search_diag(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
    ) -> ANNResult<(Vec<u32>, usize, usize, usize, usize)> {
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let aligned = AlignedQuery(*query);
        let query_vertex = Vertex::new(&aligned.0, 0);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);
        scratch.early_exit.reconfigure(early_exit_limit);

        scratch.seen.insert(entry);
        let entry_dist = {
            let v = dataset.get_vertex(entry)?;
            v.compare(&query_vertex, Metric::L2)
        };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));

        let mut total_steps: usize = 0;
        let mut converge_step: usize = 0;
        let mut converged_yet = false;
        let mut phase1_ndc: usize = 0;
        let mut phase2_ndc: usize = 0;
        let mut prev_admitted: usize = 1;

        while scratch.pq.has_notvisited_node() {
            let neighbor = scratch.pq.closest_notvisited();
            total_steps += 1;
            let id = neighbor.id as usize;

            let converged = scratch.dcc.update(prev_admitted);
            if converged && !converged_yet {
                converge_step = total_steps;
                converged_yet = true;
            }

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
            if !converged {
                phase1_ndc += n_unseen;
            } else {
                phase2_ndc += n_unseen;
            }

            let pq_worst = if scratch.pq.size() >= search_list_size {
                scratch.pq[scratch.pq.size() - 1].distance
            } else {
                f32::MAX
            };
            let mut admitted = 0usize;

            for m in 0..n_unseen {
                if m + 1 < n_unseen {
                    dataset.prefetch_vector(scratch.id_scratch[m + 1]);
                }
                let nn = scratch.id_scratch[m];
                let v = dataset.get_vertex(nn)?;
                let dist = query_vertex.compare_with_bound(&v, pq_worst);
                if dist >= 0.0 {
                    if scratch.pq.size() < search_list_size || dist < pq_worst {
                        admitted += 1;
                    }
                    scratch.pq.insert(DNeighbor::new(nn, dist));
                }
            }

            prev_admitted = admitted;

            if scratch.early_exit.should_exit(converged, admitted) {
                break;
            }
        }

        if !converged_yet {
            converge_step = total_steps;
        }
        let ids = (0..scratch.pq.size().min(k))
            .map(|i| scratch.pq[i].id)
            .collect();
        Ok((ids, converge_step, total_steps, phase1_ndc, phase2_ndc))
    }

    /// Profile search: accumulates nanosecond-level breakdown across all queries.
    pub fn search_profile(
        &self,
        queries: &[[f32; N]],
        _k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<SearchProfileStats> {
        use std::time::Instant;

        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::new(rayon::current_num_threads() + 5, search_list_size)
        });

        let mut stats = SearchProfileStats::default();

        for query in queries {
            let aligned = AlignedQuery(*query);
            let query_vertex = Vertex::new(&aligned.0, 0);
            let mut guard = pool.acquire();
            let scratch = guard.scratch();
            scratch.prepare_for_query(search_list_size);
            scratch.dcc.reconfigure(window_size, epsilon);

            scratch.seen.insert(entry);
            let t0 = Instant::now();
            let entry_dist = {
                let v = dataset.get_vertex(entry)?;
                v.compare(&query_vertex, Metric::L2)
            };
            stats.distance_ns += t0.elapsed().as_nanos() as u64;
            stats.distance_count += 1;
            scratch.pq.insert(DNeighbor::new(entry, entry_dist));

            let mut prev_admitted: usize = 1; // optimistic start

            while scratch.pq.has_notvisited_node() {
                let t_pq = Instant::now();
                let neighbor = scratch.pq.closest_notvisited();
                stats.pq_ops_ns += t_pq.elapsed().as_nanos() as u64;

                let id = neighbor.id as usize;

                let t_conv = Instant::now();
                let converged = scratch.dcc.update(prev_admitted);
                stats.convergence_ns += t_conv.elapsed().as_nanos() as u64;

                let t_graph = Instant::now();
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
                stats.graph_read_ns += t_graph.elapsed().as_nanos() as u64;
                stats.graph_read_count += 1;

                let t_seen = Instant::now();
                stats.seen_ns += t_seen.elapsed().as_nanos() as u64;

                let n_unseen = scratch.id_scratch.len();
                let pq_worst = if scratch.pq.size() >= search_list_size {
                    scratch.pq[scratch.pq.size() - 1].distance
                } else {
                    f32::MAX
                };
                let mut admitted = 0usize;

                for m in 0..n_unseen {
                    let nn = scratch.id_scratch[m];
                    let t_d = Instant::now();
                    let v = dataset.get_vertex(nn)?;
                    let dist = query_vertex.compare(&v, Metric::L2);
                    stats.distance_ns += t_d.elapsed().as_nanos() as u64;
                    stats.distance_count += 1;

                    if dist < pq_worst || scratch.pq.size() < search_list_size {
                        admitted += 1;
                    }

                    let t_ins = Instant::now();
                    scratch.pq.insert(DNeighbor::new(nn, dist));
                    stats.pq_ops_ns += t_ins.elapsed().as_nanos() as u64;
                }

                prev_admitted = admitted;
                stats.iterations += 1;
            }

            stats.queries += 1;
        }

        Ok(stats)
    }
}
