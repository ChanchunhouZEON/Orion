/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::Orion;
use crate::model::Neighbor as DNeighbor;
use crate::model::scratch::InMemScratchPool;
use diskann::common::ANNResult;
use diskann::model::Vertex;
use vector::{FullPrecisionDistance, Metric};

// ── Search-loop tuning constants ──────────────────────────────────────────
/// Cache-line lookahead for the software-pipelined `DistanceStream`
/// **Stage-1 quantized prefilter**. Stage-1 vert sizes:
/// glove100 i8 = **1 line/vert** (LA=12 ⇒ 12 verts ahead),
/// glove100 i16 = 2 lines/vert (LA=12 ⇒ 6 verts ahead),
/// SIFT u8 = 1 line/vert. Steady-state issues `LA_Q` prfm per outer
/// iter; M2 MSHR ≈ 12. Override via `ORION_DSTREAM_LA_Q=<N>`
/// (back-compat: `ORION_DSTREAM_LOOKAHEAD` still honored).
pub(super) fn dstream_la_q() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ORION_DSTREAM_LA_Q")
            .ok()
            .or_else(|| std::env::var("ORION_DSTREAM_LOOKAHEAD").ok())
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
/// Override via `ORION_DSTREAM_LA_TRUTH=<N>`.
pub(super) fn dstream_la_truth() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ORION_DSTREAM_LA_TRUTH")
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

impl<const N: usize> Orion<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
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
            } else if crate::algorithm::search::include_extras() {
                let (local, extra) = graph.rerank_candidates(id);
                for &nn in local.iter().chain(extra.iter()) {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }
            } else {
                // Ablation: extras disabled → fall back to local+remote
                // (same as the pre-convergence walk). See module docs
                // on `INCLUDE_EXTRAS`.
                for &nn in graph.neighbors(id) {
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
                } else if crate::algorithm::search::include_extras() {
                    let (local, extra) = graph.rerank_candidates(id);
                    for &nn in local.iter().chain(extra.iter()) {
                        if scratch.seen.insert(nn) {
                            scratch.id_scratch.push(nn);
                        }
                    }
                } else {
                    // Ablation: extras disabled → fall back to
                    // local+remote (same as pre-convergence walk).
                    for &nn in graph.neighbors(id) {
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

// ── PQ helpers (formerly in utils.rs) ──────────────────────
use diskann::common::ANNError;
use diskann::model::FixedChunkPQTable;
use std::sync::Arc;

impl<const N: usize> Orion<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Compute PQ distance for a point given its ID and pre-computed chunk distances.
    /// SAFETY: Should be invoked only when pq is activated.
    #[inline]
    #[allow(dead_code)]
    pub(super) fn pq_distance(&self, point_id: u32, pq_dists: &[f32]) -> ANNResult<f32> {
        let (pq, pq_codes, num_pq_chunks) = self.get_unwrapped_pq_component()?;

        let idx = point_id as usize;
        let code_start = idx * num_pq_chunks;
        let code = &pq_codes[code_start..code_start + num_pq_chunks];
        Ok(pq.adc_distance(code, pq_dists))
    }

    pub(super) fn get_unwrapped_pq_component(
        &self,
    ) -> ANNResult<(&Arc<FixedChunkPQTable>, &Vec<u8>, usize)> {
        let pq = self.pq.as_ref().ok_or_else(|| {
            ANNError::log_pq_error("Fixed Chunk PQ Table is None for now".to_string())
        })?;
        let pq_codes = self
            .pq_codes
            .as_ref()
            .ok_or_else(|| ANNError::log_pq_error("PQ codes is None for now".to_string()))?;
        let num_pq_chunks = self.num_pq_chunks.ok_or_else(|| {
            ANNError::log_pq_error("Number of pq chunks is None for now".to_string())
        })?;

        Ok((pq, pq_codes, num_pq_chunks))
    }
}

/// Detailed profiling breakdown for a single search query.
#[derive(Debug, Clone)]
pub struct SearchProfile {
    pub total_us: f64,
    pub adc_table_us: f64,
    pub phase1_us: f64,
    pub phase2_us: f64,
    pub phase2_graph_read_us: f64,
    pub phase2_prefetch_us: f64,
    pub phase2_adc_us: f64,
    pub phase2_async_adc_us: f64,
    pub rerank_us: f64,
    pub phase1_iters: u32,
    pub phase2_iters: u32,
    pub visited_count: u32,
    pub cache_hits: u32,
    pub async_batches: u32,
}
