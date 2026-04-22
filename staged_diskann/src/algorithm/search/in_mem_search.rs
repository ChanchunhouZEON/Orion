/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::StagedDiskANN;
use crate::model::scratch::InMemScratchPool;
use diskann::common::ANNResult;
use diskann::model::{Neighbor as DNeighbor, Vertex};
use rayon::prelude::*;
use vector::{FullPrecisionDistance, Metric};

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
    pub fn search_default(&self, query: &[f32; N], k: usize) -> ANNResult<Vec<u32>> {
        self.search(
            query,
            k,
            DEFAULT_SEARCH_LIST_SIZE,
            DEFAULT_WINDOW_SIZE,
            DEFAULT_EPSILON,
            DEFAULT_WINDOW_SIZE * 2,
        )
    }

    /// Greedy beam search over PhasedGraph + InmemDataset with ParlayANN-style
    /// u8 pre-filter → full-precision compute pipeline.
    ///
    /// Pipeline per hop:
    ///   1. Expand neighbors (local+remote if pre-converged, local+extra if rerank).
    ///   2. Stage-1 u8 pre-filter drops candidates obviously worse than `pq_worst`.
    ///   3. Stage-2 full-precision NEON compare on survivors (branch-free cmov push).
    ///   4. Flush the hop-batch to the PQ via 3-way routing (insert / gallop / merge)
    ///      derived from the K-vs-L crossover map measured in `pq_merge_bench`.
    ///
    /// `search_list_size` = L is the beam width; `early_exit_limit` stops
    /// the search after that many consecutive zero-admit hops in the rerank
    /// phase. SAFETY: all neighbor IDs from `graph.neighbors()` /
    /// `graph.rerank_candidates()` are trusted to be < `dataset.num_active_pts`.
    pub fn search(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
    ) -> ANNResult<Vec<u32>> {
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let aligned = AlignedQuery(*query);
        let query_vertex = Vertex::new(&aligned.0, 0);

        let q_ds = self.ensure_quantized_dataset();
        let q_query = q_ds.quantize_query(query);
        let slope_sq = q_ds.params.slope * q_ds.params.slope;
        let q_slack: f32 = 1.3;

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
            // SAFETY: entry is the stored graph entry point, always valid.
            let v_arr = unsafe { dataset.get_vertex_unchecked(entry) };
            let v = Vertex::new(v_arr, entry);
            query_vertex.compare(&v, Metric::L2)
        };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));

        let mut prev_admitted: usize = 1;
        // Unified flush cadence: 1 hop pre-converged, 4 hops post-converged
        // (cross-hop accumulation amortizes merge cost in the sparse rerank
        // phase while keeping pq_worst fresh enough for early_exit).
        let mut hops_since_flush: usize = 0;
        // Prefetch lookahead for stage-1/stage-2 distance loops. 8 iters of
        // ~20 ns compute ≈ 160 ns runway — enough to hide L3 / partial DRAM.
        const PF_BATCH: usize = 8;

        while scratch.pq.has_notvisited_node() {
            let id = scratch.pq.closest_notvisited().id as usize;

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

            // Stage 1: u8 pre-filter.
            let pf_head = n_unseen.min(PF_BATCH);
            for m in 0..pf_head {
                q_ds.prefetch_vector(scratch.id_scratch[m]);
            }
            let q_threshold: f32 = if pq_worst.is_finite() {
                pq_worst * slope_sq * q_slack
            } else {
                f32::MAX
            };
            let mut pre_kept: usize = 0;
            for m in 0..n_unseen {
                if m + PF_BATCH < n_unseen {
                    q_ds.prefetch_vector(scratch.id_scratch[m + PF_BATCH]);
                }
                let nn = scratch.id_scratch[m];
                let q_dist = unsafe { q_ds.qdist(nn, &q_query) };
                if q_dist <= q_threshold {
                    scratch.id_scratch[pre_kept] = nn;
                    pre_kept += 1;
                }
            }

            // Stage 2: branch-free cmov-style write + conditional advance.
            let pf2 = pre_kept.min(PF_BATCH);
            for m in 0..pf2 {
                dataset.prefetch_vector(scratch.id_scratch[m]);
            }
            let hop_start = scratch.dist_buffer.len();
            scratch.dist_buffer.reserve(pre_kept);
            let hop_admits: usize = unsafe {
                let base = scratch.dist_buffer.as_mut_ptr().add(hop_start);
                let mut w = 0usize;
                for m in 0..pre_kept {
                    if m + PF_BATCH < pre_kept {
                        dataset.prefetch_vector(scratch.id_scratch[m + PF_BATCH]);
                    }
                    let nn = scratch.id_scratch[m];
                    let v_arr = dataset.get_vertex_unchecked(nn);
                    let v = Vertex::new(v_arr, nn);
                    let dist = query_vertex.compare(&v, Metric::L2);
                    base.add(w).write(DNeighbor::new(nn, dist));
                    w += (dist < pq_worst) as usize;
                }
                scratch.dist_buffer.set_len(hop_start + w);
                w
            };
            prev_admitted = hop_admits;

            let should_exit = scratch.early_exit.should_exit(converged, hop_admits);
            let will_stop = should_exit | !scratch.pq.has_notvisited_node();

            hops_since_flush += 1;
            let flush_interval = 1 + 3 * (converged as usize);
            let must_flush = (hops_since_flush >= flush_interval) | will_stop;

            if must_flush {
                // 3-way routing from `pq_merge_bench` crossover map:
                //   K < L/12  → per-insert   (cache-friendly for small K)
                //   K > L/2   → linear merge (bulk rewrite)
                //   otherwise → gallop merge (log-factor wins)
                let cnt = scratch.dist_buffer.len();
                if cnt > 0 {
                    if cnt * 12 < search_list_size {
                        for c in scratch.dist_buffer.drain(..) {
                            scratch.pq.insert(c);
                        }
                    } else {
                        scratch.dist_buffer.sort_unstable_by(|a, b| {
                            a.distance
                                .total_cmp(&b.distance)
                                .then_with(|| a.id.cmp(&b.id))
                        });
                        if cnt * 2 > search_list_size {
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
            }

            if should_exit {
                break;
            }
        }

        Ok((0..scratch.pq.size().min(k))
            .map(|i| scratch.pq[i].id)
            .collect())
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

    /// Parallel batch search using rayon.
    pub fn search_batch(
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
                self.search(
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
