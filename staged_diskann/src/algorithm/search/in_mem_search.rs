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

pub const DEFAULT_SEARCH_LIST_SIZE: usize = 48;
pub const DEFAULT_WINDOW_SIZE: usize = 5;
pub const DEFAULT_EPSILON: f32 = 0.0;

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
        )
    }

    /// Greedy beam search through CsrGraph + InmemDataset.
    ///
    /// Lock-free graph traversal via CsrGraph left-right double buffering.
    /// InmemDataset::get_vertex + Vertex::compare for distance.
    pub fn search(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<Vec<u32>> {
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let query_vertex = Vertex::new(query, 0);

        let pool = self
            .inmem_scratch_pool
            .get_or_init(|| InMemScratchPool::new(32, search_list_size));

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.ensure_capacity(graph.num_nodes());
        scratch.dcc.reconfigure(window_size, epsilon);

        scratch.seen.insert(entry);
        let entry_dist = {
            let v = dataset.get_vertex(entry)?;
            v.compare(&query_vertex, Metric::L2)
        };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));

        while scratch.pq.has_notvisited_node() {
            let neighbor = scratch.pq.closest_notvisited();
            let id = neighbor.id;

            // Prefetch the NEXT node's graph slot while processing current.
            if let Some(next) = scratch.pq.peek_notvisited() {
                graph.prefetch_node(next.id as usize);
                dataset.prefetch_vector(next.id);
            }

            let phase_converged = scratch.dcc.update(neighbor.distance);
            let neighbors_to_use = if !phase_converged {
                graph.neighbors(id as usize)
            } else {
                graph.compressed_neighbors(id as usize)
            };

            scratch.id_scratch.clear();
            for &nn in neighbors_to_use {
                if scratch.seen.insert(nn) {
                    scratch.id_scratch.push(nn);
                }
            }

            // Prefetch first unseen neighbor's vector before entering distance loop.
            let n_unseen = scratch.id_scratch.len();
            if n_unseen > 0 {
                dataset.prefetch_vector(scratch.id_scratch[0]);
            }
            for m in 0..n_unseen {
                if m + 1 < n_unseen {
                    dataset.prefetch_vector(scratch.id_scratch[m + 1]);
                }
                let nn = scratch.id_scratch[m];
                let v = dataset.get_vertex(nn)?;
                let dist = query_vertex.compare(&v, Metric::L2);
                scratch.pq.insert(DNeighbor::new(nn, dist));
            }
        }

        Ok((0..scratch.pq.size().min(k))
            .map(|i| scratch.pq[i].id)
            .collect())
    }

    /// Diagnostic search: returns convergence statistics.
    pub fn search_diag(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<(Vec<u32>, usize, usize, usize, usize)> {
        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;
        let query_vertex = Vertex::new(query, 0);

        let pool = self
            .inmem_scratch_pool
            .get_or_init(|| InMemScratchPool::new(32, search_list_size));

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.ensure_capacity(graph.num_nodes());
        scratch.dcc.reconfigure(window_size, epsilon);

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

        while scratch.pq.has_notvisited_node() {
            let neighbor = scratch.pq.closest_notvisited();
            total_steps += 1;
            let id = neighbor.id;

            let phase_converged = scratch.dcc.update(neighbor.distance);
            if phase_converged && !converged_yet {
                converge_step = total_steps;
                converged_yet = true;
            }

            let neighbors_to_use = if !phase_converged {
                graph.neighbors(id as usize)
            } else {
                graph.compressed_neighbors(id as usize)
            };

            scratch.id_scratch.clear();
            for &nn in neighbors_to_use {
                if scratch.seen.insert(nn) {
                    scratch.id_scratch.push(nn);
                }
            }
            let n_unseen = scratch.id_scratch.len();
            if !phase_converged {
                phase1_ndc += n_unseen;
            } else {
                phase2_ndc += n_unseen;
            }

            for m in 0..n_unseen {
                if m + 1 < n_unseen {
                    dataset.prefetch_vector(scratch.id_scratch[m + 1]);
                }
                let nn = scratch.id_scratch[m];
                let v = dataset.get_vertex(nn)?;
                let dist = query_vertex.compare(&v, Metric::L2);
                scratch.pq.insert(DNeighbor::new(nn, dist));
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
    /// Returns (results, SearchProfileStats).
    pub fn search_profile(
        &self,
        queries: &[[f32; N]],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<SearchProfileStats> {
        use std::time::Instant;

        let entry = self.entry;
        let dataset = &self.dataset;
        let graph = &self.graph;

        let pool = self
            .inmem_scratch_pool
            .get_or_init(|| InMemScratchPool::new(32, search_list_size));

        let mut stats = SearchProfileStats::default();

        for query in queries {
            let query_vertex = Vertex::new(query, 0);
            let mut guard = pool.acquire();
            let scratch = guard.scratch();
            scratch.prepare_for_query(search_list_size);
            scratch.ensure_capacity(graph.num_nodes());
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

            while scratch.pq.has_notvisited_node() {
                let t_pq = Instant::now();
                let neighbor = scratch.pq.closest_notvisited();
                stats.pq_ops_ns += t_pq.elapsed().as_nanos() as u64;

                let id = neighbor.id;

                let t_conv = Instant::now();
                let phase_converged = scratch.dcc.update(neighbor.distance);
                stats.convergence_ns += t_conv.elapsed().as_nanos() as u64;

                let t_graph = Instant::now();
                let neighbors_to_use = if !phase_converged {
                    graph.neighbors(id as usize)
                } else {
                    graph.compressed_neighbors(id as usize)
                };
                stats.graph_read_ns += t_graph.elapsed().as_nanos() as u64;
                stats.graph_read_count += 1;

                let t_seen = Instant::now();
                scratch.id_scratch.clear();
                for &nn in neighbors_to_use {
                    if scratch.seen.insert(nn) {
                        scratch.id_scratch.push(nn);
                    }
                }
                stats.seen_ns += t_seen.elapsed().as_nanos() as u64;

                let n_unseen = scratch.id_scratch.len();
                for m in 0..n_unseen {
                    let nn = scratch.id_scratch[m];
                    let t_d = Instant::now();
                    let v = dataset.get_vertex(nn)?;
                    let dist = query_vertex.compare(&v, Metric::L2);
                    stats.distance_ns += t_d.elapsed().as_nanos() as u64;
                    stats.distance_count += 1;

                    let t_ins = Instant::now();
                    scratch.pq.insert(DNeighbor::new(nn, dist));
                    stats.pq_ops_ns += t_ins.elapsed().as_nanos() as u64;
                }

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
    ) -> ANNResult<Vec<Vec<u32>>> {
        self.inmem_scratch_pool
            .get_or_init(|| InMemScratchPool::new(32, search_list_size));

        let results: Vec<Vec<u32>> = queries
            .par_iter()
            .map(|query| {
                self.search(query, k, search_list_size, window_size, epsilon)
                    .unwrap_or_default()
            })
            .collect();
        Ok(results)
    }
}
