/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::model::scratch::InMemScratchPool;
use crate::StagedDiskANN;
use diskann::common::ANNResult;
use diskann::model::{Neighbor as DNeighbor, Vertex};
use rayon::prelude::*;
use vector::{FullPrecisionDistance, Metric};

/// Default search list size matching DiskANN (Vamana) benchmark configuration.
pub const DEFAULT_SEARCH_LIST_SIZE: usize = 48;
/// Default sliding-window size for the convergence checker.
pub const DEFAULT_WINDOW_SIZE: usize = 5;
/// Default epsilon for the convergence checker.
/// 0.0 means the checker never triggers, so the search always runs Phase 1
/// (full graph), making it equivalent to plain DiskANN greedy search.
pub const DEFAULT_EPSILON: f32 = 0.0;

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Search using hardcoded defaults tuned to match DiskANN (Vamana) recall.
    pub fn search_default(&self, query: &[f32; N], k: usize) -> ANNResult<Vec<u32>> {
        self.search(
            query,
            k,
            DEFAULT_SEARCH_LIST_SIZE,
            DEFAULT_WINDOW_SIZE,
            DEFAULT_EPSILON,
        )
    }

    /// Greedy beam search using the flat CSR adjacency list.
    ///
    /// Vector access uses `InmemDataset::get_vertex` + `Vertex::compare`,
    /// the same code path as DiskANN's search. Prefetching uses
    /// `InmemDataset::prefetch_vector`.
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
        let csr_offsets = &self.search_csr_offsets;
        let csr_compressed_end = &self.search_csr_compressed_end;
        let csr_neighbors = &self.search_csr_neighbors;
        let query_vertex = Vertex::new(query, 0);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            let n_threads = rayon::current_num_threads().max(8);
            InMemScratchPool::new(n_threads, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
        scratch.dcc.reconfigure(window_size, epsilon);

        // Seed the queue with the graph entry point.
        scratch.seen.insert(entry);
        let entry_dist = {
            let v = dataset.get_vertex(entry)?;
            v.compare(&query_vertex, Metric::L2)
        };
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));

        while scratch.pq.has_notvisited_node() {
            let neighbor = scratch.pq.closest_notvisited();

            let id = neighbor.id as usize;
            let start = csr_offsets[id] as usize;
            let end = csr_offsets[id + 1] as usize;

            // Prefetch the NEXT closest unvisited node's CSR neighbor slice + vector data.
            if let Some(next) = scratch.pq.peek_notvisited() {
                let nid = next.id as usize;
                let ns = csr_offsets[nid] as usize;
                if ns < csr_neighbors.len() {
                    vector::prefetch_vector(
                        &csr_neighbors[ns..csr_neighbors.len().min(ns + 32)],
                    );
                }
                dataset.prefetch_vector(next.id);
            }

            let phase_converged = scratch.dcc.update(neighbor.distance);
            let neighbors_to_use = if !phase_converged {
                &csr_neighbors[start..end]
            } else {
                let cd = csr_compressed_end[id] as usize;
                &csr_neighbors[start..cd]
            };

            // Phase A: filter through `seen`, collect unseen IDs.
            scratch.id_scratch.clear();
            for &nn in neighbors_to_use {
                if scratch.seen.insert(nn) {
                    scratch.id_scratch.push(nn);
                }
            }

            // Phase B: compute distance for each unseen neighbor.
            // Prefetch next vector while computing current distance.
            let n_unseen = scratch.id_scratch.len();
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

    /// Search through `CompressedGraph` (RwLock) + `InmemDataset`.
    ///
    /// Graph access: acquires a read lock per node expansion (mirrors DiskANN).
    /// Data access: `InmemDataset::get_vertex` + `Vertex::compare`.
    /// This isolates the algorithmic benefit (two-phase convergence) from the
    /// data-structure benefit (lock-free CSR).
    pub fn search_rwlock(
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

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            let n_threads = rayon::current_num_threads().max(8);
            InMemScratchPool::new(n_threads, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
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

            // Read lock on this node's neighbor list (mirrors DiskANN's pattern).
            let gv = graph.read_vertex(id).map_err(|e| {
                diskann::common::ANNError::log_index_error(format!("read_vertex failed: {e}"))
            })?;

            let phase_converged = scratch.dcc.update(neighbor.distance);
            let neighbors_to_use = if !phase_converged {
                gv.get_neighbors()
            } else {
                gv.get_compressed_neighbors()
            };

            scratch.id_scratch.clear();
            for &nn in neighbors_to_use {
                if scratch.seen.insert(nn) {
                    scratch.id_scratch.push(nn);
                }
            }
            drop(gv);

            let n_unseen = scratch.id_scratch.len();
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

    /// Diagnostic search: same as `search` but returns convergence statistics.
    ///
    /// Returns `(result_ids, steps_before_converge, total_steps, phase1_ndc, phase2_ndc)`.
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
        let csr_offsets = &self.search_csr_offsets;
        let csr_compressed_end = &self.search_csr_compressed_end;
        let csr_neighbors = &self.search_csr_neighbors;
        let query_vertex = Vertex::new(query, 0);

        let pool = self.inmem_scratch_pool.get_or_init(|| {
            let n_threads = rayon::current_num_threads().max(8);
            InMemScratchPool::new(n_threads, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);
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

            let id = neighbor.id as usize;
            let start = csr_offsets[id] as usize;
            let end = csr_offsets[id + 1] as usize;

            let phase_converged = scratch.dcc.update(neighbor.distance);
            if phase_converged && !converged_yet {
                converge_step = total_steps;
                converged_yet = true;
            }

            let neighbors_to_use = if !phase_converged {
                &csr_neighbors[start..end]
            } else {
                let cd = csr_compressed_end[id] as usize;
                &csr_neighbors[start..cd]
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

    /// Parallel batch search using rayon.
    pub fn search_batch(
        &self,
        queries: &[[f32; N]],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<Vec<Vec<u32>>> {
        self.inmem_scratch_pool.get_or_init(|| {
            let n_threads = rayon::current_num_threads().max(8);
            InMemScratchPool::new(n_threads, search_list_size)
        });

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
