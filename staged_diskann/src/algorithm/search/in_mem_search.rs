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

pub const DEFAULT_SEARCH_LIST_SIZE: usize = 48;
pub const DEFAULT_WINDOW_SIZE: usize = 5;
pub const DEFAULT_EPSILON: f32 = 0.0;

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    pub fn search_default(&self, query: &[f32; N], k: usize) -> ANNResult<Vec<u32>> {
        self.search(query, k, DEFAULT_SEARCH_LIST_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_EPSILON)
    }

    /// Greedy beam search through CompressedGraph + InmemDataset.
    ///
    /// Same data access path as DiskANN: RwLock per node expansion,
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
            let id = neighbor.id;

            let gv = graph.read_vertex(id).map_err(|e| {
                diskann::common::ANNError::log_index_error(format!("read_vertex failed: {e}"))
            })?;

            let phase_converged = scratch.dcc.update(neighbor.distance);
            if phase_converged && !converged_yet {
                converge_step = total_steps;
                converged_yet = true;
            }

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
            let n_unseen = scratch.id_scratch.len();
            if !phase_converged { phase1_ndc += n_unseen; } else { phase2_ndc += n_unseen; }
            drop(gv);

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

        if !converged_yet { converge_step = total_steps; }
        let ids = (0..scratch.pq.size().min(k)).map(|i| scratch.pq[i].id).collect();
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
            .map(|query| self.search(query, k, search_list_size, window_size, epsilon).unwrap_or_default())
            .collect();
        Ok(results)
    }
}
