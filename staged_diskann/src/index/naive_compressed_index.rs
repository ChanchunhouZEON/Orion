/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Naive compression baseline: `compressed_degree = min(max_pruned_degree, degree)`
//! with NO reordering — simply take the first N neighbors in Vamana order.
//!
//! This module exists to compare against the cluster-aware compression in
//! `StagedDiskANN`, isolating the quality benefit of clustering.

use crate::model::scratch::InMemScratchPool;
use diskann::model::{CsrGraph, InmemDataset};
use rayon::prelude::*;
use std::sync::OnceLock;
use vector::FullPrecisionDistance;

/// Naive compressed DiskANN: two-phase search with prefix-based compression.
///
/// Each node's `compressed_degree` is set to `min(max_pruned_degree, degree)`.
/// Neighbors are NOT reordered — the Vamana insertion order is preserved.
/// This provides a fair baseline to measure the benefit of cluster-aware
/// neighbor selection in `StagedDiskANN`.
pub struct NaiveStagedDiskANN<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    pub dataset: InmemDataset<f32, N>,
    pub graph: CsrGraph,
    pub entry: u32,
    pub num_nodes: usize,
    pub max_pruned_degree: usize,
    pub(crate) inmem_scratch_pool: OnceLock<InMemScratchPool>,
}

impl<const N: usize> NaiveStagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Build a naive compressed index: set compressed_degree per node without
    /// clustering or reordering. No candidate sets needed.
    pub fn new(
        dataset: InmemDataset<f32, N>,
        #[allow(unused_mut)] mut csr: CsrGraph,
        entry: u32,
        max_connection_clusters: usize,
        max_connection_per_cluster: usize,
    ) -> Self {
        let num_nodes = csr.num_nodes();
        let max_pruned_degree = max_connection_clusters * max_connection_per_cluster;

        log::info!(
            "NaiveStagedDiskANN: setting compressed_degree = min({}, degree) for {} nodes",
            max_pruned_degree,
            num_nodes,
        );

        // Set compressed_degree for each node by writing the same neighbors
        // with a split at min(max_pruned_degree, degree). No reordering.
        (0..num_nodes).into_par_iter().for_each(|node| {
            let nbrs = csr.neighbors(node);
            let deg = nbrs.len();
            let cd = deg.min(max_pruned_degree);
            if cd > 0 {
                // Write same order: first `cd` as compressed, rest as non-compressed.
                unsafe {
                    csr.set_neighbors_reordered_unchecked(node, &nbrs[..cd], &nbrs[cd..]);
                }
            }
        });

        Self {
            dataset,
            graph: csr,
            entry,
            num_nodes,
            max_pruned_degree,
            inmem_scratch_pool: OnceLock::new(),
        }
    }

    // Search methods are identical to StagedDiskANN — reuse via in_mem_search.
    // We implement them here by delegating to the same greedy beam search logic.

    pub fn search(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> diskann::common::ANNResult<Vec<u32>> {
        use diskann::model::{Neighbor as DNeighbor, Vertex};
        use vector::Metric;

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

        let mut prev_admitted: usize = 1; // optimistic start

        while scratch.pq.has_notvisited_node() {
            let neighbor = scratch.pq.closest_notvisited();
            let id = neighbor.id;

            if let Some(next) = scratch.pq.peek_notvisited() {
                graph.prefetch_node(next.id as usize);
                dataset.prefetch_vector(next.id);
            }

            let phase_converged = scratch.dcc.update(prev_admitted);
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
                let dist = query_vertex.compare(&v, Metric::L2);
                if dist < pq_worst || scratch.pq.size() < search_list_size {
                    admitted += 1;
                }
                scratch.pq.insert(DNeighbor::new(nn, dist));
            }

            prev_admitted = admitted;
        }

        Ok((0..scratch.pq.size().min(k))
            .map(|i| scratch.pq[i].id)
            .collect())
    }

    pub fn search_batch(
        &self,
        queries: &[[f32; N]],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>> {
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
