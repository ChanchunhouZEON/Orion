/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::model::scratch::InMemScratchPool;
use crate::{DistanceConvergenceChecker, StagedDiskANN};
use diskann::common::ANNResult;
use diskann::model::Neighbor as DNeighbor;
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
    ///
    /// Equivalent to `search(query, k, DEFAULT_SEARCH_LIST_SIZE,
    /// DEFAULT_WINDOW_SIZE, DEFAULT_EPSILON)`.
    pub fn search_default(&self, query: &[f32; N], k: usize) -> ANNResult<Vec<u32>> {
        self.search(query, k, DEFAULT_SEARCH_LIST_SIZE, DEFAULT_WINDOW_SIZE, DEFAULT_EPSILON)
    }

    /// Greedy beam search using the flat CSR adjacency list.
    ///
    /// Matches DiskANN's `InMemIndex::search` pattern:
    /// - Pool-based scratch (`InMemScratchPool`): pre-allocated scratch objects
    ///   checked out per query and returned on drop — no per-query allocation.
    /// - Seen-before-distance: neighbor IDs filtered through `seen` (hashbrown
    ///   HashSet, pre-allocated 20×L) before any distance is computed.
    /// - Flat CSR graph (`search_csr_offsets` / `search_csr_neighbors`):
    ///   no RwLock acquire, no pointer chasing — both full and compressed
    ///   neighbor slices are direct contiguous slice indexing.
    /// - SIMD distance via `FullPrecisionDistance::distance_compare` (NEON/AVX2).
    pub fn search(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<Vec<u32>> {
        let entry = self.entry;
        let data_arrays = &self.data_arrays;
        let csr_offsets = &self.search_csr_offsets;
        let csr_compressed_end = &self.search_csr_compressed_end;
        let csr_neighbors = &self.search_csr_neighbors;

        // Lazily initialize the scratch pool on first search call.
        let pool = self.inmem_scratch_pool.get_or_init(|| {
            let n_threads = rayon::current_num_threads().max(8);
            InMemScratchPool::new(n_threads, search_list_size)
        });

        let mut guard = pool.acquire();
        let scratch = guard.scratch();
        scratch.prepare_for_query(search_list_size);

        let mut dcc = DistanceConvergenceChecker::new(window_size, epsilon);

        // Seed the queue with the graph entry point.
        scratch.seen.insert(entry);
        let entry_dist = <[f32; N] as FullPrecisionDistance<f32, N>>::distance_compare(
            query,
            &data_arrays[entry as usize],
            Metric::L2,
        );
        scratch.pq.insert(DNeighbor::new(entry, entry_dist));

        while scratch.pq.has_notvisited_node() {
            let neighbor = scratch.pq.closest_notvisited();

            let id = neighbor.id as usize;
            let start = csr_offsets[id] as usize;
            let end = csr_offsets[id + 1] as usize;

            let phase_converged = dcc.update(neighbor.distance);
            let neighbors_to_use = if !phase_converged {
                // Phase 1: full graph neighbors (direct slice, no lock)
                &csr_neighbors[start..end]
            } else {
                // Phase 2: compressed neighbors only
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

            // Phase B: compute SIMD distance for each unseen neighbor.
            let n_unseen = scratch.id_scratch.len();
            for i in 0..n_unseen {
                let nn = scratch.id_scratch[i];
                let dist = <[f32; N] as FullPrecisionDistance<f32, N>>::distance_compare(
                    query,
                    &data_arrays[nn as usize],
                    Metric::L2,
                );
                scratch.pq.insert(DNeighbor::new(nn, dist));
            }
        }

        // Neighbors are sorted by ascending distance; take the first k.
        Ok((0..scratch.pq.size().min(k)).map(|i| scratch.pq[i].id).collect())
    }
}
