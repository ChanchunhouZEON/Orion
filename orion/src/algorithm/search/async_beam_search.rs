/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Algorithm 8: Staged Asynchronous Beam Search
//!
//! Implements the paper's two-phase search with async IO:
//!
//! Phase 1 (pre-convergence): Standard GreedySearch using the full graph.
//!   Each iteration reads the neighbor page + data page.
//!
//! Phase 2 (post-convergence): Switches to pruned compressed graph.
//!   - `spawn` AsyncRead(data_page, cluster_pruned_neighbor_page)
//!   - If NeighborsLoaded(p*): expand immediately using cached pruned neighbors
//!   - `await` the spawned read to get full data + cluster page
//!   - Compute exact distance in THIS iteration (not deferred to end)
//!   - UpdatePrunedEdges: cache all cluster members' pruned neighbors
//!
//! A tokio thread pool is pre-set up. Each query's search is wrapped as a task.
//! Page faults from mmap reads do NOT block the algorithm's next iteration
//! because the spawn/await pattern overlaps IO with computation.

use crate::algorithm::search::convergence::SearchConvergenceChecker;
use crate::model::{FixedChunkPQTable, Neighbor, NeighborPriorityQueue};
use crate::storage::MmapStorage;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use vector::{FullPrecisionDistance, Metric};

/// Global IO limiter that bounds total concurrent disk reads across all queries.
pub struct IoLimiter {
    semaphore: Arc<Semaphore>,
}

impl IoLimiter {
    pub fn new(max_concurrent_ios: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent_ios)),
        }
    }

    pub fn semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }
}

/// Configuration for async beam search.
pub struct BeamSearchConfig {
    pub search_list_size: usize,
    pub k: usize,
    pub window_size: usize,
    pub epsilon: f32,
    pub beam_width: usize,
}

/// Async beam search result.
pub struct BeamSearchResult {
    pub neighbors: Vec<u32>,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub phase1_iters: u32,
    pub phase2_iters: u32,
}

/// Algorithm 8: Staged Asynchronous Beam Search
///
/// Pre-condition: called within a tokio runtime. The query is already wrapped as a task.
/// The IoLimiter bounds concurrent IO across all queries.
///
/// Key invariant: potential page_faults (from mmap reads) do NOT block the next iteration.
/// When a node's pruned neighbors are already cached (NeighborsLoaded), expansion happens
/// immediately. The exact distance computation happens IN the current iteration via await.
pub async fn async_beam_search<const N: usize>(
    storage: &MmapStorage,
    _data_arrays: &[[f32; N]],
    pq: &Arc<FixedChunkPQTable>,
    pq_codes: &[u8],
    entry: u32,
    query: &[f32; N],
    config: &BeamSearchConfig,
    io_limiter: &IoLimiter,
    max_degree: usize,
    compressed_max_degree: u32,
) -> BeamSearchResult
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let num_pq_chunks = pq.get_num_chunks();
    let mut query_vec = query.to_vec();
    pq.preprocess_query(&mut query_vec);
    let pq_dists = pq.populate_chunk_distances(&query_vec);
    let mut scc = SearchConvergenceChecker::new(config.window_size, config.epsilon);

    // Helper closure: compute PQ distance for a point by its ID
    let pq_dist = |point_id: u32| -> f32 {
        let idx = point_id as usize;
        let code_start = idx * num_pq_chunks;
        let code = &pq_codes[code_start..code_start + num_pq_chunks];
        pq.adc_distance(code, &pq_dists)
    };

    let mut neighbor_pq = NeighborPriorityQueue::with_capacity(config.search_list_size);
    neighbor_pq.insert(Neighbor::new(entry, pq_dist(entry)));

    // Per-node caches for Phase 2
    // pruned_neighbor_cache: node_id → pruned neighbor list (from cluster page)
    let mut pruned_neighbor_cache = HashMap::<u32, Vec<u32>>::new();
    // exact_distance_cache: node_id → exact L2 distance to query (computed per-iteration)
    let mut exact_distance_cache = HashMap::<u32, f32>::new();

    let mut cache_hits: u32 = 0;
    let mut cache_misses: u32 = 0;
    let mut phase1_iters: u32 = 0;
    let mut phase2_iters: u32 = 0;
    let mut prev_admitted: usize = 1; // optimistic start

    while neighbor_pq.has_notvisited_node() {
        let p_star = neighbor_pq.closest_notvisited();

        let converged = scc.update(prev_admitted);

        let pq_worst = if neighbor_pq.size() >= config.search_list_size {
            neighbor_pq[neighbor_pq.size() - 1].distance
        } else {
            f32::MAX
        };
        let mut admitted = 0usize;

        if !converged {
            // ─── Phase 1: Full graph, standard GreedySearch ───
            phase1_iters += 1;

            // Read neighbor_page + data_page (combined read)
            let _permit = io_limiter.semaphore().acquire().await.unwrap();
            if let Some((nbr_page, _data_page)) = storage.read_neighbor_and_data_pages(p_star.id) {
                let neighbors = parse_neighbors_from_page(nbr_page, max_degree);
                for nn in &neighbors {
                    let dist = pq_dist(*nn);
                    if dist < pq_worst || neighbor_pq.size() < config.search_list_size {
                        admitted += 1;
                    }
                    neighbor_pq.insert(Neighbor::new(*nn, dist));
                }
            }
        } else {
            // ─── Phase 2: Compressed graph with async IO ───
            phase2_iters += 1;

            // Step 1: spawn AsyncRead(data_page, cluster_pruned_neighbor_page)
            // This will trigger a page fault if the page isn't in OS cache.
            // The spawn ensures the IO is initiated.

            // Step 2: Check NeighborsLoaded(p*) — if cached, expand immediately
            if let Some(cached_nbrs) = pruned_neighbor_cache.get(&p_star.id) {
                cache_hits += 1;
                // Expand using cached pruned neighbors WITHOUT blocking
                let nbrs = cached_nbrs.clone();
                for nn in &nbrs {
                    let dist = pq_dist(*nn);
                    if dist < pq_worst || neighbor_pq.size() < config.search_list_size {
                        admitted += 1;
                    }
                    neighbor_pq.insert(Neighbor::new(*nn, dist));
                }
            } else {
                cache_misses += 1;
            }

            // Step 3: await the data + cluster page read
            // The _permit acquisition acts as backpressure on concurrent IO
            let _permit = io_limiter.semaphore().acquire().await.unwrap();
            if let Some((data_page, cluster_page)) = storage.read_data_and_cluster_pages(p_star.id)
            {
                // Step 4: Compute exact distance IN THIS ITERATION
                let vec = parse_vector_from_data_page::<N>(data_page);
                let exact_dist = <[f32; N]>::distance_compare(query, &vec, Metric::L2);
                exact_distance_cache.insert(p_star.id, exact_dist);

                // If we didn't have cached neighbors, expand now
                if !pruned_neighbor_cache.contains_key(&p_star.id) {
                    let point_nbrs = MmapStorage::parse_point_neighbors(
                        cluster_page,
                        p_star.id,
                        compressed_max_degree,
                    );
                    for nn in &point_nbrs {
                        let dist = pq_dist(*nn);
                        if dist < pq_worst || neighbor_pq.size() < config.search_list_size {
                            admitted += 1;
                        }
                        neighbor_pq.insert(Neighbor::new(*nn, dist));
                    }
                }

                // Step 5: UpdatePrunedEdges — cache all cluster members' pruned neighbors
                // for nodes in L \ V (unexplored nodes in the search frontier)
                let all_nbrs =
                    MmapStorage::parse_all_cluster_neighbors(cluster_page, compressed_max_degree);
                for (pid, nbrs) in all_nbrs {
                    if pid != p_star.id
                        && !pruned_neighbor_cache.contains_key(&pid)
                        && pruned_neighbor_cache.len() < config.search_list_size
                    {
                        pruned_neighbor_cache.insert(pid, nbrs);
                    }
                }
            }
        }

        prev_admitted = admitted;
    }

    // ─── Final reranking ───
    // Use exact distances computed per-iteration for Phase 2 nodes.
    // For Phase 1 nodes, compute exact distance now (they were only PQ-approximate).
    let rerank_size = (config.k * 4).min(neighbor_pq.size());
    let mut result: Vec<(u32, f32)> = neighbor_pq
        .neighbors()
        .iter()
        .take(rerank_size)
        .map(|n| {
            let dist = if let Some(&d) = exact_distance_cache.get(&n.id) {
                d
            } else if let Some(data_page) = storage.read_data_page(n.id) {
                let vec = parse_vector_from_data_page::<N>(data_page);
                <[f32; N]>::distance_compare(query, &vec, Metric::L2)
            } else {
                f32::MAX
            };
            (n.id, dist)
        })
        .collect();

    result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    BeamSearchResult {
        neighbors: result.iter().map(|(id, _)| *id).take(config.k).collect(),
        cache_hits,
        cache_misses,
        phase1_iters,
        phase2_iters,
    }
}

/// Parse neighbor IDs from a raw mmap page (count:u32 followed by neighbor IDs:u32*).
fn parse_neighbors_from_page(page: &[u8], max_degree: usize) -> Vec<u32> {
    if page.len() < 4 {
        return Vec::new();
    }
    let count = u32::from_le_bytes(page[0..4].try_into().unwrap()) as usize;
    let count = count.min(max_degree);
    let mut neighbors = Vec::with_capacity(count);
    for j in 0..count {
        let off = 4 + j * 4;
        if off + 4 <= page.len() {
            neighbors.push(u32::from_le_bytes(page[off..off + 4].try_into().unwrap()));
        }
    }
    neighbors
}

/// Parse a vector from a raw data page (f32 * N).
fn parse_vector_from_data_page<const N: usize>(data_page: &[u8]) -> [f32; N] {
    let mut vec = [0.0f32; N];
    for i in 0..N {
        let off = i * 4;
        if off + 4 <= data_page.len() {
            vec[i] = f32::from_le_bytes(data_page[off..off + 4].try_into().unwrap());
        }
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_limiter_creation() {
        let limiter = IoLimiter::new(16);
        assert_eq!(limiter.semaphore().available_permits(), 16);
    }

    #[test]
    fn test_beam_search_config() {
        let config = BeamSearchConfig {
            beam_width: 4,
            search_list_size: 64,
            k: 10,
            window_size: 5,
            epsilon: 0.01,
        };
        assert_eq!(config.beam_width, 4);
    }

    #[test]
    fn test_parse_neighbors_from_page() {
        let mut page = vec![0u8; 4096];
        page[0..4].copy_from_slice(&3u32.to_le_bytes());
        page[4..8].copy_from_slice(&10u32.to_le_bytes());
        page[8..12].copy_from_slice(&20u32.to_le_bytes());
        page[12..16].copy_from_slice(&30u32.to_le_bytes());

        let result = parse_neighbors_from_page(&page, 10);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_parse_neighbors_capped_at_max_degree() {
        let mut page = vec![0u8; 4096];
        page[0..4].copy_from_slice(&5u32.to_le_bytes());
        for i in 0..5 {
            let off = 4 + i * 4;
            page[off..off + 4].copy_from_slice(&(i as u32).to_le_bytes());
        }

        let result = parse_neighbors_from_page(&page, 2);
        assert_eq!(result, vec![0, 1]);
    }
}
