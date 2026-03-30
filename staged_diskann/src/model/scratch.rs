/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crossbeam::queue::ArrayQueue;

use crate::algorithm::search::convergence::DistanceConvergenceChecker;
use crate::model::NeighborPriorityQueue;

/// Pre-allocated scratch space for compressed-diskann search queries.
pub struct CompressedSearchScratch {
    pub visited: HashSet<u32>,
    pub neighbor_pq: NeighborPriorityQueue,
    pub convergence_checker: DistanceConvergenceChecker,
    pub compressed_neighbors_in_mem: HashSet<u32>,
    pub rerank_buffer: Vec<(u32, f32)>,
    /// Cached neighbors-of-neighbors for post-convergence Phase 2.
    pub neighbor_cache: HashMap<u32, Vec<u32>>,
}

impl CompressedSearchScratch {
    pub fn new(
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        estimated_nodes: usize,
    ) -> Self {
        Self {
            visited: HashSet::with_capacity(estimated_nodes),
            neighbor_pq: NeighborPriorityQueue::with_capacity(search_list_size),
            convergence_checker: DistanceConvergenceChecker::new(window_size, epsilon),
            compressed_neighbors_in_mem: HashSet::with_capacity(estimated_nodes),
            rerank_buffer: Vec::with_capacity(estimated_nodes),
            neighbor_cache: HashMap::with_capacity(estimated_nodes),
        }
    }

    pub fn clear(&mut self) {
        self.visited.clear();
        self.neighbor_pq.clear();
        self.convergence_checker.reset();
        self.compressed_neighbors_in_mem.clear();
        self.rerank_buffer.clear();
        self.neighbor_cache.clear();
    }
}

/// Pool of pre-allocated scratch spaces for concurrent search.
pub struct ScratchPool {
    pool: Arc<ArrayQueue<Box<CompressedSearchScratch>>>,
}

impl ScratchPool {
    pub fn new(
        num_threads: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        estimated_nodes: usize,
    ) -> Self {
        let pool = Arc::new(ArrayQueue::new(num_threads));
        for _ in 0..num_threads {
            let scratch = Box::new(CompressedSearchScratch::new(
                search_list_size,
                window_size,
                epsilon,
                estimated_nodes,
            ));
            pool.push(scratch).ok();
        }
        Self { pool }
    }

    pub fn acquire(&self) -> ScratchGuard {
        loop {
            if let Some(scratch) = self.pool.pop() {
                return ScratchGuard {
                    scratch: Some(scratch),
                    pool: self.pool.clone(),
                };
            }
            std::hint::spin_loop();
        }
    }
}

/// RAII guard that returns the scratch to the pool on drop.
pub struct ScratchGuard {
    scratch: Option<Box<CompressedSearchScratch>>,
    pool: Arc<ArrayQueue<Box<CompressedSearchScratch>>>,
}

impl ScratchGuard {
    pub fn scratch(&mut self) -> &mut CompressedSearchScratch {
        self.scratch.as_deref_mut().unwrap()
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        if let Some(mut scratch) = self.scratch.take() {
            scratch.clear();
            self.pool.push(scratch).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scratch_clear() {
        let mut scratch = CompressedSearchScratch::new(10, 5, 0.01, 100);
        scratch.visited.insert(1);
        scratch.visited.insert(2);
        scratch.compressed_neighbors_in_mem.insert(3);
        scratch.rerank_buffer.push((0, 1.0));
        scratch.neighbor_cache.insert(1, vec![2, 3, 4]);

        scratch.clear();
        assert!(scratch.visited.is_empty());
        assert!(scratch.compressed_neighbors_in_mem.is_empty());
        assert!(scratch.rerank_buffer.is_empty());
        assert_eq!(scratch.neighbor_pq.size(), 0);
        assert!(scratch.neighbor_cache.is_empty());
    }

    #[test]
    fn test_pool_acquire_release() {
        let pool = ScratchPool::new(2, 10, 5, 0.01, 100);
        let mut g1 = pool.acquire();
        let mut g2 = pool.acquire();
        g1.scratch().visited.insert(1);
        g2.scratch().visited.insert(2);
        drop(g1);
        drop(g2);

        let g3 = pool.acquire();
        assert!(g3.scratch.as_ref().unwrap().visited.is_empty());
    }
}
