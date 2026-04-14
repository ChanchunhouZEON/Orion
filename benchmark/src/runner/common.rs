/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Results from a single search query.
pub struct SearchResult {
    pub neighbors: Vec<u32>,
    pub duration: Duration,
}

/// Trait for all ANNS algorithm runners in the benchmark.
pub trait AlgorithmRunner: Sync {
    /// Algorithm name for display.
    fn name(&self) -> &str;

    /// Build the index from base data. Returns timing breakdown.
    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming;

    /// Search for k nearest neighbors of a single query vector.
    fn search(&self, query: &[f32], k: usize) -> SearchResult;

    /// Batch search for multiple queries. Default uses rayon parallel.
    fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Vec<SearchResult> {
        use rayon::prelude::*;
        queries.par_iter().map(|q| self.search(q, k)).collect()
    }

    /// Batch search with a dedicated thread pool of given size.
    fn search_batch_with_threads(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        num_threads: usize,
    ) -> Vec<SearchResult> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build search thread pool");
        pool.install(|| self.search_batch(queries, k))
    }

    /// Approximate memory usage in bytes.
    #[allow(dead_code)]
    fn memory_bytes(&self) -> usize;

    /// Whether this runner supports mmap-based search.
    fn supports_mmap(&self) -> bool {
        false
    }

    /// Save the built index to an mmap-compatible file. Returns the file path.
    fn save_mmap(&self, _dir: &Path) -> anyhow::Result<PathBuf> {
        anyhow::bail!("mmap save not supported")
    }

    /// Switch to mmap-backed search mode. After this, `search()` reads from mmap.
    fn enable_mmap_search(&mut self, _graph_path: &Path) -> anyhow::Result<()> {
        anyhow::bail!("mmap search not supported")
    }

    /// Warm the OS page cache by prefetching the entry point neighborhood.
    fn warm_cache(&self, _max_hops: usize) {}

    /// Release in-memory vectors and graph after mmap search is enabled.
    /// Only `search()` via mmap remains functional after this call.
    fn drop_inmem_vectors(&mut self) {}
}
