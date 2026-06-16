/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use std::time::Duration;

/// Results from a single search query.
pub struct SearchResult {
    pub neighbors: Vec<u32>,
    #[allow(dead_code)]
    pub duration: Duration,
}

/// Trait for all ANNS algorithm runners in the benchmark.
pub trait AlgorithmRunner: Sync {
    /// Algorithm name for display.
    #[allow(dead_code)]
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
}
