pub mod fvecs;
pub mod random;

pub use fvecs::{read_fvecs, read_fvecs_n, read_ivecs};
pub use random::generate_random_dataset;

use rayon::prelude::*;

/// A loaded dataset with base vectors, query vectors, and ground truth.
pub struct Dataset {
    #[allow(dead_code)]
    pub name: String,
    pub dimension: usize,
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<u32>>,
}

impl Dataset {
    pub fn num_base(&self) -> usize {
        self.base.len()
    }

    pub fn num_queries(&self) -> usize {
        self.queries.len()
    }

    /// Get base vectors as a flat f32 slice (row-major).
    pub fn base_flat(&self) -> Vec<f32> {
        self.base.iter().flatten().copied().collect()
    }

    /// Get query vectors as a flat f32 slice (row-major).
    #[allow(dead_code)]
    pub fn queries_flat(&self) -> Vec<f32> {
        self.queries.iter().flatten().copied().collect()
    }

    /// Recompute ground truth via brute-force L2 k-NN against the current base set.
    /// Use this when the base has been truncated and the original ground truth is invalid.
    pub fn recompute_ground_truth(&mut self, k: usize) {
        log::info!(
            "Recomputing ground truth (k={}) for {} queries against {} base points...",
            k,
            self.queries.len(),
            self.base.len()
        );
        self.ground_truth = self
            .queries
            .par_iter()
            .map(|query| random::brute_force_knn(query, &self.base, k))
            .collect();
        log::info!("Ground truth recomputed successfully");
    }
}
