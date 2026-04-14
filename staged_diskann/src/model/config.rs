/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Configuration for Staged DiskANN.
#[derive(Debug, Clone)]
pub struct CompressedConfig {
    /// Alpha parameter for robust pruning (occlusion factor).
    pub alpha: f32,
    /// Maximum out-degree of each node in the base graph.
    pub graph_degree: usize,
    /// Search list size for greedy search during index construction.
    pub search_list_size: usize,
    /// Base local count (t) — minimum neighbors promoted to local zone.
    pub base_local_count: usize,
    /// Whether to use product quantization.
    pub use_pq: bool,
    /// Number of PQ sub-quantizers.
    pub n_subquantizers: usize,
    /// Number of bits per PQ code.
    pub n_bits: u32,
    /// Whether to save index artifacts to disk.
    pub is_save: bool,
}

impl Default for CompressedConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            graph_degree: 32,
            search_list_size: 48,
            base_local_count: 16,
            use_pq: true,
            n_subquantizers: 8,
            n_bits: 8,
            is_save: true,
        }
    }
}
