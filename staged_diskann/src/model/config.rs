/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Configuration for Compressed DiskANN.
#[derive(Debug, Clone)]
pub struct CompressedConfig {
    /// Alpha parameter for robust pruning (occlusion factor).
    pub alpha: f32,
    /// Maximum out-degree of each node in the base graph.
    pub graph_degree: usize,
    /// Search list size for greedy search during index construction.
    pub search_list_size: usize,
    /// Maximum number of points per cohesive cluster (M in the report).
    pub max_cluster_point_size: usize,
    /// Maximum number of external clusters a node can connect to (m).
    pub max_connection_clusters: usize,
    /// Maximum edges per external cluster (n).
    pub max_connection_per_cluster: usize,
    /// Critical minimum rate (τ) for cluster eviction.
    pub critical_minimum_rate: f32,
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
            max_cluster_point_size: 8,
            max_connection_clusters: 4,
            max_connection_per_cluster: 2,
            critical_minimum_rate: 0.7,
            use_pq: true,
            n_subquantizers: 8,
            n_bits: 8,
            is_save: true,
        }
    }
}
