/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::runner::common::{AlgorithmRunner, SearchResult};
use ndarray::Array2;
use staged_diskann::{build_diskann_index, ClusteringMethod, StagedDiskANN, DIM_128, DIM_960};
use std::time::{Duration, Instant};

/// Benchmark runner for Staged DiskANN with compile-time dimension dispatch.
pub struct StagedDiskANNRunner {
    // DiskANN build params
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    // Staged params
    max_cluster_point_size: usize,
    max_connection_clusters: usize,
    max_connection_per_cluster: usize,
    critical_minimum_rate: f32,
    // PQ params
    n_subquantizers: usize,
    n_bits: u32,
    // Search params
    window_size: usize,
    epsilon: f32,
    // Clustering method
    clustering_method: ClusteringMethod,
    // State
    dimension: usize,
    inner: Option<StagedInner>,
}

enum StagedInner {
    Dim128 { compressed: StagedDiskANN<128> },
    Dim960 { compressed: StagedDiskANN<960> },
}

impl StagedDiskANNRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        alpha: f32,
        graph_degree: usize,
        search_list_size: usize,
        max_cluster_point_size: usize,
        max_connection_clusters: usize,
        max_connection_per_cluster: usize,
        critical_minimum_rate: f32,
        n_subquantizers: usize,
        n_bits: u32,
        window_size: usize,
        epsilon: f32,
        clustering_method: ClusteringMethod,
    ) -> Self {
        Self {
            alpha,
            graph_degree,
            search_list_size,
            max_cluster_point_size,
            max_connection_clusters,
            max_connection_per_cluster,
            critical_minimum_rate,
            n_subquantizers,
            n_bits,
            window_size,
            epsilon,
            clustering_method,
            dimension: 0,
            inner: None,
        }
    }
}

impl AlgorithmRunner for StagedDiskANNRunner {
    fn name(&self) -> &str {
        "StagedDiskANN"
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> Duration {
        self.dimension = dimension;
        let start = Instant::now();

        let data_2d = Array2::from_shape_vec((num_points, dimension), data.to_vec())
            .expect("Failed to reshape data")
            .to_shared();

        let result = build_diskann_index(
            &data_2d,
            self.alpha,
            self.graph_degree as u32,
            self.search_list_size as u32,
            false,
            None,
            None,
            true, // compute candidate sets
        );
        log::info!(
            "DiskANN graph build (parallel Vamana + candidate sets): {:.2}s",
            result.graph_build_time.as_secs_f32()
        );
        // log::info!("PQ build: {:.2}s", result.pq_build_time.as_secs_f32());

        match dimension {
            DIM_128 => {
                let t1 = Instant::now();
                let compressed = StagedDiskANN::<128>::new(
                    data_2d,
                    result.graph,
                    result.candidate_sets,
                    result.entry_point,
                    None,
                    None,
                    self.max_cluster_point_size,
                    self.max_connection_clusters,
                    self.max_connection_per_cluster,
                    self.critical_minimum_rate,
                    None,
                    self.clustering_method,
                    true,
                );
                let compressed_time = t1.elapsed();
                log::info!(
                    "StagedDiskANN overhead: {:.2}s",
                    compressed_time.as_secs_f32()
                );

                self.inner = Some(StagedInner::Dim128 { compressed });
            }
            DIM_960 => {
                let t1 = Instant::now();
                let compressed = StagedDiskANN::<960>::new(
                    data_2d,
                    result.graph,
                    result.candidate_sets,
                    result.entry_point,
                    None,
                    None,
                    self.max_cluster_point_size,
                    self.max_connection_clusters,
                    self.max_connection_per_cluster,
                    self.critical_minimum_rate,
                    None,
                    self.clustering_method,
                    true,
                );
                let compressed_time = t1.elapsed();
                log::info!(
                    "StagedDiskANN overhead: {:.2}s",
                    compressed_time.as_secs_f32()
                );

                self.inner = Some(StagedInner::Dim960 { compressed });
            }
            _ => panic!("Unsupported dimension: {dimension}"),
        }

        start.elapsed()
    }

    fn search(&self, query: &[f32], k: usize) -> SearchResult {
        let start = Instant::now();
        let neighbors = match self.inner.as_ref().expect("Index not built") {
            StagedInner::Dim128 { compressed, .. } => {
                let mut q = [0.0f32; 128];
                q.copy_from_slice(&query[..128]);
                compressed.search(&q, k, self.search_list_size, self.window_size, self.epsilon)
            }
            StagedInner::Dim960 { compressed, .. } => {
                let mut q = [0.0f32; 960];
                q.copy_from_slice(&query[..960]);
                compressed.search(&q, k, self.search_list_size, self.window_size, self.epsilon)
            }
        }
        .expect("Searching process failed");
        let duration = start.elapsed();
        SearchResult {
            neighbors,
            duration,
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}
