/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use diskann::index::InmemIndex;
use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128, DIM_960};
use std::time::{Duration, Instant};

/// Benchmark runner for Staged DiskANN with compile-time dimension dispatch.
pub struct StagedDiskANNRunner {
    // Runner identity
    name: &'static str,
    // DiskANN build params
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    // Staged params
    max_cluster_point_size: usize,
    max_connection_clusters: usize,
    max_connection_per_cluster: usize,
    critical_minimum_rate: f32,
    // Search params
    window_size: usize,
    epsilon: f32,
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
        name: &'static str,
        alpha: f32,
        graph_degree: usize,
        search_list_size: usize,
        max_cluster_point_size: usize,
        max_connection_clusters: usize,
        max_connection_per_cluster: usize,
        critical_minimum_rate: f32,
        window_size: usize,
        epsilon: f32,
    ) -> Self {
        Self {
            name,
            alpha,
            graph_degree,
            search_list_size,
            max_cluster_point_size,
            max_connection_clusters,
            max_connection_per_cluster,
            critical_minimum_rate,
            window_size,
            epsilon,
            dimension: 0,
            inner: None,
        }
    }

}

impl AlgorithmRunner for StagedDiskANNRunner {
    fn name(&self) -> &str {
        self.name
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        let mut result = build_diskann_index(
            data,
            num_points,
            dimension,
            self.alpha,
            self.graph_degree as u32,
            self.search_list_size as u32,
            false,
            None,
            None,
            true, // compute candidate sets
        )
        .expect("build failed");
        log::info!(
            "DiskANN graph build (parallel Vamana + candidate sets): {:.2}s",
            result.graph_build_time.as_secs_f32()
        );
        log::info!("  mem after build_diskann_index: {}", crate::metrics::memory::format_bytes(crate::ALLOCATOR.current_bytes()));

        match dimension {
            DIM_128 => {
                // Downcast to take the InmemDataset from the InmemIndex.
                let dataset = {
                    let idx = result.index.as_any_mut()
                        .downcast_mut::<InmemIndex<f32, 128>>()
                        .expect("downcast to InmemIndex<f32, 128>");
                    std::mem::replace(
                        &mut idx.dataset,
                        diskann::model::InmemDataset::new(0, 1.0).unwrap(),
                    )
                };
                log::info!("  mem after take(dataset):  {}", crate::metrics::memory::format_bytes(crate::ALLOCATOR.current_bytes()));
                drop(result.index);
                log::info!("  mem after drop(index):    {}", crate::metrics::memory::format_bytes(crate::ALLOCATOR.current_bytes()));

                let t1 = Instant::now();
                let compressed = StagedDiskANN::<128>::new(
                    dataset,
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
                    false,
                );
                let compressed_time = t1.elapsed();
                log::info!(
                    "StagedDiskANN overhead: {:.2}s",
                    compressed_time.as_secs_f32()
                );

                self.inner = Some(StagedInner::Dim128 { compressed });
            }
            DIM_960 => {
                let dataset = {
                    let idx = result.index.as_any_mut()
                        .downcast_mut::<InmemIndex<f32, 960>>()
                        .expect("downcast to InmemIndex<f32, 960>");
                    std::mem::replace(
                        &mut idx.dataset,
                        diskann::model::InmemDataset::new(0, 1.0).unwrap(),
                    )
                };
                drop(result.index);

                let t1 = Instant::now();
                let compressed = StagedDiskANN::<960>::new(
                    dataset,
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
                    false,
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

        let total = start.elapsed();
        BuildTiming {
            graph_build: result.graph_build_time,
            overhead: total - result.graph_build_time,
        }
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

    fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Vec<SearchResult> {
        use std::time::Instant;

        let start = Instant::now();
        let batch_results = match self.inner.as_ref().expect("Index not built") {
            StagedInner::Dim128 { compressed, .. } => {
                let qs: Vec<[f32; 128]> = queries
                    .iter()
                    .map(|q| {
                        let mut arr = [0.0f32; 128];
                        arr.copy_from_slice(&q[..128]);
                        arr
                    })
                    .collect();
                compressed
                    .search_batch(&qs, k, self.search_list_size, self.window_size, self.epsilon)
                    .expect("batch search failed")
            }
            StagedInner::Dim960 { compressed, .. } => {
                let qs: Vec<[f32; 960]> = queries
                    .iter()
                    .map(|q| {
                        let mut arr = [0.0f32; 960];
                        arr.copy_from_slice(&q[..960]);
                        arr
                    })
                    .collect();
                compressed
                    .search_batch(&qs, k, self.search_list_size, self.window_size, self.epsilon)
                    .expect("batch search failed")
            }
        };
        let total = start.elapsed();
        let per_query = total / queries.len().max(1) as u32;

        batch_results
            .into_iter()
            .map(|neighbors| SearchResult {
                neighbors,
                duration: per_query,
            })
            .collect()
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}
