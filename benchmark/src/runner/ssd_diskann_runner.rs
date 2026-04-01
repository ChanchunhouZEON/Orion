/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::runner::common::{AlgorithmRunner, SearchResult};
use diskann::model::InMemoryGraph;
use ndarray::Array2;
use ssd_diskann::SSDIndex;
use staged_diskann::{CsrGraph, build_diskann_index, DIM_128, DIM_960};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Convert CsrGraph to InMemoryGraph for SSD index build.
fn csr_to_inmem_graph(csr: &CsrGraph) -> InMemoryGraph {
    let n = csr.num_nodes();
    let max_deg = (0..n).map(|i| csr.degree(i)).max().unwrap_or(0) as u32;
    let g = InMemoryGraph::new(n, max_deg);
    for i in 0..n {
        g.set_neighbors_from_vec(i as u32, csr.neighbors(i).to_vec()).ok();
    }
    g
}

/// Benchmark runner for SSD-based DiskANN.
/// Keeps only PQ codes in memory; full vectors are read from disk.
pub struct SSDDiskANNRunner {
    // Build params
    beam_width: usize,
    search_list_size: usize,
    graph_degree: u32,
    alpha: f32,
    n_pq_chunks: usize,
    n_bits: u32,
    // State
    dimension: usize,
    inner: Option<SSDInner>,
    disk_path: Option<PathBuf>,
}

enum SSDInner {
    Dim128 { index: SSDIndex<128> },
    Dim960 { index: SSDIndex<960> },
}

impl SSDDiskANNRunner {
    pub fn new(
        beam_width: usize,
        search_list_size: usize,
        graph_degree: u32,
        alpha: f32,
        n_pq_chunks: usize,
        n_bits: u32,
    ) -> Self {
        Self {
            beam_width,
            search_list_size,
            graph_degree,
            alpha,
            n_pq_chunks,
            n_bits,
            dimension: 0,
            inner: None,
            disk_path: None,
        }
    }
}

impl AlgorithmRunner for SSDDiskANNRunner {
    fn name(&self) -> &str {
        "SSD-DiskANN"
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> Duration {
        self.dimension = dimension;
        let start = Instant::now();

        let data_2d = Array2::from_shape_vec((num_points, dimension), data.to_vec())
            .expect("Failed to reshape data")
            .to_shared();

        // Build DiskANN graph + PQ
        let result = build_diskann_index(
            &data_2d,
            self.alpha,
            self.graph_degree,
            self.search_list_size as u32,
            true,
            Some(self.n_pq_chunks),
            Some(self.n_bits),
            false, // no need for candidate sets
        );
        log::info!(
            "DiskANN graph build: {:.2}s, PQ build: {:.2}s",
            result.graph_build_time.as_secs_f32(),
            result.pq_build_time.as_secs_f32()
        );

        let pq = result.pq.expect("PQ should be built");
        let pq_codes = result.pq_codes.expect("PQ codes should be built");

        // Write disk index and create SSD index
        let disk_dir = PathBuf::from("ssd_diskann_graphs");
        std::fs::create_dir_all(&disk_dir).expect("create disk dir");
        let disk_path = disk_dir.join("ssd_index.bin");
        self.disk_path = Some(disk_path.clone());

        match dimension {
            DIM_128 => {
                let index = SSDIndex::<128>::build(
                    &data_2d,
                    &csr_to_inmem_graph(&result.graph),
                    result.entry_point,
                    pq,
                    pq_codes,
                    &disk_path,
                    self.beam_width,
                    self.search_list_size,
                )
                .expect("Failed to build SSD index");

                self.inner = Some(SSDInner::Dim128 { index });
            }
            DIM_960 => {
                let index = SSDIndex::<960>::build(
                    &data_2d,
                    &csr_to_inmem_graph(&result.graph),
                    result.entry_point,
                    pq,
                    pq_codes,
                    &disk_path,
                    self.beam_width,
                    self.search_list_size,
                )
                .expect("Failed to build SSD index");

                self.inner = Some(SSDInner::Dim960 { index });
            }
            _ => panic!("Unsupported dimension: {dimension}"),
        }

        // After build, data_2d, result.graph are dropped — only PQ codes remain in memory
        start.elapsed()
    }

    fn search(&self, query: &[f32], k: usize) -> SearchResult {
        let start = Instant::now();
        let neighbors = match self.inner.as_ref().expect("Index not built") {
            SSDInner::Dim128 { index } => {
                let mut q = [0.0f32; 128];
                q.copy_from_slice(&query[..128]);
                index.search(&q, k)
            }
            SSDInner::Dim960 { index } => {
                let mut q = [0.0f32; 960];
                q.copy_from_slice(&query[..960]);
                index.search(&q, k)
            }
        };
        let duration = start.elapsed();
        SearchResult {
            neighbors,
            duration,
        }
    }

    fn memory_bytes(&self) -> usize {
        match self.inner.as_ref() {
            Some(SSDInner::Dim128 { index }) => index.memory_bytes(),
            Some(SSDInner::Dim960 { index }) => index.memory_bytes(),
            None => 0,
        }
    }

    fn warm_cache(&self, max_hops: usize) {
        match self.inner.as_ref() {
            Some(SSDInner::Dim128 { index }) => index.warm_cache(max_hops),
            Some(SSDInner::Dim960 { index }) => index.warm_cache(max_hops),
            None => {}
        }
    }
}
