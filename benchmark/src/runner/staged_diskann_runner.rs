/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
use std::time::Instant;

/// Benchmark runner for Staged DiskANN with compile-time dimension dispatch.
pub struct StagedDiskANNRunner {
    name: &'static str,
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    key_neighbor_count: usize,
    base_local_count: usize,
    max_extra: usize,
    window_size: usize,
    epsilon: f32,
    early_exit_limit: usize,
    dimension: usize,
    inner: Option<StagedInner>,
}

enum StagedInner {
    Dim32 { staged: StagedDiskANN<32> },
    Dim100 { staged: StagedDiskANN<100> },
    Dim128 { staged: StagedDiskANN<128> },
    Dim960 { staged: StagedDiskANN<960> },
}

macro_rules! build_staged {
    ($self:ident, $data:ident, $num_points:ident, $result:ident, $N:literal, $variant:ident) => {{
        drop($result.index);
        log::info!(
            "  mem after drop(index):    {}",
            crate::metrics::memory::format_bytes(crate::ALLOCATOR.current_bytes())
        );

        let t1 = Instant::now();
        let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
        let mut staged = StagedDiskANN::<$N>::new(
            empty_ds,
            $result.graph,
            &$result.candidate_sets,
            $result.entry_point,
            $self.base_local_count,
            $self.max_extra,
            None,
            None,
            None,
            false,
        );
        log::info!("StagedDiskANN overhead: {:.2}s", t1.elapsed().as_secs_f32());

        let mut ds = diskann::model::InmemDataset::<f32, $N>::new($num_points, 1.0).unwrap();
        ds.data.memcpy(&$data[..$num_points * $N]).unwrap();
        staged.dataset = ds;

        $self.inner = Some(StagedInner::$variant { staged });
    }};
}

macro_rules! search_staged {
    ($staged:ident, $query:ident, $k:ident, $self:ident, $N:literal) => {{
        let mut q = [0.0f32; $N];
        q.copy_from_slice(&$query[..$N]);
        $staged.search(
            &q,
            $k,
            $self.search_list_size,
            $self.window_size,
            $self.epsilon,
            $self.early_exit_limit,
        )
    }};
}

impl StagedDiskANNRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &'static str,
        alpha: f32,
        graph_degree: usize,
        search_list_size: usize,
        key_neighbor_count: usize,
        base_local_count: usize,
        max_extra: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
    ) -> Self {
        Self {
            name,
            alpha,
            graph_degree,
            search_list_size,
            key_neighbor_count,
            base_local_count,
            max_extra,
            window_size,
            epsilon,
            early_exit_limit,
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

        let result = build_diskann_index(
            data,
            num_points,
            dimension,
            self.alpha,
            self.graph_degree as u32,
            self.search_list_size as u32,
            false,
            None,
            None,
            true,
            self.key_neighbor_count,
        )
        .expect("build failed");
        log::info!(
            "DiskANN graph build (parallel Vamana + candidate sets): {:.2}s",
            result.graph_build_time.as_secs_f32()
        );

        match dimension {
            DIM_32 => build_staged!(self, data, num_points, result, 32, Dim32),
            DIM_100 => build_staged!(self, data, num_points, result, 100, Dim100),
            DIM_128 => build_staged!(self, data, num_points, result, 128, Dim128),
            DIM_960 => build_staged!(self, data, num_points, result, 960, Dim960),
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
            StagedInner::Dim32 { staged } => search_staged!(staged, query, k, self, 32),
            StagedInner::Dim100 { staged } => search_staged!(staged, query, k, self, 100),
            StagedInner::Dim128 { staged } => search_staged!(staged, query, k, self, 128),
            StagedInner::Dim960 { staged } => search_staged!(staged, query, k, self, 960),
        }
        .expect("Searching process failed");
        SearchResult {
            neighbors,
            duration: start.elapsed(),
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}
