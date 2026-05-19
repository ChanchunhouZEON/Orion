/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! StagedDiskANN + ADSampling runner.
//!
//! Builds the staged two-phase graph on pre-rotated vectors, then at search
//! time rotates the query with the same matrix and calls
//! `StagedDiskANN::search_adsampling` which reuses the τ/ee convergence loop
//! but with the scaled-partial early-abort distance kernel.

use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use adsampling::Rotator;
use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
use std::time::Instant;

pub struct StagedDiskANNAdsRunner {
    name: &'static str,
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    max_extra: usize,
    window_size: usize,
    ads_epsilon: f32,
    epsilon: f32,
    early_exit_limit: usize,
    dimension: usize,
    inner: Option<StagedInner>,
    rotator: RotatorKind,
}

enum StagedInner {
    Dim32 { staged: StagedDiskANN<32> },
    Dim100 { staged: StagedDiskANN<100> },
    Dim128 { staged: StagedDiskANN<128> },
    Dim960 { staged: StagedDiskANN<960> },
}

enum RotatorKind {
    None,
    Dim32(Box<Rotator<32>>),
    Dim100(Box<Rotator<100>>),
    Dim128(Box<Rotator<128>>),
    Dim960(Box<Rotator<960>>),
}

macro_rules! build_staged_ads {
    ($self:ident, $rotated:ident, $num_points:ident, $result:ident, $N:literal, $variant:ident) => {{
        drop($result.index);
        let t1 = Instant::now();
        let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
        let mut staged = StagedDiskANN::<$N>::new(
            empty_ds,
            &$result.partitions,
            $result.entry_point,
            $self.graph_degree as u32,
            $self.max_extra,
            None,
            None,
            None,
            false,
        );
        log::info!("Staged+ADS overhead: {:.2}s", t1.elapsed().as_secs_f32());

        let mut ds = diskann::model::InmemDataset::<f32, $N>::new($num_points, 1.0).unwrap();
        ds.data.memcpy(&$rotated[..$num_points * $N]).unwrap();
        staged.dataset = ds;

        // Auto-calibrate τ/ee using rotated warmup queries (same space as the graph).
        let calib_n = $num_points.min(200);
        let calib_queries: Vec<[f32; $N]> = (0..calib_n)
            .map(|i| {
                let mut q = [0.0f32; $N];
                q.copy_from_slice(&$rotated[i * $N..(i + 1) * $N]);
                q
            })
            .collect();
        if let Ok(calib) =
            staged.calibrate(&calib_queries, $self.search_list_size, $self.window_size)
        {
            $self.epsilon = calib.threshold;
            $self.early_exit_limit = calib.early_exit_limit;
        }

        $self.inner = Some(StagedInner::$variant { staged });
    }};
}

macro_rules! search_staged_ads {
    ($staged:ident, $rotated:ident, $k:ident, $self:ident, $N:literal) => {{
        let mut q = [0.0f32; $N];
        q.copy_from_slice(&$rotated[..$N]);
        $staged.search_adsampling(
            &q,
            $k,
            $self.search_list_size,
            $self.window_size,
            $self.epsilon,
            $self.early_exit_limit,
            $self.ads_epsilon,
        )
    }};
}

impl StagedDiskANNAdsRunner {
    pub fn set_search_list_size(&mut self, sls: usize) {
        self.search_list_size = sls;
    }

    pub fn new(
        name: &'static str,
        alpha: f32,
        graph_degree: usize,
        search_list_size: usize,
        max_extra: usize,
        window_size: usize,
        ads_epsilon: f32,
    ) -> Self {
        Self {
            name,
            alpha,
            graph_degree,
            search_list_size,
            max_extra,
            window_size,
            ads_epsilon,
            epsilon: 0.0,
            early_exit_limit: 0,
            dimension: 0,
            inner: None,
            rotator: RotatorKind::None,
        }
    }

    fn rotate_query(&self, query: &[f32], out: &mut Vec<f32>) {
        out.clear();
        match &self.rotator {
            RotatorKind::None => out.extend_from_slice(query),
            RotatorKind::Dim32(r) => {
                let mut a = [0.0f32; 32];
                a.copy_from_slice(&query[..32]);
                out.extend_from_slice(&r.apply(&a));
            }
            RotatorKind::Dim100(r) => {
                let mut a = [0.0f32; 100];
                a.copy_from_slice(&query[..100]);
                out.extend_from_slice(&r.apply(&a));
            }
            RotatorKind::Dim128(r) => {
                let mut a = [0.0f32; 128];
                a.copy_from_slice(&query[..128]);
                out.extend_from_slice(&r.apply(&a));
            }
            RotatorKind::Dim960(r) => {
                let mut a = [0.0f32; 960];
                a.copy_from_slice(&query[..960]);
                out.extend_from_slice(&r.apply(&a));
            }
        }
    }
}

impl AlgorithmRunner for StagedDiskANNAdsRunner {
    fn name(&self) -> &str {
        self.name
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        // One-shot dataset rotation.
        let mut rotated: Vec<f32> = data.to_vec();
        const ROTATION_SEED: u64 = 0xA05A_A05A;
        match dimension {
            DIM_32 => {
                let r = Box::new(Rotator::<32>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim32(r);
            }
            DIM_100 => {
                let r = Box::new(Rotator::<100>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim100(r);
            }
            DIM_128 => {
                let r = Box::new(Rotator::<128>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim128(r);
            }
            DIM_960 => {
                let r = Box::new(Rotator::<960>::new(ROTATION_SEED));
                r.apply_batch_inplace(&mut rotated);
                self.rotator = RotatorKind::Dim960(r);
            }
            d => panic!("Unsupported dimension for Staged+ADS: {d}"),
        }

        let result = build_diskann_index(
            &rotated,
            num_points,
            dimension,
            self.alpha,
            self.graph_degree as u32,
            self.search_list_size as u32,
            false,
            None,
            None,
            true,
            self.max_extra,
        )
        .expect("build failed");
        let graph_build_time = result.graph_build_time;

        match dimension {
            DIM_32 => build_staged_ads!(self, rotated, num_points, result, 32, Dim32),
            DIM_100 => build_staged_ads!(self, rotated, num_points, result, 100, Dim100),
            DIM_128 => build_staged_ads!(self, rotated, num_points, result, 128, Dim128),
            DIM_960 => build_staged_ads!(self, rotated, num_points, result, 960, Dim960),
            _ => panic!("Unsupported dimension: {dimension}"),
        }

        let total = start.elapsed();
        BuildTiming {
            graph_build: graph_build_time,
            overhead: total - graph_build_time,
        }
    }

    fn search(&self, query: &[f32], k: usize) -> SearchResult {
        let mut rotated = Vec::with_capacity(self.dimension);
        self.rotate_query(query, &mut rotated);
        let start = Instant::now();
        let neighbors = match self.inner.as_ref().expect("Index not built") {
            StagedInner::Dim32 { staged } => search_staged_ads!(staged, rotated, k, self, 32),
            StagedInner::Dim100 { staged } => search_staged_ads!(staged, rotated, k, self, 100),
            StagedInner::Dim128 { staged } => search_staged_ads!(staged, rotated, k, self, 128),
            StagedInner::Dim960 { staged } => search_staged_ads!(staged, rotated, k, self, 960),
        }
        .expect("Staged+ADS search failed");
        SearchResult {
            neighbors,
            duration: start.elapsed(),
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}
