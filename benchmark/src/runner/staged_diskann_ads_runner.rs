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
use std::path::PathBuf;
use std::time::Instant;

pub struct StagedDiskANNAdsRunner {
    name: &'static str,
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    max_extra: usize,
    window_size: usize,
    epsilon: f32,
    early_exit_limit: usize,
    dimension: usize,
    inner: Option<StagedInner>,
    rotator: RotatorKind,
    /// Optional cache base path. When set and `<cache>.bin` +
    /// `<cache>.pgraph` both exist on disk, `build()` skips the
    /// Vamana construction entirely and rehydrates the rotated dataset
    /// from `data` via the same in-memory rotation. The rotation seed
    /// is fixed (see `ROTATION_SEED` below) so the rotated dataset is
    /// reproducible from the same input bytes.
    cache_path: Option<PathBuf>,
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
        // Pass `Some(cache_path)` + save=true on cache-enabled runs so
        // the PhasedGraph slab is written next to the runner's bin
        // path. The next invocation will land on the cache-hit
        // `load_from_cache` arm in `build()` and skip the Vamana build
        // entirely.
        let (cache_arg, save_arg) = match &$self.cache_path {
            Some(p) => (Some(p.clone()), true),
            None => (None, false),
        };
        let mut staged = StagedDiskANN::<$N>::new(
            empty_ds,
            &$result.partitions,
            $result.entry_point,
            $self.graph_degree as u32,
            $self.max_extra,
            None,
            None,
            cache_arg,
            save_arg,
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
        // Cascade dispatch via the unified pipeline:
        //   prefilter = none, admission = AdsF32, rerank = f32 truth.
        // Replaces the standalone `search_adsampling` codepath — ADS
        // is now a first-class admission tier and reuses every search-
        // loop optimisation in `search_unified` (peeled hops, 3-way
        // merge cadence, two-slice rerank, sharded counters, ...).
        let ad = crate::runner::cascade::build_admission::<$N>(
            $staged,
            crate::runner::cascade::AdmissionChoice::AdsF32,
        );
        let rr = crate::runner::cascade::build_rerank::<$N>(
            $staged,
            crate::runner::cascade::RerankChoice::F32,
        );
        $staged.search_unified::<staged_diskann::algorithm::search::stage::NoPrefilter, _, _>(
            &q,
            $k,
            $self.search_list_size,
            $self.window_size,
            $self.epsilon,
            $self.early_exit_limit,
            None,
            ad.as_ref(),
            rr.as_ref(),
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
    ) -> Self {
        Self {
            name,
            alpha,
            graph_degree,
            search_list_size,
            max_extra,
            window_size,
            epsilon: 0.0,
            early_exit_limit: 0,
            dimension: 0,
            inner: None,
            rotator: RotatorKind::None,
            cache_path: None,
        }
    }

    pub fn set_cache_path<P: Into<PathBuf>>(&mut self, p: P) {
        self.cache_path = Some(p.into());
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

        // One-shot dataset rotation — deterministic (fixed seed) so
        // the same input bytes produce the same rotated vectors,
        // independent of cache presence. Rotation is cheap (~1s on
        // 1M D=128); we always run it so the rotator handle is
        // populated for query-time rotation.
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

        // Cache fast-path. Mirrors `StagedDiskANNRunner::build`:
        // if both `<cache>.bin` and `<cache>.pgraph` exist on disk,
        // load the PhasedGraph from cache and skip the in-process
        // Vamana build entirely. We still re-populate the rotated
        // dataset on the fly from `rotated` since the cache only
        // stores the graph topology.
        if let Some(cache) = self.cache_path.clone() {
            let pgraph_path = cache.with_extension("pgraph");
            if cache.exists() && pgraph_path.exists() {
                log::info!("Loading cached Staged+ADS from {:?}", cache);
                macro_rules! load_cached_ads {
                    ($N:literal, $variant:ident) => {{
                        let empty_ds =
                            diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                        let mut staged = StagedDiskANN::<$N>::load_from_cache(&cache, empty_ds)
                            .expect("StagedDiskANN::load_from_cache failed");
                        let mut ds =
                            diskann::model::InmemDataset::<f32, $N>::new(num_points, 1.0).unwrap();
                        ds.data.memcpy(&rotated[..num_points * $N]).unwrap();
                        staged.dataset = ds;

                        // Re-calibrate from rotated warmup queries —
                        // the cached PhasedGraph carries no calibration
                        // state since (threshold, ee) depend on the
                        // search-time L which `set_search_list_size`
                        // can change later.
                        let calib_n = num_points.min(200);
                        let calib_queries: Vec<[f32; $N]> = (0..calib_n)
                            .map(|i| {
                                let mut q = [0.0f32; $N];
                                q.copy_from_slice(&rotated[i * $N..(i + 1) * $N]);
                                q
                            })
                            .collect();
                        if let Ok(calib) = staged.calibrate(
                            &calib_queries,
                            self.search_list_size,
                            self.window_size,
                        ) {
                            self.epsilon = calib.threshold;
                            self.early_exit_limit = calib.early_exit_limit;
                        }

                        self.inner = Some(StagedInner::$variant { staged });
                    }};
                }
                match dimension {
                    DIM_32 => load_cached_ads!(32, Dim32),
                    DIM_100 => load_cached_ads!(100, Dim100),
                    DIM_128 => load_cached_ads!(128, Dim128),
                    DIM_960 => load_cached_ads!(960, Dim960),
                    _ => panic!("Unsupported dimension: {dimension}"),
                }
                let elapsed = start.elapsed();
                return BuildTiming {
                    graph_build: elapsed,
                    overhead: std::time::Duration::ZERO,
                };
            }
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
}
