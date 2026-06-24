/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use crate::runner::cascade::{AdmissionChoice, PrefilterChoice, RerankChoice};
use crate::runner::common::{AlgorithmRunner, SearchResult};
use orion::{
    build_diskann_index, Orion, DIM_100, DIM_128, DIM_1536, DIM_32, DIM_768, DIM_784,
    DIM_960,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Benchmark runner for Orion with compile-time dimension dispatch
/// and runtime metric dispatch (L2 / L2-Q / MIPS / MIPS-Q).
///
/// Both `search` (single query) and `search_batch` (multi-query) route to the
/// metric's dedicated kernel — the latter calls
/// `Orion::search_batch_*` which has the L-adaptive `par_chunks(BATCH)`
/// rayon shape, instead of the trait's default per-query `par_iter`.
pub struct OrionRunner {
    name: &'static str,
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    max_extra: usize,
    window_size: usize,
    /// Cascade triple — drives both single-query `search` and
    /// multi-query `search_batch` via the unified pipeline. Mirrors
    /// the `Cascade::default_for_dataset` choice from
    /// `orion.rs`. Loaded by callers from `sweep.yaml`'s
    /// per-dataset `orion.{prefilter, admission, rerank}` fields.
    prefilter: crate::runner::cascade::PrefilterChoice,
    admission: crate::runner::cascade::AdmissionChoice,
    rerank: crate::runner::cascade::RerankChoice,
    /// Auto-calibrated during build.
    epsilon: f32,
    /// Auto-calibrated during build.
    early_exit_limit: usize,
    dimension: usize,
    inner: Option<OrionInner>,
    /// Optional cache path. When set and `<cache>.bin` + `<cache>.pgraph`
    /// both exist, `build()` loads from disk and skips the in-process
    /// Vamana build entirely — mirrors `orion`'s `load_from_cache`
    /// arm so thread-sweep and orion run on identical graphs.
    cache_path: Option<PathBuf>,
}

enum OrionInner {
    Dim32 { idx: Orion<32> },
    Dim100 { idx: Orion<100> },
    Dim128 { idx: Orion<128> },
    Dim768 { idx: Orion<768> },
    Dim784 { idx: Orion<784> },
    Dim960 { idx: Orion<960> },
    Dim1536 { idx: Orion<1536> },
}

macro_rules! build_orion {
    ($self:ident, $data:ident, $num_points:ident, $result:ident, $N:literal, $variant:ident) => {{
        drop($result.index);
        log::info!(
            "  mem after drop(index):    {}",
            crate::metrics::memory::format_bytes(crate::ALLOCATOR.current_bytes())
        );

        let t1 = Instant::now();
        let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
        let mut idx = Orion::<$N>::new(
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
        log::info!("Orion overhead: {:.2}s", t1.elapsed().as_secs_f32());

        let mut ds = diskann::model::InmemDataset::<f32, $N>::new($num_points, 1.0).unwrap();
        ds.data.memcpy(&$data[..$num_points * $N]).unwrap();
        idx.dataset = ds;

        // Auto-calibrate convergence parameters from warmup queries.
        let calib_n = $num_points.min(500);
        let calib_queries: Vec<[f32; $N]> = (0..calib_n)
            .map(|i| {
                let mut q = [0.0f32; $N];
                q.copy_from_slice(&$data[i * $N..(i + 1) * $N]);
                q
            })
            .collect();
        if let Ok(calib) =
            idx.calibrate(&calib_queries, $self.search_list_size, $self.window_size)
        {
            $self.epsilon = calib.threshold;
            $self.early_exit_limit = calib.early_exit_limit;
            log::info!(
                "Calibrated: threshold={:.2}, early_exit_limit={}",
                calib.threshold,
                calib.early_exit_limit
            );
        }

        $self.inner = Some(OrionInner::$variant { idx });
    }};
}

/// Single-query dispatch via the unified cascade pipeline.
macro_rules! search_orion {
    ($idx:ident, $query:ident, $k:ident, $self:ident, $N:literal) => {{
        let mut q = [0.0f32; $N];
        q.copy_from_slice(&$query[..$N]);
        let sls = $self.search_list_size;
        let ws = $self.window_size;
        let eps = $self.epsilon;
        let ee = $self.early_exit_limit;
        let pf = crate::runner::cascade::build_prefilter::<$N>(
            $idx,
            $self.prefilter,
            $self.admission,
        );
        let ad = crate::runner::cascade::build_admission::<$N>($idx, $self.admission);
        let rr = crate::runner::cascade::build_rerank::<$N>($idx, $self.rerank);
        $idx.search_unified(
            &q,
            $k,
            sls,
            ws,
            eps,
            ee,
            pf.as_deref(),
            ad.as_ref(),
            rr.as_ref(),
        )
    }};
}

/// Batch dispatch via the unified cascade pipeline.
macro_rules! search_batch_orion {
    ($idx:ident, $queries:ident, $k:ident, $self:ident, $N:literal) => {{
        let qs: Vec<[f32; $N]> = $queries
            .iter()
            .map(|q| {
                let mut arr = [0.0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            })
            .collect();
        let sls = $self.search_list_size;
        let ws = $self.window_size;
        let eps = $self.epsilon;
        let ee = $self.early_exit_limit;
        crate::runner::cascade::search_batch_compose::<$N>(
            $idx,
            &qs,
            $k,
            sls,
            ws,
            eps,
            ee,
            $self.prefilter,
            $self.admission,
            $self.rerank,
        )
    }};
}

/// Pin the dataset, cascade-tier sidecars, and `PhasedGraph` slab
/// into RAM via `mlock(2)`. Best-effort: pin failures are logged
/// inside `utils::mlock_bytes`, no abort. Routes the per-tier pin
/// through `cascade::pin_cascade` so the cascade builders and the
/// pin policy stay in lockstep.
macro_rules! pin_orion_hot {
    ($idx:ident, $self:ident) => {{
        // f32 base dataset is always hot regardless of cascade choice.
        let ds_ptr = $idx.dataset.data.as_ptr() as *const u8;
        let ds_len = $idx.dataset.data.len() * std::mem::size_of::<f32>();
        crate::utils::mlock_bytes("orion dataset (f32)", ds_ptr, ds_len);

        crate::runner::cascade::pin_cascade(
            $idx,
            $self.prefilter,
            $self.admission,
            $self.rerank,
        );

        // PhasedGraph slot slab.
        let pg = $idx.graph.buffer_bytes();
        crate::utils::mlock_bytes("orion pgraph slab", pg.as_ptr(), pg.len());
    }};
}

impl OrionRunner {
    pub fn set_search_list_size(&mut self, sls: usize) {
        self.search_list_size = sls;
    }

    /// Pin the dataset, quantized sidecar, and PhasedGraph slab into
    /// RAM via `mlock(2)` so warmup + timed trials see the same
    /// resident pages. Mirrors the timed-region setup in
    /// `orion.rs`. Call once after `build()` and before the
    /// timing loop.
    pub fn pin_hot_regions(&self) {
        match self.inner.as_ref().expect("Index not built") {
            OrionInner::Dim32 { idx } => pin_orion_hot!(idx, self),
            OrionInner::Dim100 { idx } => pin_orion_hot!(idx, self),
            OrionInner::Dim128 { idx } => pin_orion_hot!(idx, self),
            OrionInner::Dim768 { idx } => pin_orion_hot!(idx, self),
            OrionInner::Dim784 { idx } => pin_orion_hot!(idx, self),
            OrionInner::Dim960 { idx } => pin_orion_hot!(idx, self),
            OrionInner::Dim1536 { idx } => pin_orion_hot!(idx, self),
        }
    }

    /// Re-run calibration at the **current** `search_list_size` using
    /// real test queries. The build-time calibration done inside
    /// `build()` runs at the build's L (e.g. 128 for SIFT) and uses the
    /// first N base vectors as warmup — both differ from the timed
    /// region's L (48 in thread-sweep) and from real query
    /// distributions. `orion` calibrates at CALIB_L=48 with 200
    /// real queries; mirror that here so the early-exit / threshold
    /// params match and the QPS gap from calibration drift closes.
    pub fn recalibrate(&mut self, queries: &[Vec<f32>], sample: usize) {
        let n = sample.min(queries.len());
        let sls = self.search_list_size;
        let ws = self.window_size;
        macro_rules! recal {
            ($idx:ident, $N:literal) => {{
                let qs: Vec<[f32; $N]> = queries[..n]
                    .iter()
                    .map(|q| {
                        let mut a = [0.0f32; $N];
                        a.copy_from_slice(&q[..$N]);
                        a
                    })
                    .collect();
                $idx.calibrate(&qs, sls, ws).ok()
            }};
        }
        let calib = match self.inner.as_ref().expect("Index not built") {
            OrionInner::Dim32 { idx } => recal!(idx, 32),
            OrionInner::Dim100 { idx } => recal!(idx, 100),
            OrionInner::Dim128 { idx } => recal!(idx, 128),
            OrionInner::Dim768 { idx } => recal!(idx, 768),
            OrionInner::Dim784 { idx } => recal!(idx, 784),
            OrionInner::Dim960 { idx } => recal!(idx, 960),
            OrionInner::Dim1536 { idx } => recal!(idx, 1536),
        };
        if let Some(c) = calib {
            self.epsilon = c.threshold;
            self.early_exit_limit = c.early_exit_limit;
        }
    }

    pub fn calibrated_params(&self) -> (f32, usize) {
        (self.epsilon, self.early_exit_limit)
    }

    pub fn new(
        name: &'static str,
        alpha: f32,
        graph_degree: usize,
        search_list_size: usize,
        max_extra: usize,
        window_size: usize,
        prefilter: PrefilterChoice,
        admission: AdmissionChoice,
        rerank: RerankChoice,
    ) -> Self {
        Self {
            name,
            alpha,
            graph_degree,
            search_list_size,
            max_extra,
            window_size,
            prefilter,
            admission,
            rerank,
            epsilon: 0.0,
            early_exit_limit: 0,
            dimension: 0,
            inner: None,
            cache_path: None,
        }
    }

    pub fn set_cache_path<P: Into<PathBuf>>(&mut self, p: P) {
        self.cache_path = Some(p.into());
    }
}

impl AlgorithmRunner for OrionRunner {
    fn name(&self) -> &str {
        self.name
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        // Cache fast-path. When `cache_path` is set and both `<cache>.bin`
        // + `<cache>.pgraph` exist on disk, skip the in-process Vamana
        // build and the Orion partition+slot construction — load
        // straight from the cached PhasedGraph. Mirrors `orion`'s
        // `load_from_cache` arm so both binaries run on the same graph
        // instance and topology jitter from rebuild-each-run goes away.
        if let Some(cache) = self.cache_path.clone() {
            let pgraph_path = cache.with_extension("pgraph");
            if cache.exists() && pgraph_path.exists() {
                log::info!("Loading cached Orion from {:?}", cache);
                macro_rules! load_cached {
                    ($N:literal, $variant:ident) => {{
                        let empty_ds =
                            diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                        let mut idx = Orion::<$N>::load_from_cache(&cache, empty_ds)
                            .expect("Orion::load_from_cache failed");
                        let mut ds =
                            diskann::model::InmemDataset::<f32, $N>::new(num_points, 1.0).unwrap();
                        ds.data.memcpy(&data[..num_points * $N]).unwrap();
                        idx.dataset = ds;
                        self.inner = Some(OrionInner::$variant { idx });
                    }};
                }
                match dimension {
                    DIM_32 => load_cached!(32, Dim32),
                    DIM_100 => load_cached!(100, Dim100),
                    DIM_128 => load_cached!(128, Dim128),
                    DIM_768 => load_cached!(768, Dim768),
                    DIM_784 => load_cached!(784, Dim784),
                    DIM_960 => load_cached!(960, Dim960),
                    DIM_1536 => load_cached!(1536, Dim1536),
                    _ => panic!("Unsupported dimension: {dimension}"),
                }
                let elapsed = start.elapsed();
                return BuildTiming {
                    graph_build: elapsed,
                    overhead: Duration::ZERO,
                };
            }
        }

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
            self.max_extra,
        )
        .expect("build failed");
        log::info!(
            "DiskANN graph build (parallel Vamana + candidate sets): {:.2}s",
            result.graph_build_time.as_secs_f32()
        );

        match dimension {
            DIM_32 => build_orion!(self, data, num_points, result, 32, Dim32),
            DIM_100 => build_orion!(self, data, num_points, result, 100, Dim100),
            DIM_128 => build_orion!(self, data, num_points, result, 128, Dim128),
            DIM_768 => build_orion!(self, data, num_points, result, 768, Dim768),
            DIM_784 => build_orion!(self, data, num_points, result, 784, Dim784),
            DIM_960 => build_orion!(self, data, num_points, result, 960, Dim960),
            DIM_1536 => build_orion!(self, data, num_points, result, 1536, Dim1536),
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
            OrionInner::Dim32 { idx } => search_orion!(idx, query, k, self, 32),
            OrionInner::Dim100 { idx } => search_orion!(idx, query, k, self, 100),
            OrionInner::Dim128 { idx } => search_orion!(idx, query, k, self, 128),
            OrionInner::Dim768 { idx } => search_orion!(idx, query, k, self, 768),
            OrionInner::Dim784 { idx } => search_orion!(idx, query, k, self, 784),
            OrionInner::Dim960 { idx } => search_orion!(idx, query, k, self, 960),
            OrionInner::Dim1536 { idx } => search_orion!(idx, query, k, self, 1536),
        }
        .expect("Searching process failed");
        SearchResult {
            neighbors,
            duration: start.elapsed(),
        }
    }

    /// Override the trait default (per-query `par_iter`) so the L-adaptive
    /// `par_chunks(BATCH)` shape inside `search_batch_*` actually gets used.
    /// Per-query duration is not preserved (callers that need wall time
    /// measure the whole batch externally — see `search_batch_with_threads`).
    fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Vec<SearchResult> {
        let neighbors_vec: Vec<Vec<u32>> = match self.inner.as_ref().expect("Index not built") {
            OrionInner::Dim32 { idx } => search_batch_orion!(idx, queries, k, self, 32),
            OrionInner::Dim100 { idx } => search_batch_orion!(idx, queries, k, self, 100),
            OrionInner::Dim128 { idx } => search_batch_orion!(idx, queries, k, self, 128),
            OrionInner::Dim768 { idx } => search_batch_orion!(idx, queries, k, self, 768),
            OrionInner::Dim784 { idx } => search_batch_orion!(idx, queries, k, self, 784),
            OrionInner::Dim960 { idx } => search_batch_orion!(idx, queries, k, self, 960),
            OrionInner::Dim1536 { idx } => search_batch_orion!(idx, queries, k, self, 1536),
        }
        .expect("Batch searching process failed");
        neighbors_vec
            .into_iter()
            .map(|n| SearchResult {
                neighbors: n,
                duration: Duration::default(),
            })
            .collect()
    }
}
