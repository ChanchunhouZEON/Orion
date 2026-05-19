/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::config::Metric;
use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Benchmark runner for Staged DiskANN with compile-time dimension dispatch
/// and runtime metric dispatch (L2 / L2-Q / MIPS / MIPS-Q).
///
/// Both `search` (single query) and `search_batch` (multi-query) route to the
/// metric's dedicated kernel — the latter calls
/// `StagedDiskANN::search_batch_*` which has the L-adaptive `par_chunks(BATCH)`
/// rayon shape, instead of the trait's default per-query `par_iter`.
pub struct StagedDiskANNRunner {
    name: &'static str,
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    max_extra: usize,
    window_size: usize,
    /// Search metric — chosen at construction time, drives both single-query
    /// `search` and multi-query `search_batch` dispatch.
    metric: Metric,
    /// Auto-calibrated during build.
    epsilon: f32,
    /// Auto-calibrated during build.
    early_exit_limit: usize,
    dimension: usize,
    inner: Option<StagedInner>,
    /// Optional cache path. When set and `<cache>.bin` + `<cache>.pgraph`
    /// both exist, `build()` loads from disk and skips the in-process
    /// Vamana build entirely — mirrors `staged_sweep`'s `load_from_cache`
    /// arm so thread-sweep and staged_sweep run on identical graphs.
    cache_path: Option<PathBuf>,
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
            &$result.partitions,
            $result.entry_point,
            $self.graph_degree as u32,
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
            staged.calibrate(&calib_queries, $self.search_list_size, $self.window_size)
        {
            $self.epsilon = calib.threshold;
            $self.early_exit_limit = calib.early_exit_limit;
            log::info!(
                "Calibrated: threshold={:.2}, early_exit_limit={}",
                calib.threshold,
                calib.early_exit_limit
            );
        }

        $self.inner = Some(StagedInner::$variant { staged });
    }};
}

/// Single-query dispatch by metric. Mirrors `staged_sweep`'s metric arm
/// without the warmup/cache-flush boilerplate — pure search call.
macro_rules! search_staged {
    ($staged:ident, $query:ident, $k:ident, $self:ident, $N:literal) => {{
        let mut q = [0.0f32; $N];
        q.copy_from_slice(&$query[..$N]);
        let sls = $self.search_list_size;
        let ws = $self.window_size;
        let eps = $self.epsilon;
        let ee = $self.early_exit_limit;
        match $self.metric {
            Metric::L2 => $staged.search_l2_u8(&q, $k, sls, ws, eps, ee),
            Metric::L2Q => $staged.search_l2_u8_q(&q, $k, sls, ws, eps, ee),
            Metric::Mips => $staged.search_mips(&q, $k, sls, ws, eps, ee),
            Metric::MipsQ => {
                let q_ds = $staged.ensure_quantized_dataset_mips();
                $staged.search_mips_q::<staged_diskann::model::MipsI8>(
                    &q, q_ds, $k, sls, ws, eps, ee,
                )
            }
        }
    }};
}

/// Batch dispatch by metric. Routes to the appropriate `search_batch_*`
/// family method on `StagedDiskANN`, which carries the L-adaptive
/// `par_chunks(BATCH)` rayon shape (kills low-L dispatch overhead).
/// Returns `Vec<Vec<u32>>` (per-query neighbour lists); the wrapping
/// `search_batch` impl below stuffs them into `SearchResult`.
macro_rules! search_batch_staged {
    ($staged:ident, $queries:ident, $k:ident, $self:ident, $N:literal) => {{
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
        match $self.metric {
            Metric::L2 => $staged.search_batch_l2_u8(&qs, $k, sls, ws, eps, ee),
            Metric::L2Q => $staged.search_batch_l2_u8_q(&qs, $k, sls, ws, eps, ee),
            Metric::Mips => $staged.search_batch_mips(&qs, $k, sls, ws, eps, ee),
            Metric::MipsQ => {
                let q_ds = $staged.ensure_quantized_dataset_mips();
                $staged.search_batch_mips_q::<staged_diskann::model::MipsI8>(
                    &qs, q_ds, $k, sls, ws, eps, ee,
                )
            }
        }
    }};
}

/// Inner helper for the `pin_hot_regions!` macro — pins the dataset,
/// the metric-appropriate quantized sidecar, and the `PhasedGraph`
/// slab into RAM via `mlock(2)`. Mirrors the corresponding block in
/// `staged_sweep.rs`. Best-effort: pin failures are logged inside
/// `utils::mlock_bytes`, no abort.
macro_rules! pin_staged_hot {
    ($staged:ident, $metric:expr) => {{
        // f32 base dataset.
        let ds_ptr = $staged.dataset.data.as_ptr() as *const u8;
        let ds_len = $staged.dataset.data.len() * std::mem::size_of::<f32>();
        crate::utils::mlock_bytes("staged dataset (f32)", ds_ptr, ds_len);

        // Metric-appropriate quantized sidecar. Reading via the public
        // `ensure_quantized_*` accessors guarantees the sidecar is
        // built/loaded before we mlock its bytes.
        match $metric {
            crate::config::Metric::L2 | crate::config::Metric::L2Q => {
                let q = $staged.ensure_quantized_dataset();
                let len_bytes = q.data.len() * std::mem::size_of::<u8>();
                crate::utils::mlock_bytes(
                    "staged qdataset (u8)",
                    q.data.as_ptr() as *const u8,
                    len_bytes,
                );
            }
            crate::config::Metric::Mips => { /* no quantized sidecar */ }
            crate::config::Metric::MipsQ => {
                let q = $staged.ensure_quantized_dataset_mips();
                let len_bytes = q.data.len() * std::mem::size_of::<i8>();
                crate::utils::mlock_bytes(
                    "staged qdataset (i8)",
                    q.data.as_ptr() as *const u8,
                    len_bytes,
                );
            }
        }

        // PhasedGraph slot slab.
        let pg = $staged.graph.buffer_bytes();
        crate::utils::mlock_bytes("staged pgraph slab", pg.as_ptr(), pg.len());
    }};
}

impl StagedDiskANNRunner {
    pub fn set_search_list_size(&mut self, sls: usize) {
        self.search_list_size = sls;
    }

    /// Pin the dataset, quantized sidecar, and PhasedGraph slab into
    /// RAM via `mlock(2)` so warmup + timed trials see the same
    /// resident pages. Mirrors the timed-region setup in
    /// `staged_sweep.rs`. Call once after `build()` and before the
    /// timing loop.
    pub fn pin_hot_regions(&self) {
        let m = self.metric;
        match self.inner.as_ref().expect("Index not built") {
            StagedInner::Dim32 { staged } => pin_staged_hot!(staged, m),
            StagedInner::Dim100 { staged } => pin_staged_hot!(staged, m),
            StagedInner::Dim128 { staged } => pin_staged_hot!(staged, m),
            StagedInner::Dim960 { staged } => pin_staged_hot!(staged, m),
        }
    }

    /// Re-run calibration at the **current** `search_list_size` using
    /// real test queries. The build-time calibration done inside
    /// `build()` runs at the build's L (e.g. 128 for SIFT) and uses the
    /// first N base vectors as warmup — both differ from the timed
    /// region's L (48 in thread-sweep) and from real query
    /// distributions. `staged_sweep` calibrates at CALIB_L=48 with 200
    /// real queries; mirror that here so the early-exit / threshold
    /// params match and the QPS gap from calibration drift closes.
    pub fn recalibrate(&mut self, queries: &[Vec<f32>], sample: usize) {
        let n = sample.min(queries.len());
        let sls = self.search_list_size;
        let ws = self.window_size;
        macro_rules! recal {
            ($staged:ident, $N:literal) => {{
                let qs: Vec<[f32; $N]> = queries[..n]
                    .iter()
                    .map(|q| {
                        let mut a = [0.0f32; $N];
                        a.copy_from_slice(&q[..$N]);
                        a
                    })
                    .collect();
                $staged.calibrate(&qs, sls, ws).ok()
            }};
        }
        let calib = match self.inner.as_ref().expect("Index not built") {
            StagedInner::Dim32 { staged } => recal!(staged, 32),
            StagedInner::Dim100 { staged } => recal!(staged, 100),
            StagedInner::Dim128 { staged } => recal!(staged, 128),
            StagedInner::Dim960 { staged } => recal!(staged, 960),
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
        metric: Metric,
    ) -> Self {
        Self {
            name,
            alpha,
            graph_degree,
            search_list_size,
            max_extra,
            window_size,
            metric,
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

impl AlgorithmRunner for StagedDiskANNRunner {
    fn name(&self) -> &str {
        self.name
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        // Cache fast-path. When `cache_path` is set and both `<cache>.bin`
        // + `<cache>.pgraph` exist on disk, skip the in-process Vamana
        // build and the StagedDiskANN partition+slot construction — load
        // straight from the cached PhasedGraph. Mirrors `staged_sweep`'s
        // `load_from_cache` arm so both binaries run on the same graph
        // instance and topology jitter from rebuild-each-run goes away.
        if let Some(cache) = self.cache_path.clone() {
            let pgraph_path = cache.with_extension("pgraph");
            if cache.exists() && pgraph_path.exists() {
                log::info!("Loading cached StagedDiskANN from {:?}", cache);
                macro_rules! load_cached {
                    ($N:literal, $variant:ident) => {{
                        let empty_ds =
                            diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                        let mut staged =
                            StagedDiskANN::<$N>::load_from_cache(&cache, empty_ds)
                                .expect("StagedDiskANN::load_from_cache failed");
                        let mut ds =
                            diskann::model::InmemDataset::<f32, $N>::new(num_points, 1.0)
                                .unwrap();
                        ds.data
                            .memcpy(&data[..num_points * $N])
                            .unwrap();
                        staged.dataset = ds;
                        self.inner = Some(StagedInner::$variant { staged });
                    }};
                }
                match dimension {
                    DIM_32 => load_cached!(32, Dim32),
                    DIM_100 => load_cached!(100, Dim100),
                    DIM_128 => load_cached!(128, Dim128),
                    DIM_960 => load_cached!(960, Dim960),
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

    /// Override the trait default (per-query `par_iter`) so the L-adaptive
    /// `par_chunks(BATCH)` shape inside `search_batch_*` actually gets used.
    /// Per-query duration is not preserved (callers that need wall time
    /// measure the whole batch externally — see `search_batch_with_threads`).
    fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Vec<SearchResult> {
        let neighbors_vec: Vec<Vec<u32>> = match self.inner.as_ref().expect("Index not built") {
            StagedInner::Dim32 { staged } => search_batch_staged!(staged, queries, k, self, 32),
            StagedInner::Dim100 { staged } => search_batch_staged!(staged, queries, k, self, 100),
            StagedInner::Dim128 { staged } => search_batch_staged!(staged, queries, k, self, 128),
            StagedInner::Dim960 { staged } => search_batch_staged!(staged, queries, k, self, 960),
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

    fn memory_bytes(&self) -> usize {
        0
    }
}
