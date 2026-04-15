/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

mod dataset;
mod metrics;
mod report;
mod runner;

use clap::Parser;
use dataset::Dataset;
use metrics::LatencyStats;
use report::table::{print_results_table, BenchmarkResult};
use runner::common::AlgorithmRunner;
use runner::{DiskANNRunner, HNSWRunner, NSGRunner, SSDDiskANNRunner, StagedDiskANNRunner};
use std::path::PathBuf;

/// Resolved per-dataset staged config (defaults merged with dataset overrides).
struct StagedConfig {
    alpha: f32,
    graph_degree: u32,
    build_search_list_size: usize,
    max_extra: usize,
    window_size: usize,
}

fn load_staged_config(dimension: usize) -> StagedConfig {
    #[derive(serde::Deserialize)]
    struct Cfg {
        defaults: Defaults,
        #[serde(default)]
        datasets: std::collections::HashMap<String, DsCfg>,
    }
    #[derive(serde::Deserialize)]
    struct Defaults {
        staged: StagedDef,
    }
    #[derive(serde::Deserialize)]
    struct StagedDef {
        alpha: f32,
        graph_degree: u32,
        build_search_list_size: usize,
        max_extra: usize,
        window_size: usize,
    }
    #[derive(serde::Deserialize, Default)]
    struct DsCfg {
        staged: Option<DsOverride>,
    }
    #[derive(serde::Deserialize, Default)]
    struct DsOverride {
        alpha: Option<f32>,
        build_search_list_size: Option<usize>,
    }

    let path = "benchmark/configs/sweep.yaml";
    let cfg: Cfg = serde_yaml::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Cannot read {path}")),
    )
    .expect("Invalid sweep config YAML");

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };
    let ds = cfg.datasets.get(dim_name);
    let ov = ds.and_then(|d| d.staged.as_ref());

    StagedConfig {
        alpha: ov.and_then(|o| o.alpha).unwrap_or(cfg.defaults.staged.alpha),
        graph_degree: cfg.defaults.staged.graph_degree,
        build_search_list_size: ov
            .and_then(|o| o.build_search_list_size)
            .unwrap_or(cfg.defaults.staged.build_search_list_size),
        max_extra: cfg.defaults.staged.max_extra,
        window_size: cfg.defaults.staged.window_size,
    }
}

#[global_allocator]
static ALLOCATOR: metrics::TrackingAllocator = metrics::TrackingAllocator::new();

#[derive(Parser)]
#[command(name = "ann-benchmark", about = "Unified ANNS benchmark suite")]
struct Args {
    /// Path to base vectors file (.fvecs)
    #[arg(long)]
    base: Option<PathBuf>,

    /// Path to query vectors file (.fvecs)
    #[arg(long)]
    query: Option<PathBuf>,

    /// Path to ground truth file (.ivecs)
    #[arg(long)]
    groundtruth: Option<PathBuf>,

    /// Number of nearest neighbors to retrieve
    #[arg(long, default_value = "10")]
    k: usize,

    /// Algorithms to benchmark (comma-separated: diskann,hnsw,nsg,compressed-diskann)
    #[arg(long, default_value = "diskann,hnsw,nsg")]
    algorithms: String,

    /// Maximum number of base vectors to use (0 = use all)
    #[arg(long, default_value = "0")]
    max_points: usize,

    /// Generate random dataset instead of loading files
    #[arg(long)]
    random_dataset: bool,

    /// Number of random base points (default: 10000)
    #[arg(long, default_value = "10000")]
    random_num_points: usize,

    /// Number of random queries (default: 100)
    #[arg(long, default_value = "100")]
    random_num_queries: usize,

    /// Enable mmap-based search (real disk IO instead of simulation)
    #[arg(long)]
    mmap_search: bool,

    /// Directory for mmap graph files
    #[arg(long, default_value = "./anns_graphs/")]
    graph_dir: PathBuf,

    /// Path for PQ data on ramdisk (optional)
    #[arg(long)]
    ramdisk_path: Option<PathBuf>,

    /// Entry point prefetch depth for warm cache (default: 4)
    #[arg(long, default_value = "4")]
    warm_cache_hops: usize,

    /// Memory limit in MB for search phase (0 = unlimited). Uses TrackingAllocator.
    #[arg(long, default_value = "0")]
    memory_limit_mb: usize,

    /// Drop in-memory vectors after enabling mmap search (reduces heap usage)
    #[arg(long)]
    drop_inmem: bool,
}

/// Rebuild an InmemDataset from flat base vectors.
/// Used after extract_graph_and_candidates frees the dataset to reduce peak memory.
fn rebuild_dataset<const N: usize>(
    flat_base: &[f32],
    num_points: usize,
) -> diskann::model::InmemDataset<f32, N>
where
    [f32; N]: vector::FullPrecisionDistance<f32, N>,
{
    let mut ds = diskann::model::InmemDataset::<f32, N>::new(num_points, 1.0).unwrap();
    ds.data.memcpy(&flat_base[..num_points * N]).unwrap();
    ds
}

fn load_dataset(args: &Args) -> Dataset {
    let base_path = args
        .base
        .as_ref()
        .expect("--base is required when not using --random-dataset");
    let query_path = args
        .query
        .as_ref()
        .expect("--query is required when not using --random-dataset");
    let gt_path = args
        .groundtruth
        .as_ref()
        .expect("--groundtruth is required when not using --random-dataset");

    // Read only the needed number of base vectors (avoids loading full SIFT1M).
    let max_base = if args.max_points > 0 {
        args.max_points
    } else {
        0
    };
    log::info!(
        "Loading base vectors from {:?} (max={})",
        base_path,
        if max_base == 0 {
            "all".to_string()
        } else {
            max_base.to_string()
        }
    );
    log::info!(
        "  mem before base load: {}",
        metrics::memory::format_bytes(ALLOCATOR.current_bytes())
    );
    let base = dataset::read_fvecs_n(base_path, max_base).expect("Failed to read base vectors");
    let dimension = base.first().map(|v| v.len()).unwrap_or(0);
    log::info!(
        "  mem after base load:  {} ({} vecs)",
        metrics::memory::format_bytes(ALLOCATOR.current_bytes()),
        base.len()
    );

    log::info!("Loading query vectors from {:?}", query_path);
    let queries = dataset::read_fvecs(query_path).expect("Failed to read query vectors");
    log::info!(
        "  mem after query load: {}",
        metrics::memory::format_bytes(ALLOCATOR.current_bytes())
    );

    log::info!("Loading ground truth from {:?}", gt_path);
    let ground_truth = dataset::read_ivecs(gt_path).expect("Failed to read ground truth");
    log::info!(
        "  mem after gt load:    {}",
        metrics::memory::format_bytes(ALLOCATOR.current_bytes())
    );

    log::info!(
        "Dataset: {} base vectors, {} queries, dim={}",
        base.len(),
        queries.len(),
        dimension
    );

    let truncated = max_base > 0 && max_base < 1_000_000; // heuristic: if we limited, gt may be stale

    let mut ds = Dataset {
        name: base_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        dimension,
        base,
        queries,
        ground_truth,
    };

    // When the base is truncated, the original ground truth references points
    // that may no longer exist. Recompute with bounded-heap kNN (O(k) memory per query).
    if truncated {
        let gt_k = ds.ground_truth.first().map(|v| v.len()).unwrap_or(100);
        println!(
            "Base loaded with {} points — recomputing ground truth (k={})...",
            ds.num_base(),
            gt_k
        );
        ds.recompute_ground_truth(gt_k);
        log::info!(
            "  mem after gt recomp:  {}",
            metrics::memory::format_bytes(ALLOCATOR.current_bytes())
        );
    }

    ds
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    let dataset = if args.random_dataset {
        // Dimension is 128 (the most commonly supported across all algorithms)
        let dimension = 128;
        let gt_k = 100; // ground truth depth
        dataset::generate_random_dataset(
            args.random_num_points,
            args.random_num_queries,
            dimension,
            gt_k,
            42, // seed
        )
    } else {
        load_dataset(&args)
    };

    let selected: Vec<&str> = args.algorithms.split(',').map(|s| s.trim()).collect();
    let mut results = Vec::new();

    for algo in &selected {
        match *algo {
            "diskann" => {
                let mut runner = DiskANNRunner::new(48, 32, 2.0);
                run_benchmark(
                    &mut runner,
                    &dataset,
                    args.k,
                    args.mmap_search,
                    args.drop_inmem,
                    &args.graph_dir,
                    args.warm_cache_hops,
                    args.memory_limit_mb,
                    &mut results,
                );
            }
            "hnsw" => {
                let mut runner = HNSWRunner::new(16, 200, 100);
                run_benchmark(
                    &mut runner,
                    &dataset,
                    args.k,
                    args.mmap_search,
                    args.drop_inmem,
                    &args.graph_dir,
                    args.warm_cache_hops,
                    args.memory_limit_mb,
                    &mut results,
                );
            }
            "nsg" => {
                let mut runner = NSGRunner::new(32, 40, 300, 50);
                run_benchmark(
                    &mut runner,
                    &dataset,
                    args.k,
                    args.mmap_search,
                    args.drop_inmem,
                    &args.graph_dir,
                    args.warm_cache_hops,
                    args.memory_limit_mb,
                    &mut results,
                );
            }
            "staged-diskann" => {
                let mut runner = StagedDiskANNRunner::new(
                    "StagedDiskANN",
                    1.2, // alpha (sparser graph, more remote shortcuts)
                    32,  // graph_degree
                    48,  // search_list_size
                    4,   // max_extra
                    5,   // window_size
                );
                run_benchmark(
                    &mut runner,
                    &dataset,
                    args.k,
                    args.mmap_search,
                    args.drop_inmem,
                    &args.graph_dir,
                    args.warm_cache_hops,
                    args.memory_limit_mb,
                    &mut results,
                );
            }
            "in_mem_staged_diskann" => {
                // staged_diskann in_mem_search with auto-calibrated convergence.
                let mut runner = StagedDiskANNRunner::new(
                    "InMemStagedDiskANN",
                    1.2, // alpha
                    32,  // graph_degree
                    48,  // search_list_size
                    4,   // max_extra
                    5,   // window_size
                );
                run_benchmark(
                    &mut runner,
                    &dataset,
                    args.k,
                    args.mmap_search,
                    args.drop_inmem,
                    &args.graph_dir,
                    args.warm_cache_hops,
                    args.memory_limit_mb,
                    &mut results,
                );
            }
            "ssd_diskann" => {
                let mut runner = SSDDiskANNRunner::new(8, 64, 32, 1.2, 8, 8);
                run_benchmark(
                    &mut runner,
                    &dataset,
                    args.k,
                    args.mmap_search,
                    args.drop_inmem,
                    &args.graph_dir,
                    args.warm_cache_hops,
                    args.memory_limit_mb,
                    &mut results,
                );
            }
            "build-profile" => {
                run_build_profile(&dataset, args.k);
            }
            "convergence-diag" => {
                run_convergence_diag(&dataset, args.k);
            }
            "thread-sweep" => {
                run_thread_sweep(&dataset, args.k);
            }
            "search-profile" => {
                run_search_profile(&dataset, args.k);
            }
            "memory-profile" => {
                run_memory_profile(&dataset);
            }
            "qps-recall-sweep" => {
                run_qps_recall_sweep(&dataset, args.k);
            }
            "graph-stats" => {
                run_graph_stats(&dataset);
            }
            "cliff-profile" => {
                run_cliff_profile(&dataset);
            }
            "neighbor-contribution" => {
                run_neighbor_contribution_profile(&dataset);
            }
            "extra-profile" => {
                run_extra_profile(&dataset, args.k);
            }
            "calibration-diag" => {
                run_calibration_diag(&dataset);
            }
            other => {
                log::warn!("Unknown algorithm: {other}");
            }
        }
    }

    if !results.is_empty() {
        println!("\n=== Benchmark Results ===\n");
        print_results_table(&results);
    }
}

fn run_benchmark(
    runner: &mut dyn AlgorithmRunner,
    dataset: &Dataset,
    k: usize,
    mmap_search: bool,
    drop_inmem: bool,
    graph_dir: &std::path::Path,
    warm_cache_hops: usize,
    memory_limit_mb: usize,
    results: &mut Vec<BenchmarkResult>,
) {
    // Build first
    let flat_base = dataset.base_flat();
    let dimension = dataset.dimension;
    let num_points = dataset.num_base();

    log::info!("[{}] Building index...", runner.name());
    let mem_before_build = ALLOCATOR.current_bytes();
    ALLOCATOR.reset_peak();
    let build_timing = runner.build(&flat_base, num_points, dimension);
    let peak_memory = ALLOCATOR.peak_bytes();
    let index_memory = ALLOCATOR.current_bytes();
    log::info!(
        "[{}] Memory: before_build={} peak={} after_build={} index_net={}",
        runner.name(),
        metrics::memory::format_bytes(mem_before_build),
        metrics::memory::format_bytes(peak_memory),
        metrics::memory::format_bytes(index_memory),
        metrics::memory::format_bytes(index_memory.saturating_sub(mem_before_build)),
    );
    log::info!(
        "[{}] Build complete: {:.2}s",
        runner.name(),
        build_timing.total().as_secs_f64()
    );

    // Enable mmap search if requested and supported
    if mmap_search && runner.supports_mmap() {
        log::info!(
            "[{}] Saving mmap graph to {:?}...",
            runner.name(),
            graph_dir
        );
        match runner.save_mmap(graph_dir) {
            Ok(path) => {
                log::info!(
                    "[{}] Enabling mmap search from {:?}...",
                    runner.name(),
                    path
                );
                if let Err(e) = runner.enable_mmap_search(&path) {
                    log::warn!("[{}] Failed to enable mmap search: {e}", runner.name());
                } else {
                    log::info!(
                        "[{}] Mmap search enabled, warming cache ({} hops)...",
                        runner.name(),
                        warm_cache_hops
                    );
                    runner.warm_cache(warm_cache_hops);
                    log::info!("[{}] Cache warm complete", runner.name());

                    if drop_inmem {
                        log::info!(
                            "[{}] Dropping in-memory vectors and graph...",
                            runner.name()
                        );
                        runner.drop_inmem_vectors();
                        log::info!("[{}] In-memory data released", runner.name());
                    }
                }
            }
            Err(e) => {
                log::warn!("[{}] Failed to save mmap graph: {e}", runner.name());
            }
        }
    }

    // Apply memory limit for search phase (relative to current usage)
    if memory_limit_mb > 0 {
        let current = ALLOCATOR.current_bytes();
        let limit_bytes = current + memory_limit_mb * 1024 * 1024;
        log::info!(
            "[{}] Setting memory limit: current {} + {} MB budget = {}",
            runner.name(),
            metrics::memory::format_bytes(current),
            memory_limit_mb,
            metrics::memory::format_bytes(limit_bytes)
        );
        ALLOCATOR.set_limit(limit_bytes);
    }

    // Search — parallel batch for both DiskANN and StagedDiskANN
    log::info!(
        "[{}] Running {} queries (parallel batch)...",
        runner.name(),
        dataset.num_queries()
    );
    let queries = &dataset.queries;

    let batch_start = std::time::Instant::now();
    let batch_results = runner.search_batch(queries, k);
    let batch_wall = batch_start.elapsed();

    let qps = queries.len() as f64 / batch_wall.as_secs_f64();
    let durations: Vec<_> = batch_results.iter().map(|r| r.duration).collect();
    let all_results: Vec<Vec<u32>> = batch_results.into_iter().map(|r| r.neighbors).collect();
    let latency = LatencyStats::from_durations(durations);

    // Calculate recall
    let recall_at_1 = metrics::recall::mean_recall(&all_results, &dataset.ground_truth, 1);
    let recall_at_10 = metrics::recall::mean_recall(&all_results, &dataset.ground_truth, 10);
    let recall_at_100 = metrics::recall::mean_recall(&all_results, &dataset.ground_truth, 100);

    log::info!(
        "[{}] R@1={:.4} R@10={:.4} R@100={:.4} QPS={:.1} {}",
        runner.name(),
        recall_at_1,
        recall_at_10,
        recall_at_100,
        qps,
        latency,
    );

    // Clear memory limit so next algorithm isn't affected
    if memory_limit_mb > 0 {
        ALLOCATOR.set_limit(0);
    }

    results.push(BenchmarkResult {
        algorithm: runner.name().to_string(),
        params: String::new(),
        build_timing,
        recall_at_1,
        recall_at_10,
        recall_at_100,
        qps,
        latency,
        peak_memory,
        index_memory,
    });
}

/// Profile and compare DiskANN build time vs StagedDiskANN build time.
///
/// Measures:
///   - DiskANN Vamana graph build time
///   - StagedDiskANN clustering + compression overhead
/// Runs 3 trials each and reports mean ± stddev.
/// Diagnose convergence behavior and compressed graph quality.
///
/// Sweeps over base_local_count values to compare
/// degree distribution and convergence/recall trade-offs.
fn run_convergence_diag(dataset: &Dataset, k: usize) {
    use ndarray::Array2;
    use staged_diskann::{
        build_diskann_index, PhasedGraph, StagedDiskANN, DEFAULT_EARLY_EXIT_LIMIT, DIM_128,
    };
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128);

    let flat_base = dataset.base_flat();
    let data_2d = Array2::from_shape_vec((num_points, dimension), dataset.base_flat())
        .expect("reshape")
        .to_shared();

    // Remove cached graph directory
    let cache_dir = std::path::PathBuf::from("staged_diskann_graphs");
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    println!("Building DiskANN graph ({num_points} pts, dim={dimension})...");
    let result = build_diskann_index(
        &flat_base, num_points, dimension, 1.2, 32, 48, false, None, None, true, 16,
    )
    .expect("build failed");
    println!(
        "  Graph build: {:.2}s\n",
        result.graph_build_time.as_secs_f32()
    );

    drop(result.index);

    // Build PhasedGraph once, clone for each config.
    let phased_graph = PhasedGraph::build_from_partitions(
        &result.partitions,
        16,
        4, // max_extra
    );

    // Helper: build an InmemDataset<f32, 128> from the ndarray data.
    let make_dataset = |data: &ndarray::ArcArray2<f32>| -> diskann::model::InmemDataset<f32, 128> {
        let n = data.nrows();
        let mut ds = diskann::model::InmemDataset::<f32, 128>::new(n, 1.0).unwrap();
        let flat: &[f32] = data.as_slice().expect("contiguous data_2d");
        ds.data.memcpy(&flat[..n * 128]).unwrap();
        ds
    };

    let queries_arr: Vec<[f32; 128]> = dataset
        .queries
        .iter()
        .map(|q| {
            let mut arr = [0.0f32; 128];
            arr.copy_from_slice(&q[..128]);
            arr
        })
        .collect();

    // ── Sweep base_local_count values ───────────────────────────────
    let local_counts: &[usize] = &[4, 8, 12, 16];
    let epsilons: &[f32] = &[0.0, 0.005, 0.01, 0.05, 0.1];
    let ws = 5usize;

    for &lc in local_counts {
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir).ok();
        }

        let t = Instant::now();
        let dataset_128 = make_dataset(&data_2d);
        let staged = StagedDiskANN::<128>::from_phased_graph(
            dataset_128,
            phased_graph.clone(),
            result.entry_point,
            None,
            None,
        );
        let build_t = t.elapsed();

        // ── Degree distribution ──────────────────────────────────────
        let degree_stats = staged.graph_degree_stats();
        let nn = degree_stats.len();
        let mut comp_degrees = Vec::with_capacity(nn);
        let mut zero_compressed = 0usize;
        for &(_full, comp) in &degree_stats {
            comp_degrees.push(comp);
            if comp == 0 {
                zero_compressed += 1;
            }
        }
        let avg_full = degree_stats.iter().map(|d| d.0).sum::<usize>() as f64 / nn as f64;
        let avg_comp = comp_degrees.iter().sum::<usize>() as f64 / nn as f64;
        let max_comp = comp_degrees.iter().max().copied().unwrap_or(0);

        println!(
            "═══ base_local_count={}  build={:.3}s ═══",
            lc,
            build_t.as_secs_f32()
        );
        println!("  Avg full degree: {avg_full:.1}  |  Avg local: {avg_comp:.1}  |  Max local: {max_comp}  |  Zero: {zero_compressed}");

        // Histogram (compact)
        let max_bucket = lc.min(32) + 1;
        let mut hist = vec![0usize; max_bucket + 1];
        for &cd in &comp_degrees {
            hist[cd.min(max_bucket)] += 1;
        }
        print!("  Degree histogram: ");
        for (deg, &count) in hist.iter().enumerate() {
            if count > 0 {
                print!("[{}]={} ", deg, count);
            }
        }
        println!();

        // ── Convergence sweep ────────────────────────────────────────
        println!(
            "\n  {:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10} {:<8}",
            "Epsilon", "R@10", "QPS", "AvgSteps", "ConvAt", "Conv%", "P1_NDC", "P2_NDC"
        );
        println!("  {}", "─".repeat(78));

        for &eps in epsilons {
            let nq = queries_arr.len();
            let mut total_steps = 0usize;
            let mut total_conv_at = 0usize;
            let mut total_p1_ndc = 0usize;
            let mut total_p2_ndc = 0usize;
            let mut all_results = Vec::with_capacity(nq);

            let t0 = Instant::now();
            for q in &queries_arr {
                let (res, conv_at, steps, p1_ndc, p2_ndc) = staged
                    .search_diag(q, k, 48, ws, eps, DEFAULT_EARLY_EXIT_LIMIT)
                    .unwrap();
                all_results.push(res);
                total_steps += steps;
                total_conv_at += conv_at;
                total_p1_ndc += p1_ndc;
                total_p2_ndc += p2_ndc;
            }
            let qps = nq as f64 / t0.elapsed().as_secs_f64();

            let recall = metrics::recall::mean_recall(&all_results, &dataset.ground_truth, k);
            let avg_steps = total_steps as f64 / nq as f64;
            let avg_conv = total_conv_at as f64 / nq as f64;
            let conv_pct = if avg_steps > 0.0 {
                avg_conv / avg_steps * 100.0
            } else {
                0.0
            };
            let avg_p1 = total_p1_ndc as f64 / nq as f64;
            let avg_p2 = total_p2_ndc as f64 / nq as f64;

            println!(
                "  {:<8.4} {:<8.4} {:<8.0} {:<10.1} {:<10.1} {:<10.1} {:<10.1} {:<8.1}",
                eps, recall, qps, avg_steps, avg_conv, conv_pct, avg_p1, avg_p2,
            );
        }
        println!();
    }

    // Clean up
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }
}

fn run_thread_sweep(dataset: &Dataset, k: usize) {
    use staged_diskann::DIM_128;
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128);

    let flat_base = dataset.base_flat();

    // Build DiskANN index
    println!("Building DiskANN index ({num_points} pts)...");
    let mut diskann_runner = runner::DiskANNRunner::new(48, 32, 1.2);
    diskann_runner.build(&flat_base, num_points, dimension);

    // Build StagedDiskANN index
    println!("Building StagedDiskANN index ({num_points} pts)...");
    let mut staged_runner =
        runner::StagedDiskANNRunner::new("StagedDiskANN", 1.2, 32, 48, 4, 5);
    staged_runner.build(&flat_base, num_points, dimension);

    let queries = &dataset.queries;
    let thread_counts = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32];

    println!(
        "\n─── Thread Sweep (k={k}, {} queries) ───\n",
        queries.len()
    );
    println!(
        "  {:<8} {:<14} {:<14} {:<10} {:<14} {:<14} {:<10}",
        "Threads", "DiskANN QPS", "DiskANN P99", "D_R@10", "Staged QPS", "Staged P99", "S_R@10",
    );
    println!("  {}", "─".repeat(90));

    for &nt in &thread_counts {
        // DiskANN
        let t0 = Instant::now();
        let d_results = diskann_runner.search_batch_with_threads(queries, k, nt);
        let d_wall = t0.elapsed();
        let d_qps = queries.len() as f64 / d_wall.as_secs_f64();
        let d_recall = {
            let r: Vec<Vec<u32>> = d_results.into_iter().map(|r| r.neighbors).collect();
            metrics::recall::mean_recall(&r, &dataset.ground_truth, k)
        };
        let d_p99 = {
            let mut durs: Vec<f64> = Vec::new();
            // approximate p99 from wall time / threads
            durs.push(d_wall.as_secs_f64() * 1000.0);
            durs[0]
        };

        // StagedDiskANN
        let t1 = Instant::now();
        let s_results = staged_runner.search_batch_with_threads(queries, k, nt);
        let s_wall = t1.elapsed();
        let s_qps = queries.len() as f64 / s_wall.as_secs_f64();
        let s_recall = {
            let r: Vec<Vec<u32>> = s_results.into_iter().map(|r| r.neighbors).collect();
            metrics::recall::mean_recall(&r, &dataset.ground_truth, k)
        };
        let s_p99 = s_wall.as_secs_f64() * 1000.0;

        println!(
            "  {:<8} {:<14.0} {:<14.2} {:<10.4} {:<14.0} {:<14.2} {:<10.4}",
            nt, d_qps, d_p99, d_recall, s_qps, s_p99, s_recall,
        );
    }
}

fn run_build_profile(dataset: &Dataset, _k: usize) {
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(
        dimension, DIM_128,
        "build-profile currently requires dim=128 (SIFT); got {}",
        dimension
    );

    let flat_base = dataset.base_flat();

    const TRIALS: usize = 10;
    let mut vamana_times = Vec::with_capacity(TRIALS);
    let mut overhead_times = Vec::with_capacity(TRIALS);
    let mut total_times = Vec::with_capacity(TRIALS);

    for trial in 0..TRIALS {
        println!("=== Trial {}/{TRIALS} ===", trial + 1);

        let cache_dir = std::path::PathBuf::from("staged_diskann_graphs");
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir).ok();
        }

        // ── DiskANN baseline: create_inmem_index + build_from_data ──
        let diskann_link_t = {
            let num_threads = rayon::current_num_threads() as u32;
            let wp = diskann::model::IndexWriteParametersBuilder::new(48, 32)
                .with_alpha(1.2)
                .with_num_threads(num_threads)
                .build();
            let config = diskann::model::IndexConfiguration::new(
                vector::Metric::L2,
                dimension,
                dimension,
                num_points,
                false,
                0,
                false,
                0,
                1.0,
                wp,
            );
            let mut idx = diskann::index::create_inmem_index::<f32>(config).expect("create index");
            let t = Instant::now();
            idx.build_from_data(&flat_base, num_points)
                .expect("build_from_data");
            let elapsed = t.elapsed().as_secs_f32();
            drop(idx);
            elapsed
        };

        // StagedDiskANN: compute_candidate_sets=true
        let t_total = Instant::now();
        let mut result = build_diskann_index(
            &flat_base, num_points, dimension, 1.2, 32, 48, false, None, None, true, 16,
        )
        .expect("staged build");
        let staged_link_t = t_total.elapsed().as_secs_f32();

        let dataset_128 = {
            let idx = result
                .index
                .as_any_mut()
                .downcast_mut::<diskann::index::InmemIndex<f32, 128>>()
                .expect("downcast");
            std::mem::replace(
                &mut idx.dataset,
                diskann::model::InmemDataset::new(0, 1.0).unwrap(),
            )
        };
        drop(result.index);
        let extract_t = t_total.elapsed().as_secs_f32() - staged_link_t;

        let t_staged = Instant::now();
        let _ = StagedDiskANN::<128>::new(
            dataset_128,
            &result.partitions,
            result.entry_point,
            32, // max_degree
            4,  // max_extra
            None,
            None,
            None,
            false,
        );
        let staged_t = t_staged.elapsed().as_secs_f32();
        let total_t = t_total.elapsed().as_secs_f32();
        let cs_overhead = staged_link_t - diskann_link_t;

        vamana_times.push(diskann_link_t);
        overhead_times.push(staged_t + cs_overhead + extract_t);
        total_times.push(total_t);

        println!("  DiskANN link (no CS):     {diskann_link_t:.3}s");
        println!(
            "  Staged link (with CS):    {staged_link_t:.3}s  (CS overhead: {cs_overhead:+.3}s)"
        );
        println!("  Extract:                  {extract_t:.3}s");
        println!("  Clustering+reorder:       {staged_t:.3}s");
        println!(
            "  Total extra overhead:     {:.3}s  ({:.1}% of DiskANN link)",
            total_t - diskann_link_t,
            (total_t - diskann_link_t) / diskann_link_t * 100.0,
        );
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let stddev = |v: &[f32], m: f32| {
        let variance = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
        variance.sqrt()
    };

    let v_mean = mean(&vamana_times);
    let v_std = stddev(&vamana_times, v_mean);
    let o_mean = mean(&overhead_times);
    let o_std = stddev(&overhead_times, o_mean);
    let t_mean = mean(&total_times);
    let t_std = stddev(&total_times, t_mean);

    println!("\n=== Build Profile Summary ({num_points} points, dim={dimension}) ===\n");
    println!(
        "  {:<35} {:.3}s ± {:.3}s",
        "Vamana graph build:", v_mean, v_std
    );
    println!(
        "  {:<35} {:.3}s ± {:.3}s",
        "StagedDiskANN overhead:", o_mean, o_std
    );
    println!(
        "  {:<35} {:.3}s ± {:.3}s",
        "Total StagedDiskANN build:", t_mean, t_std
    );
    println!(
        "\n  StagedDiskANN overhead ratio: {:.1}% of Vamana build time",
        o_mean / v_mean * 100.0
    );
}

fn run_search_profile(dataset: &Dataset, k: usize) {
    use staged_diskann::{
        build_diskann_index, StagedDiskANN, DEFAULT_EARLY_EXIT_LIMIT, DIM_100, DIM_128, DIM_32,
        DIM_960,
    };

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let scfg = load_staged_config(dimension);

    macro_rules! profile {
        ($N:literal) => {{
            let alpha = scfg.alpha;
            println!("Building StagedDiskANN ({}-dim, alpha={})...", $N, alpha);
            let result = build_diskann_index(
                &flat_base, num_points, dimension, alpha, scfg.graph_degree, scfg.build_search_list_size as u32, false, None, None, true, scfg.max_extra,
            )
            .expect("build failed");
            drop(result.index);

            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged = StagedDiskANN::<$N>::new(
                empty_ds,
                &result.partitions,
                result.entry_point,
                32,
                4,
                None,
                None,
                None,
                false,
            );
            staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

            let queries: Vec<[f32; $N]> = dataset
                .queries
                .iter()
                .map(|q| {
                    let mut arr = [0f32; $N];
                    arr.copy_from_slice(&q[..$N]);
                    arr
                })
                .collect();

            println!("Warm-up (1000 queries)...");
            for q in &queries[..queries.len().min(1000)] {
                staged
                    .search(q, k, 48, 5, 0.15, DEFAULT_EARLY_EXIT_LIMIT)
                    .ok();
            }

            // Early-exit sweep: simulate search with different early_exit_count.
            // Uses search_diag which returns (results, converge_step, total_steps, p1_ndc, p2_ndc).
            println!(
                "\n─── Early Exit Sweep ({} queries, L=48, ws=5, thr=0.15) ───\n",
                queries.len()
            );
            println!(
                "  {:<12} {:<12} {:<14} {:<10} {:<10}",
                "EarlyExit", "Iter/query", "DistCalls/q", "R@10", "QPS_est"
            );
            println!("  {}", "─".repeat(60));

            let ws = 5usize;

            // Auto-calibrate.
            let calib_sample: Vec<[f32; $N]> = queries[..queries.len().min(200)].to_vec();
            let calib = staged
                .calibrate(&calib_sample, 48, ws)
                .expect("calibrate failed");
            println!(
                "  Calibrated: threshold={:.2}, early_exit_limit={}",
                calib.threshold, calib.early_exit_limit
            );

            let thr = calib.threshold;
            let ee = calib.early_exit_limit;

            for &sls in &[48usize, 100, 200] {
                println!(
                    "\n─── L={}, ws={}, thr={:.2}, ee={} ───\n",
                    sls, ws, thr, ee
                );
                println!(
                    "  {:<12} {:<12} {:<14} {:<10} {:<10}",
                    "Mode", "Iter/query", "DistCalls/q", "R@10", "QPS_est"
                );
                println!("  {}", "─".repeat(60));

                // Baseline: no convergence, no early exit
                {
                    let mut total_iter = 0u64;
                    let mut total_dist = 0u64;
                    let mut recall_sum = 0.0f64;
                    let t_start = std::time::Instant::now();
                    for (qi, q) in queries.iter().enumerate() {
                        let (res, _conv, steps, p1, p2) =
                            staged.search_diag(q, k, sls, ws, 0.0, 0).unwrap();
                        total_iter += steps as u64;
                        total_dist += (p1 + p2) as u64;
                        let gt = &dataset.ground_truth[qi];
                        let hits = res
                            .iter()
                            .take(k)
                            .filter(|r| gt.iter().take(k).any(|g| g == *r))
                            .count();
                        recall_sum += hits as f64 / k as f64;
                    }
                    let wall = t_start.elapsed();
                    let nq = queries.len() as f64;
                    let qps = nq / wall.as_secs_f64();
                    println!(
                        "  {:<12} {:<12.1} {:<14.1} {:<10.4} {:<10.0}",
                        "none(DA)",
                        total_iter as f64 / nq,
                        total_dist as f64 / nq,
                        recall_sum / nq,
                        qps
                    );
                }

                // Staged with early exit = 0 (no early exit, but with convergence)
                {
                    let mut total_iter = 0u64;
                    let mut total_dist = 0u64;
                    let mut recall_sum = 0.0f64;
                    let t_start = std::time::Instant::now();
                    for (qi, q) in queries.iter().enumerate() {
                        let (res, _conv, steps, p1, p2) =
                            staged.search_diag(q, k, sls, ws, thr, ee).unwrap();
                        total_iter += steps as u64;
                        total_dist += (p1 + p2) as u64;
                        let gt = &dataset.ground_truth[qi];
                        let hits = res
                            .iter()
                            .take(k)
                            .filter(|r| gt.iter().take(k).any(|g| g == *r))
                            .count();
                        recall_sum += hits as f64 / k as f64;
                    }
                    let wall = t_start.elapsed();
                    let nq = queries.len() as f64;
                    let qps = nq / wall.as_secs_f64();
                    println!(
                        "  {:<12} {:<12.1} {:<14.1} {:<10.4} {:<10.0}",
                        "staged",
                        total_iter as f64 / nq,
                        total_dist as f64 / nq,
                        recall_sum / nq,
                        qps
                    );
                }
            } // end for sls
        }};
    }

    match dimension {
        DIM_32 => profile!(32),
        DIM_100 => profile!(100),
        DIM_128 => profile!(128),
        DIM_960 => profile!(960),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}

fn run_memory_profile(dataset: &Dataset) {
    use diskann::index::{create_inmem_index, ANNInmemIndex};
    use diskann::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
    use diskann::model::{IndexConfiguration, InmemDataset};
    use staged_diskann::{StagedDiskANN, DIM_128};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128);
    let flat_base = dataset.base_flat();

    let fmt = |b: usize| -> String { metrics::memory::format_bytes(b) };

    // Free queries and ground truth — they are benchmark infrastructure,
    // not part of index build/search memory.
    let queries_mem = std::mem::size_of_val(dataset.queries.as_slice())
        + dataset
            .queries
            .iter()
            .map(|q| q.capacity() * 4)
            .sum::<usize>();
    let gt_mem = std::mem::size_of_val(dataset.ground_truth.as_slice())
        + dataset
            .ground_truth
            .iter()
            .map(|g| g.capacity() * 4)
            .sum::<usize>();

    // Memory occupied by queries + ground truth (benchmark infrastructure).
    let bench_overhead = queries_mem + gt_mem;

    ALLOCATOR.reset_peak();
    let baseline = ALLOCATOR.current_bytes();

    println!("\n═══ Memory Profile (SIFT {num_points}) ═══\n");
    println!(
        "  Baseline (base vectors + queries + GT): {}",
        fmt(baseline)
    );
    println!(
        "  Queries + GT overhead:                  {} (subtracted in index-only column)\n",
        fmt(bench_overhead)
    );

    println!(
        "  {:<50} {:>12} {:>12} {:>14}",
        "Step", "Current", "Peak", "Index-Only Cur"
    );
    println!("  {}", "─".repeat(90));

    let m = |label: &str| {
        let cur = ALLOCATOR.current_bytes();
        let peak = ALLOCATOR.peak_bytes();
        let idx_only = cur.saturating_sub(baseline);
        println!(
            "  {:<50} {:>12} {:>12} {:>14}",
            label,
            fmt(cur),
            fmt(peak),
            fmt(idx_only),
        );
    };

    // ── DiskANN baseline (no candidate sets) ─────────────────────────
    println!("  ── DiskANN (compute_candidate_sets=false) ──");
    m("D0. baseline");

    let num_threads = rayon::current_num_threads() as u32;
    {
        let wp = IndexWriteParametersBuilder::new(48, 32)
            .with_alpha(1.2)
            .with_num_threads(num_threads)
            .with_compute_candidate_sets(false)
            .build();
        let config = IndexConfiguration::new(
            vector::Metric::L2,
            dimension,
            dimension,
            num_points,
            false,
            0,
            false,
            0,
            1.0,
            wp,
        );
        let mut idx_d: Box<dyn ANNInmemIndex<f32>> =
            create_inmem_index::<f32>(config).expect("create index");
        m("D1. create_inmem_index (no CS)");

        idx_d
            .build_from_data(&flat_base, num_points)
            .expect("build");
        m("D2. build_from_data (no CS)");
    }
    let diskann_peak = ALLOCATOR.peak_bytes();
    // idx_d dropped
    m("D3. drop(InmemIndex)");

    // ── StagedDiskANN (with candidate sets) ──────────────────────────
    ALLOCATOR.reset_peak();
    println!("\n  ── StagedDiskANN (compute_candidate_sets=true) ──");
    m("S0. baseline (after DiskANN freed)");

    let wp = IndexWriteParametersBuilder::new(48, 32)
        .with_alpha(1.2)
        .with_num_threads(num_threads)
        .with_compute_candidate_sets(true)
        .build();
    let config = IndexConfiguration::new(
        vector::Metric::L2,
        dimension,
        dimension,
        num_points,
        false,
        0,
        false,
        0,
        1.0,
        wp,
    );
    let mut idx: Box<dyn ANNInmemIndex<f32>> =
        create_inmem_index::<f32>(config).expect("create index");
    m("S1. create_inmem_index (with CS)");

    idx.build_from_data(&flat_base, num_points).expect("build");
    m("S2. build_from_data (Vamana + anchor sets)");

    let entry_point = idx.start_node();

    ALLOCATOR.reset_peak();
    let peak_before_extract = ALLOCATOR.current_bytes();
    let partitions = idx.extract_graph_and_candidates(16).expect("extract");
    let peak_during_extract = ALLOCATOR.peak_bytes();
    m("S3. extract_graph_and_candidates");
    println!(
        "     ↳ peak DURING extract:       {} (index-only: {})",
        fmt(peak_during_extract),
        fmt(peak_during_extract.saturating_sub(baseline))
    );
    println!(
        "     ↳ delta from pre-extract:     {}",
        fmt(peak_during_extract - peak_before_extract)
    );

    m("S4. (candidate_sets ready)");

    drop(idx);
    m("S5. drop(InmemIndex)");

    ALLOCATOR.reset_peak();
    let peak_before_staged = ALLOCATOR.current_bytes();
    let empty_ds = InmemDataset::<f32, 128>::new(0, 1.0).unwrap();
    let mut _staged = StagedDiskANN::<128>::new(
        empty_ds,
        &partitions,
        entry_point,
        32, // max_degree
        4,  // max_extra
        None,
        None,
        None,
        false,
    );
    let peak_during_staged = ALLOCATOR.peak_bytes();
    m("S6. StagedDiskANN::new (PhasedGraph build)");
    println!(
        "     ↳ peak DURING staged:        {} (index-only: {})",
        fmt(peak_during_staged),
        fmt(peak_during_staged.saturating_sub(baseline))
    );
    println!(
        "     ↳ delta from pre-staged:      {}",
        fmt(peak_during_staged - peak_before_staged)
    );

    // Rebuild dataset AFTER build completes.
    _staged.dataset = rebuild_dataset::<128>(&flat_base, num_points);
    m("S7. rebuild dataset (post-build)");

    println!("\n  {}", "─".repeat(90));
    m("FINAL");

    // ── Summary: index-only view ──
    let final_cur = ALLOCATOR.current_bytes();
    println!(
        "\n  ── Index-Only Summary (baseline {} subtracted) ──",
        fmt(baseline)
    );
    println!(
        "  DiskANN peak (index-only):      {}",
        fmt(diskann_peak.saturating_sub(baseline))
    );
    println!(
        "  StagedDiskANN peak (index-only): {}",
        fmt(peak_during_extract.saturating_sub(baseline))
    );
    println!(
        "  StagedDiskANN final (index-only):{}",
        fmt(final_cur.saturating_sub(baseline))
    );
    println!(
        "  Peak delta (Staged - DiskANN):   {}",
        fmt(peak_during_extract.saturating_sub(diskann_peak))
    );
    println!();
}

fn run_qps_recall_sweep(dataset: &Dataset, k: usize) {
    use rayon::prelude::*;
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();

    // Load config.
    #[derive(serde::Deserialize)]
    struct SweepConfig {
        defaults: DefaultsConfig,
        datasets: std::collections::HashMap<String, DatasetConfig>,
    }
    #[derive(serde::Deserialize)]
    struct DefaultsConfig {
        diskann: DiskANNDefaults,
        staged: StagedDefaults,
        sweep: SweepDefaults,
    }
    #[derive(serde::Deserialize)]
    struct DiskANNDefaults {
        alpha: f32,
        graph_degree: u32,
        build_search_list_size: usize,
    }
    #[derive(serde::Deserialize)]
    struct StagedDefaults {
        alpha: f32,
        graph_degree: u32,
        build_search_list_size: usize,
        max_extra: usize,
        window_size: usize,
    }
    #[derive(serde::Deserialize)]
    struct SweepDefaults {
        search_list_sizes: Vec<usize>,
        threads: usize,
        trials: usize,
    }
    #[derive(serde::Deserialize, Default, Clone)]
    struct DatasetConfig {
        #[allow(dead_code)]
        dimension: Option<usize>,
        staged: Option<DatasetStagedOverride>,
    }
    #[derive(serde::Deserialize, Default, Clone)]
    struct DatasetStagedOverride {
        alpha: Option<f32>,
        build_search_list_size: Option<usize>,
    }

    let config_path = "benchmark/configs/sweep.yaml";
    let cfg: SweepConfig = serde_yaml::from_str(
        &std::fs::read_to_string(config_path)
            .unwrap_or_else(|_| panic!("Cannot read {config_path}")),
    )
    .expect("Invalid sweep config YAML");

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };
    let ds_cfg = cfg.datasets.get(dim_name).cloned().unwrap_or_default();
    let staged_alpha = ds_cfg
        .staged
        .as_ref()
        .and_then(|s| s.alpha)
        .unwrap_or(cfg.defaults.staged.alpha);
    let staged_build_l = ds_cfg
        .staged
        .as_ref()
        .and_then(|s| s.build_search_list_size)
        .unwrap_or(cfg.defaults.staged.build_search_list_size);
    let num_threads = cfg.defaults.sweep.threads;
    let trials = cfg.defaults.sweep.trials;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let search_list_sizes = &cfg.defaults.sweep.search_list_sizes;
    let staged_ws = cfg.defaults.staged.window_size;
    let da_alpha = cfg.defaults.diskann.alpha;
    let da_degree = cfg.defaults.diskann.graph_degree;
    let da_build_l = cfg.defaults.diskann.build_search_list_size;
    let st_degree = cfg.defaults.staged.graph_degree;
    let st_max_extra = cfg.defaults.staged.max_extra;

    println!("Config: DiskANN(R={da_degree}, α={da_alpha}) vs StagedDiskANN(R={st_degree}, α={staged_alpha})");

    // Generic sweep: DiskANN via native runner, StagedDiskANN independently.
    macro_rules! run_sweep {
        ($N:literal, $dim_name:expr) => {{
            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            let max_l = *search_list_sizes.last().unwrap();
            println!("Building DiskANN ({}-dim, R={}, α={}, build_L={})...", $N, da_degree, da_alpha, da_build_l);
            let mut diskann_runner = crate::runner::DiskANNRunner::new(da_build_l, da_degree, da_alpha);
            diskann_runner.build(&flat_base, num_points, dimension);
            // Pre-expand scratch on ALL threads to avoid runtime resize.
            if max_l > da_build_l {
                diskann_runner.set_search_list_size(max_l);
                // DiskANN creates 5+num_threads scratch objects; send enough
                // queries to ensure every scratch is expanded.
                let warmup_queries: Vec<Vec<f32>> = (0..num_threads * 4 + 20)
                    .map(|_| vec![0.0f32; dimension])
                    .collect();
                pool.install(|| {
                    warmup_queries.par_iter().for_each(|q| {
                        let _ = diskann_runner.search(q, k);
                    });
                });
            }
            println!("  DiskANN built.");

            // ── StagedDiskANN: R=32, alpha=1.2 (more remote shortcuts) ──
            println!("Building StagedDiskANN ({}-dim, R={}, α={})...", $N, st_degree, staged_alpha);
            let st_result = build_diskann_index(
                &flat_base, num_points, dimension, staged_alpha, st_degree, staged_build_l as u32,
                false, None, None, true, st_max_extra,
            ).expect("build failed");
            let st_entry = st_result.entry_point;
            drop(st_result.index);
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged_idx = StagedDiskANN::<$N>::new(
                empty_ds, &st_result.partitions, st_entry,
                st_degree, st_max_extra, None, None, None, false,
            );
            staged_idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

            let measure_staged = |run: &(dyn Fn() -> Vec<Vec<u32>> + Sync)| -> (f64, f64) {
                let mut qps_samples = Vec::with_capacity(trials);
                let mut recall = 0.0f64;
                for _ in 0..trials {
                    let t = Instant::now();
                    let results = pool.install(|| run());
                    let wall = t.elapsed();
                    qps_samples.push(queries.len() as f64 / wall.as_secs_f64());
                    recall = metrics::recall::mean_recall(&results, &dataset.ground_truth, k);
                }
                qps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (recall, qps_samples[trials / 2])
            };

            let staged_graph_mb = (staged_idx.graph.stride() * staged_idx.graph.num_nodes() * 4) as f64 / 1_048_576.0;
            println!("\n  Memory: DiskANN(R={}, α={}) ≈ {:.1} MB",
                da_degree, da_alpha, num_points as f64 * da_degree as f64 * 4.0 / 1_048_576.0);
            println!("  Memory: StagedDiskANN(R={}, α={}) PhasedGraph = {:.1} MB",
                st_degree, staged_alpha, staged_graph_mb);

            println!("\n═══ QPS vs Recall@10: {} ({num_points} pts, {num_threads} threads, k={k}) ═══\n", $dim_name);

            // DiskANN(R=32): sweep L via native runner.
            println!("DiskANN (R=32):");
            let mut diskann_data: Vec<(f64, f64)> = Vec::new();
            for &sls in search_list_sizes {
                diskann_runner.set_search_list_size(sls);
                let mut qps_samples = Vec::with_capacity(trials);
                let mut recall = 0.0f64;
                for _ in 0..trials {
                    let t = Instant::now();
                    let results: Vec<crate::runner::common::SearchResult> = pool.install(|| {
                        dataset.queries.par_iter().map(|q| diskann_runner.search(q, k)).collect()
                    });
                    let wall = t.elapsed();
                    qps_samples.push(dataset.queries.len() as f64 / wall.as_secs_f64());
                    let ids: Vec<Vec<u32>> = results.into_iter().map(|r| r.neighbors).collect();
                    recall = metrics::recall::mean_recall(&ids, &dataset.ground_truth, k);
                }
                qps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let qps = qps_samples[trials / 2];
                diskann_data.push((recall, qps));
                println!("  L={sls:>4}  R@10={recall:.4}  QPS={qps:.0}");
            }

            // Auto-calibrate convergence params from warmup queries.
            let calib_sample: Vec<[f32; $N]> = queries[..queries.len().min(200)].to_vec();
            let calib = staged_idx.calibrate(&calib_sample, 48, staged_ws)
                .expect("calibrate failed");
            let cal_thr = calib.threshold;
            let cal_ee = calib.early_exit_limit;
            println!("\nCalibrated: threshold={:.2}, early_exit_limit={}", cal_thr, cal_ee);

            println!("StagedDiskANN (R={}, α={}, ws={staged_ws}, thr={cal_thr:.2}, ee={cal_ee}):",
                st_degree, staged_alpha);
            let mut staged_data: Vec<(f64, f64)> = Vec::new();
            for &sls in search_list_sizes {
                let (recall, qps) = measure_staged(&|| {
                    staged_idx.search_batch(&queries, k, sls, staged_ws, cal_thr, cal_ee).unwrap()
                });
                staged_data.push((recall, qps));
                println!("  L={sls:>4}  R@10={recall:.4}  QPS={qps:.0}");
            }

            // Write JSON
            let json_path = format!("visualizations/qps_recall_{}.json", $dim_name);
            std::fs::create_dir_all("visualizations").ok();
            let json = format!(
                "{{\n  \"dataset\": \"{}\",\n  \"dimension\": {},\n  \"num_points\": {},\n  \"threads\": {},\n  \"diskann\": [{}],\n  \"staged\": [{}]\n}}",
                $dim_name, $N, num_points, num_threads,
                diskann_data.iter().map(|(r, q)| format!("[{:.4}, {:.0}]", r, q)).collect::<Vec<_>>().join(", "),
                staged_data.iter().map(|(r, q)| format!("[{:.4}, {:.0}]", r, q)).collect::<Vec<_>>().join(", "),
            );
            std::fs::write(&json_path, &json).expect("write json");
            println!("\nSaved to {json_path}");
        }};
    }

    match dimension {
        DIM_32 => run_sweep!(32, "glove25"),
        DIM_100 => run_sweep!(100, "glove100"),
        DIM_128 => run_sweep!(128, "sift"),
        DIM_960 => run_sweep!(960, "gist"),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}

fn run_graph_stats(_dataset: &Dataset) {
    println!("graph-stats: Use 'qps-recall-sweep' or 'neighbor-contribution' instead.");
}

fn run_cliff_profile(dataset: &Dataset) {
    use staged_diskann::algorithm::analysis::{
        annotate_bf_ranks, compute_cliff_stats, print_node_cliff_detail, summarize_cliff_ranks,
        summarize_cliff_stats,
    };
    use staged_diskann::{build_diskann_index, PhasedGraph};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let dim = dimension;

    // ── Build Vamana graph ──────────────────────────────────────────────
    let alpha = 2.0f32;
    let graph_degree = 64u32;
    let search_list = 100u32;
    println!("Building Vamana graph ({num_points} pts, R={graph_degree}, L={search_list}, alpha={alpha})...");
    let result = build_diskann_index(
        &flat_base,
        num_points,
        dimension,
        alpha,
        graph_degree,
        search_list,
        false,
        None,
        None,
        false,
        16,
    )
    .expect("build failed");
    let phased = PhasedGraph::build_from_partitions(&result.partitions, 16, 4);
    let graph = &phased;

    // ── Phase 1: Full cliff analysis (all nodes, no brute-force) ────────
    println!(
        "\n═══ Cliff Neighbor Analysis (all {} nodes) ═══\n",
        num_points
    );
    let stats = compute_cliff_stats(graph, &flat_base, dim);
    let summary = summarize_cliff_stats(&stats);
    println!("{summary}");

    // ── Gap ratio distribution (bucketed histogram) ─────────────────────
    println!("Gap ratio distribution (all consecutive pairs):");
    let mut all_gaps: Vec<f32> = stats
        .iter()
        .flat_map(|s| s.gap_ratios.iter().copied())
        .collect();
    all_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let buckets = [
        1.0,
        1.05,
        1.1,
        1.2,
        1.3,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        f32::INFINITY,
    ];
    let mut bucket_counts = vec![0usize; buckets.len()];
    for &g in &all_gaps {
        for (bi, &upper) in buckets.iter().enumerate() {
            if g < upper {
                bucket_counts[bi] += 1;
                break;
            }
        }
    }
    let total_gaps = all_gaps.len();
    for (bi, &upper) in buckets.iter().enumerate() {
        let lower = if bi == 0 { 0.0 } else { buckets[bi - 1] };
        let label = if upper.is_infinite() {
            format!("{:.1}+", lower)
        } else {
            format!("{:.2}-{:.2}", lower, upper)
        };
        println!(
            "  {:<12}: {} ({:.1}%)",
            label,
            bucket_counts[bi],
            bucket_counts[bi] as f64 / total_gaps.max(1) as f64 * 100.0
        );
    }

    // ── Phase 2: Brute-force rank annotation (sampled nodes) ────────────
    let num_samples = 500.min(num_points);
    let step = num_points / num_samples;
    let sample_indices: Vec<usize> = (0..num_points)
        .step_by(step.max(1))
        .take(num_samples)
        .collect();

    println!(
        "\n═══ Brute-Force Rank Analysis ({} sampled nodes) ═══\n",
        sample_indices.len()
    );
    println!("  Computing brute-force KNN ranks (max_rank=200)...");

    let mut sampled_stats: Vec<_> = sample_indices.iter().map(|&i| stats[i].clone()).collect();
    annotate_bf_ranks(&mut sampled_stats, &flat_base, dim, num_points, 200);

    let topk_thresholds = vec![5, 10, 15, 20, 30, 50];
    let rank_summary = summarize_cliff_ranks(&sampled_stats, &topk_thresholds);
    println!("{rank_summary}");

    // ── Phase 3: Detailed examples ──────────────────────────────────────
    // Show nodes with extreme cliff ratios.
    println!("═══ Example Nodes ═══\n");

    // Top-5 by cliff ratio.
    let mut by_ratio: Vec<usize> = (0..sampled_stats.len()).collect();
    by_ratio.sort_by(|&a, &b| {
        sampled_stats[b]
            .cliff_ratio
            .partial_cmp(&sampled_stats[a].cliff_ratio)
            .unwrap()
    });
    println!("── Top-5 nodes by cliff ratio (sharpest cliffs) ──\n");
    for &idx in by_ratio.iter().take(5) {
        print_node_cliff_detail(&sampled_stats[idx]);
        println!();
    }

    // Top-5 by earliest cliff position (cliff_pos = 0 means cliff between
    // nearest and second-nearest, which is very interesting).
    let mut by_early: Vec<usize> = (0..sampled_stats.len())
        .filter(|&i| sampled_stats[i].degree > 1)
        .collect();
    by_early.sort_by(|&a, &b| {
        sampled_stats[a]
            .cliff_pos
            .cmp(&sampled_stats[b].cliff_pos)
            .then(
                sampled_stats[b]
                    .cliff_ratio
                    .partial_cmp(&sampled_stats[a].cliff_ratio)
                    .unwrap(),
            )
    });
    println!("── Top-5 nodes with earliest cliff (pos=0 or 1) ──\n");
    for &idx in by_early.iter().take(5) {
        print_node_cliff_detail(&sampled_stats[idx]);
        println!();
    }

    // ── Phase 4: Per-position average gap ratio ─────────────────────────
    // For each position i in the sorted neighbor list, what's the average gap ratio?
    let max_degree = stats.iter().map(|s| s.degree).max().unwrap_or(0);
    let mut pos_sum = vec![0.0f64; max_degree];
    let mut pos_cnt = vec![0usize; max_degree];
    for s in &stats {
        for (i, &r) in s.gap_ratios.iter().enumerate() {
            pos_sum[i] += r as f64;
            pos_cnt[i] += 1;
        }
    }
    println!("═══ Average Gap Ratio by Position ═══\n");
    println!("  {:<6} {:<12} {:<10}", "Pos", "Avg Ratio", "Count");
    println!("  {}", "─".repeat(30));
    for i in 0..max_degree.min(40) {
        if pos_cnt[i] > 0 {
            println!(
                "  {:<6} {:<12.4} {:<10}",
                i,
                pos_sum[i] / pos_cnt[i] as f64,
                pos_cnt[i]
            );
        }
    }

    // ── Phase 5: Rank-Precision Cliff ───────────────────────────────────
    use staged_diskann::algorithm::analysis::{
        compute_rank_precision_profiles, sort_neighbors_by_distance, summarize_rank_precision,
    };

    println!(
        "\n═══ Rank-Precision Cliff ({} sampled nodes) ═══\n",
        sample_indices.len()
    );

    let sorted_adj = sort_neighbors_by_distance(graph, &flat_base, dim);

    // BF top-K already computed as bf_knn in the BF rank section — recompute
    // for the same sample with depth = max_degree.
    let bf_max = max_degree.min(200);
    println!("  Computing brute-force top-{bf_max} for rank-precision...");
    let l2 = |a: usize, b: usize| -> f32 {
        let pa = &flat_base[a * dim..(a + 1) * dim];
        let pb = &flat_base[b * dim..(b + 1) * dim];
        pa.iter().zip(pb).map(|(x, y)| (x - y) * (x - y)).sum()
    };
    let bf_knn_rp: Vec<Vec<u32>> = sample_indices
        .iter()
        .map(|&u| {
            let mut dists: Vec<(u32, f32)> = (0..num_points)
                .filter(|&i| i != u)
                .map(|i| (i as u32, l2(u, i)))
                .collect();
            dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(bf_max).map(|&(id, _)| id).collect()
        })
        .collect();

    let profiles = compute_rank_precision_profiles(&sorted_adj, &sample_indices, &bf_knn_rp);
    let summary = summarize_rank_precision(&profiles);
    println!("{summary}");
}

fn run_neighbor_contribution_profile(dataset: &Dataset) {
    use diskann::model::{Neighbor as DNeighbor, Vertex};
    use staged_diskann::algorithm::search::convergence::SearchConvergenceChecker;
    use staged_diskann::{build_diskann_index, StagedDiskANN};
    use vector::Metric;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let _k = 10;
    let scfg = load_staged_config(dimension);

    macro_rules! run_profile {
        ($N:literal) => {{
            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            println!("Building StagedDiskANN ({}-dim, alpha={})...", $N, scfg.alpha);
            let result = build_diskann_index(
                &flat_base, num_points, dimension, scfg.alpha, scfg.graph_degree,
                scfg.build_search_list_size as u32, false, None, None, true, scfg.max_extra,
            ).expect("build failed");
            let entry = result.entry_point;
            drop(result.index);
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged = StagedDiskANN::<$N>::new(
                empty_ds, &result.partitions, entry,
                scfg.graph_degree, scfg.max_extra, None, None, None, false,
            );
            staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

            let graph = &staged.graph;
            let ds = &staged.dataset;

            // Per-zone counters split by phase.
            // Zones: local (0..local_count), remote (local_count..degree), extra
            // Phases: navigation (pre-convergence), reranking (post-convergence)
            #[derive(Default)]
            struct ZoneStats {
                visited: u64,
                unseen: u64,
                admitted: u64,
            }

            let search_list_sizes = [32, 48, 64, 100];
            let ws = 5;
            let eps = 0.01f32;

            for &sls in &search_list_sizes {
                let mut nav_local = ZoneStats::default();
                let mut nav_remote = ZoneStats::default();
                let mut rerank_local = ZoneStats::default();
                let mut rerank_remote = ZoneStats::default();
                let mut rerank_extra = ZoneStats::default();
                let mut total_nav_steps = 0u64;
                let mut total_rerank_steps = 0u64;

                for query in &queries {
                    let query_vertex = Vertex::new(query, 0);
                    let mut dcc = SearchConvergenceChecker::new(ws, eps);
                    let mut seen = vec![false; num_points];
                    let mut pq = diskann::model::NeighborPriorityQueue::with_capacity(sls);

                    seen[entry as usize] = true;
                    let ed = ds.get_vertex(entry).unwrap().compare(&query_vertex, Metric::L2);
                    pq.insert(DNeighbor::new(entry, ed));

                    let mut prev_admitted: usize = 1; // optimistic start

                    while pq.has_notvisited_node() {
                        let cur = pq.closest_notvisited();
                        let id = cur.id as usize;
                        let converged = dcc.update(prev_admitted);

                        let pq_worst = if pq.size() >= sls { pq[pq.size() - 1].distance } else { f32::MAX };

                        let mut step_admitted = 0usize;

                        if !converged {
                            total_nav_steps += 1;
                            // Navigation: expand all neighbors, tag zone.
                            let local = graph.local_neighbors(id);
                            let remote = graph.remote_neighbors(id);

                            for &nn in local {
                                nav_local.visited += 1;
                                if !seen[nn as usize] {
                                    seen[nn as usize] = true;
                                    nav_local.unseen += 1;
                                    let d = ds.get_vertex(nn).unwrap().compare(&query_vertex, Metric::L2);
                                    if d < pq_worst || pq.size() < sls {
                                        nav_local.admitted += 1;
                                        step_admitted += 1;
                                    }
                                    pq.insert(DNeighbor::new(nn, d));
                                }
                            }
                            for &nn in remote {
                                nav_remote.visited += 1;
                                if !seen[nn as usize] {
                                    seen[nn as usize] = true;
                                    nav_remote.unseen += 1;
                                    let d = ds.get_vertex(nn).unwrap().compare(&query_vertex, Metric::L2);
                                    if d < pq_worst || pq.size() < sls {
                                        nav_remote.admitted += 1;
                                        step_admitted += 1;
                                    }
                                    pq.insert(DNeighbor::new(nn, d));
                                }
                            }
                        } else {
                            total_rerank_steps += 1;
                            // Reranking: expand local + remote + extra, track all zones.
                            let local = graph.local_neighbors(id);
                            let remote = graph.remote_neighbors(id);
                            let extra = graph.extra_candidates(id);

                            for &nn in local {
                                rerank_local.visited += 1;
                                if !seen[nn as usize] {
                                    seen[nn as usize] = true;
                                    rerank_local.unseen += 1;
                                    let d = ds.get_vertex(nn).unwrap().compare(&query_vertex, Metric::L2);
                                    if d < pq_worst || pq.size() < sls {
                                        rerank_local.admitted += 1;
                                        step_admitted += 1;
                                    }
                                    pq.insert(DNeighbor::new(nn, d));
                                }
                            }
                            for &nn in remote {
                                rerank_remote.visited += 1;
                                if !seen[nn as usize] {
                                    seen[nn as usize] = true;
                                    rerank_remote.unseen += 1;
                                    let d = ds.get_vertex(nn).unwrap().compare(&query_vertex, Metric::L2);
                                    if d < pq_worst || pq.size() < sls {
                                        rerank_remote.admitted += 1;
                                        step_admitted += 1;
                                    }
                                    pq.insert(DNeighbor::new(nn, d));
                                }
                            }
                            for &nn in extra {
                                rerank_extra.visited += 1;
                                if !seen[nn as usize] {
                                    seen[nn as usize] = true;
                                    rerank_extra.unseen += 1;
                                    let d = ds.get_vertex(nn).unwrap().compare(&query_vertex, Metric::L2);
                                    if d < pq_worst || pq.size() < sls {
                                        rerank_extra.admitted += 1;
                                        step_admitted += 1;
                                    }
                                    pq.insert(DNeighbor::new(nn, d));
                                }
                            }
                        }

                        prev_admitted = step_admitted;
                    }
                }

                let nq = queries.len() as f64;
                let pct = |a: u64, b: u64| if b > 0 { a as f64 / b as f64 * 100.0 } else { 0.0 };

                println!("\n═══ Neighbor Contribution (L={sls}, ws={ws}, ε={eps}) ═══");
                println!("  Avg nav steps:    {:.1}    Avg rerank steps: {:.1}",
                    total_nav_steps as f64 / nq, total_rerank_steps as f64 / nq);
                println!("\n  {:<20} {:<10} {:<10} {:<10} {:<12}", "Zone", "Visited", "Unseen", "Admitted", "Admit/Unseen");
                println!("  {}", "─".repeat(64));
                let rows: &[(&str, &ZoneStats)] = &[
                    ("nav:local",    &nav_local),
                    ("nav:remote",   &nav_remote),
                    ("rerank:local", &rerank_local),
                    ("rerank:remote",&rerank_remote),
                    ("rerank:extra", &rerank_extra),
                ];
                for &(name, s) in rows {
                    println!("  {:<20} {:<10.1} {:<10.1} {:<10.1} {:<12.1}%",
                        name,
                        s.visited as f64 / nq,
                        s.unseen as f64 / nq,
                        s.admitted as f64 / nq,
                        pct(s.admitted, s.unseen));
                }

                // What if we also expanded remote during reranking? (counterfactual)
                // Show nav:remote admission rate as proxy for its value post-convergence.
                let total_nav = nav_local.admitted + nav_remote.admitted;
                let remote_share = if total_nav > 0 { nav_remote.admitted as f64 / total_nav as f64 * 100.0 } else { 0.0 };
                println!("\n  Remote share of nav admissions: {:.1}%", remote_share);
            }

            // ── Bidir edge distribution: per-neighbor-rank bidir rate ──
            // For each rank position r (0..max_degree), compute what fraction
            // of nodes have a bidirectional edge at rank r.
            // Bidir = neighbor is in local zone (rank < local_count).
            let max_deg = scfg.graph_degree as usize;
            let mut bidir_at_rank = vec![0u64; max_deg];
            let mut total_at_rank = vec![0u64; max_deg];
            for i in 0..num_points {
                let deg = graph.neighbors(i).len();
                let lc = graph.local_count(i);
                for r in 0..deg.min(max_deg) {
                    total_at_rank[r] += 1;
                    if r < lc {
                        bidir_at_rank[r] += 1;
                    }
                }
            }
            let bidir_rate_by_rank: Vec<f64> = (0..max_deg)
                .map(|r| {
                    if total_at_rank[r] > 0 {
                        bidir_at_rank[r] as f64 / total_at_rank[r] as f64
                    } else {
                        0.0
                    }
                })
                .collect();

            // Per-node bidir fraction histogram
            let bidir_fractions: Vec<f64> = (0..num_points)
                .map(|i| {
                    let deg = graph.neighbors(i).len();
                    if deg > 0 { graph.local_count(i) as f64 / deg as f64 } else { 0.0 }
                })
                .collect();

            let dim_name = match $N {
                32 => "glove25", 100 => "glove100", 128 => "sift", 960 => "gist", _ => "unknown",
            };
            let bidir_json = serde_json::json!({
                "dataset": dim_name,
                "dimension": $N,
                "alpha": scfg.alpha,
                "num_points": num_points,
                "bidir_rate_by_rank": bidir_rate_by_rank,
                "bidir_fractions": bidir_fractions,
            });
            let bidir_path = format!("visualizations/bidir_distribution_{}.json", dim_name);
            std::fs::write(&bidir_path, serde_json::to_string(&bidir_json).unwrap())
                .expect("write bidir json");
            println!("\nSaved {bidir_path}");
        }};
    }

    match dimension {
        32 => run_profile!(32),
        100 => run_profile!(100),
        128 => run_profile!(128),
        960 => run_profile!(960),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}

fn run_extra_profile(dataset: &Dataset, k: usize) {
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
    let scfg = load_staged_config(dataset.dimension);

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();

    macro_rules! run_extra {
        ($N:literal) => {{
            let queries: Vec<[f32; $N]> = dataset
                .queries
                .iter()
                .map(|q| {
                    let mut arr = [0f32; $N];
                    arr.copy_from_slice(&q[..$N]);
                    arr
                })
                .collect();

            let alpha = scfg.alpha;

            // Sweep max_extra = 0, 2, 4, 8, 16, 32.
            println!(
                "\n═══ Extra Candidate Profile ({}-dim, alpha={}, {} queries) ═══\n",
                $N,
                alpha,
                queries.len()
            );
            println!(
                "  {:<10} {:<10} {:<10} {:<12} {:<14} {:<10} {:<10}",
                "MaxExtra", "AvgExtra", "AvgLocal", "Iter/query", "DistCalls/q", "R@10", "QPS"
            );
            println!("  {}", "─".repeat(78));

            for &max_extra in &[0usize, 2, 4, 8, 16, 32] {
                let result = build_diskann_index(
                    &flat_base, num_points, dimension, alpha, 32, 48, false, None, None, true,
                    max_extra,
                )
                .expect("build failed");
                let entry = result.entry_point;
                drop(result.index);

                let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                let mut staged = StagedDiskANN::<$N>::new(
                    empty_ds,
                    &result.partitions,
                    entry,
                    32,
                    max_extra,
                    None,
                    None,
                    None,
                    false,
                );
                staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

                let avg_extra = (0..num_points)
                    .map(|i| staged.graph.extra_count(i))
                    .sum::<usize>() as f64
                    / num_points as f64;
                let avg_local = (0..num_points)
                    .map(|i| staged.graph.local_count(i))
                    .sum::<usize>() as f64
                    / num_points as f64;

                // Calibrate and search.
                let calib = staged
                    .calibrate(&queries[..queries.len().min(200)], 48, 5)
                    .expect("calibrate");

                // Run search_diag for detailed stats.
                let mut total_iter = 0u64;
                let mut total_dist = 0u64;
                let mut recall_sum = 0.0f64;
                let t_start = std::time::Instant::now();
                for (qi, q) in queries.iter().enumerate() {
                    let (res, _, steps, p1, p2) = staged
                        .search_diag(q, k, 48, 5, calib.threshold, calib.early_exit_limit)
                        .unwrap();
                    total_iter += steps as u64;
                    total_dist += (p1 + p2) as u64;
                    let gt = &dataset.ground_truth[qi];
                    let hits = res
                        .iter()
                        .take(k)
                        .filter(|r| gt.iter().take(k).any(|g| g == *r))
                        .count();
                    recall_sum += hits as f64 / k as f64;
                }
                let wall = t_start.elapsed();
                let nq = queries.len() as f64;
                let qps = nq / wall.as_secs_f64();

                println!(
                    "  {:<10} {:<10.1} {:<10.1} {:<12.1} {:<14.1} {:<10.4} {:<10.0}",
                    max_extra,
                    avg_extra,
                    avg_local,
                    total_iter as f64 / nq,
                    total_dist as f64 / nq,
                    recall_sum / nq,
                    qps
                );
            }

            // Show brute-force top-k coverage by extras (using the last built index).
            // The last sweep iteration (max_extra=32) has the most extras.
            println!("\n  Extra coverage of true top-10 (brute-force, 200 sampled nodes):");
            // Rebuild with max_extra=32 for coverage check.
            let result_cov = build_diskann_index(
                &flat_base, num_points, dimension, alpha, 32, 48, false, None, None, true, 32,
            )
            .expect("build failed");
            drop(result_cov.index);
            let pg =
                staged_diskann::PhasedGraph::build_from_partitions(&result_cov.partitions, 32, 32);

            let dim = $N;
            let l2 = |a: usize, b: usize| -> f32 {
                let pa = &flat_base[a * dim..(a + 1) * dim];
                let pb = &flat_base[b * dim..(b + 1) * dim];
                pa.iter()
                    .zip(pb)
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
            };

            let step = (num_points / 200).max(1);
            let samples: Vec<usize> = (0..num_points).step_by(step).take(200).collect();
            let mut extra_hits = 0u64;
            let mut local_hits = 0u64;
            let mut remote_hits = 0u64;
            let mut total_topk = 0u64;

            for &node in &samples {
                let mut dists: Vec<(u32, f32)> = (0..num_points)
                    .filter(|&i| i != node)
                    .map(|i| (i as u32, l2(node, i)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let extras = pg.extra_candidates(node);
                let locals = pg.local_neighbors(node);
                let remotes = pg.remote_neighbors(node);
                let top10 = &dists[..10.min(dists.len())];
                for &(id, _) in top10 {
                    if extras.contains(&id) {
                        extra_hits += 1;
                    }
                    if locals.contains(&id) {
                        local_hits += 1;
                    }
                    if remotes.contains(&id) {
                        remote_hits += 1;
                    }
                    total_topk += 1;
                }
            }

            // 1-hop reachability: for true top-10 NOT in graph/extras,
            // check if reachable as a neighbor of an extra candidate.
            let mut extra_1hop_hits = 0u64;
            for &node in &samples {
                let mut dists: Vec<(u32, f32)> = (0..num_points)
                    .filter(|&i| i != node)
                    .map(|i| (i as u32, l2(node, i)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let extras = pg.extra_candidates(node);
                let all_nbrs = pg.neighbors(node);
                let top10 = &dists[..10.min(dists.len())];

                let mut extra_1hop: Vec<u32> = Vec::new();
                for &e in extras {
                    for &nn in pg.neighbors(e as usize) {
                        extra_1hop.push(nn);
                    }
                }
                extra_1hop.sort_unstable();
                extra_1hop.dedup();

                for &(id, _) in top10 {
                    if !all_nbrs.contains(&id) && !extras.contains(&id) {
                        if extra_1hop.contains(&id) {
                            extra_1hop_hits += 1;
                        }
                    }
                }
            }

            let pct = |h: u64| h as f64 / total_topk.max(1) as f64 * 100.0;
            println!("    local:         {:.1}% of true top-10", pct(local_hits));
            println!("    remote:        {:.1}% of true top-10", pct(remote_hits));
            println!("    extra direct:  {:.1}% of true top-10", pct(extra_hits));
            println!(
                "    extra 1-hop:   {:.1}% reachable via extra's neighbors",
                pct(extra_1hop_hits)
            );
            println!(
                "    unreachable:   {:.1}% not reachable",
                pct(total_topk - extra_hits - local_hits - remote_hits - extra_1hop_hits)
            );
        }};
    }

    match dimension {
        DIM_32 => run_extra!(32),
        DIM_100 => run_extra!(100),
        DIM_128 => run_extra!(128),
        DIM_960 => run_extra!(960),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}

fn run_calibration_diag(dataset: &Dataset) {
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let scfg = load_staged_config(dimension);

    let dim_name = match dimension {
        32 => "glove25", 100 => "glove100", 128 => "sift", 960 => "gist",
        _ => "unknown",
    };

    macro_rules! run_diag {
        ($N:literal) => {{
            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            // Build StagedDiskANN
            let t_build = Instant::now();
            let result = build_diskann_index(
                &flat_base, num_points, dimension, scfg.alpha,
                scfg.graph_degree, scfg.build_search_list_size as u32,
                false, None, None, true, scfg.max_extra,
            ).expect("build failed");
            let graph_build_s = result.graph_build_time.as_secs_f64();
            drop(result.index);

            let t_staged = Instant::now();
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged = StagedDiskANN::<$N>::new(
                empty_ds, &result.partitions, result.entry_point,
                scfg.graph_degree, scfg.max_extra, None, None, None, false,
            );
            staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
            let staged_overhead_s = t_staged.elapsed().as_secs_f64();
            let total_build_s = t_build.elapsed().as_secs_f64();

            // Also build vanilla DiskANN for comparison
            let t_da = Instant::now();
            let _da_result = build_diskann_index(
                &flat_base, num_points, dimension, 2.0, 32, 48,
                false, None, None, false, 0,
            ).expect("diskann build failed");
            let diskann_build_s = t_da.elapsed().as_secs_f64();

            // Calibrate with diagnostics
            let calib_qs: Vec<[f32; $N]> = queries[..queries.len().min(500)].to_vec();
            let diag = staged.calibrate_with_diagnostics(&calib_qs, 100, 5)
                .expect("calibrate failed");

            // Write JSON
            let json = serde_json::json!({
                "dataset": dim_name,
                "dimension": $N,
                "num_points": num_points,
                "alpha": scfg.alpha,
                "admission_rates": diag.admission_rates,
                "useful_gaps": diag.useful_gaps,
                "tail_gaps": diag.tail_gaps,
                "threshold": diag.params.threshold,
                "early_exit_limit": diag.params.early_exit_limit,
                "build": {
                    "diskann_s": diskann_build_s,
                    "staged_graph_s": graph_build_s,
                    "staged_overhead_s": staged_overhead_s,
                    "staged_total_s": total_build_s,
                }
            });

            let path = format!("visualizations/calibration_diag_{}.json", dim_name);
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
                .expect("write json");
            println!("Saved {path}");
            println!("  threshold={:.2}, early_exit_limit={}", diag.params.threshold, diag.params.early_exit_limit);
            println!("  DiskANN build: {:.2}s", diskann_build_s);
            println!("  Staged graph build: {:.2}s  overhead: {:.2}s  total: {:.2}s",
                graph_build_s, staged_overhead_s, total_build_s);
        }};
    }

    match dimension {
        DIM_32 => run_diag!(32),
        DIM_100 => run_diag!(100),
        DIM_128 => run_diag!(128),
        DIM_960 => run_diag!(960),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}
