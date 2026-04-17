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
use runner::{DiskANNRunner, SSDDiskANNRunner, StagedDiskANNRunner};
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
        alpha: ov
            .and_then(|o| o.alpha)
            .unwrap_or(cfg.defaults.staged.alpha),
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

    /// Algorithms to benchmark (comma-separated, e.g., diskann,staged-diskann)
    #[arg(long, default_value = "diskann,staged-diskann")]
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
            "ablation" => {
                run_ablation(&dataset, args.k);
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

fn run_convergence_diag(dataset: &Dataset, k: usize) {
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let scfg = load_staged_config(dimension);

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    let l_values: [usize; 3] = [48, 100, 200];
    let ws: usize = 5;

    macro_rules! run_diag {
        ($N:literal) => {{
            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            println!(
                "Building StagedDiskANN ({}-dim, alpha={:.2}, R={}, L_build={})...",
                $N, scfg.alpha, scfg.graph_degree, scfg.build_search_list_size,
            );
            let result = build_diskann_index(
                &flat_base, num_points, dimension, scfg.alpha,
                scfg.graph_degree, scfg.build_search_list_size as u32,
                false, None, None, true, scfg.max_extra,
            ).expect("build failed");
            drop(result.index);

            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged = StagedDiskANN::<$N>::new(
                empty_ds, &result.partitions, result.entry_point,
                scfg.graph_degree, scfg.max_extra, None, None, None, false,
            );
            staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

            // Calibrate once on the first 500 queries.
            let calib_qs: Vec<[f32; $N]> = queries[..queries.len().min(500)].to_vec();
            let calib = staged.calibrate(&calib_qs, 48, ws).expect("calibrate failed");
            let thr = calib.threshold;
            let ee = calib.early_exit_limit;
            println!("Calibrated: threshold={:.2}, early_exit_limit={}\n", thr, ee);

            // Graph structure stats (same graph across all L).
            let n = staged.graph.num_nodes();
            let (mut sum_deg, mut sum_local, mut sum_extra) = (0usize, 0usize, 0usize);
            for i in 0..n {
                sum_deg += staged.graph.degree(i);
                sum_local += staged.graph.local_count(i);
                sum_extra += staged.graph.extra_count(i);
            }
            let avg_deg = sum_deg as f64 / n as f64;
            let avg_local = sum_local as f64 / n as f64;
            let avg_extra = sum_extra as f64 / n as f64;
            let avg_rerank = avg_local + avg_extra;

            // Measure a configuration: returns (avg_steps, avg_ndc, recall).
            let measure = |thr_use: f32, ee_use: usize| -> (f64, f64, f64) {
                let mut total_steps = 0u64;
                let mut total_ndc = 0u64;
                let mut results = Vec::with_capacity(queries.len());
                for _ in 0..queries.len() { results.push(Vec::new()); }
                for (qi, q) in queries.iter().enumerate() {
                    let (res, _conv, steps, p1, p2) = staged
                        .search_diag(q, k, 0 /*placeholder*/, ws, thr_use, ee_use)
                        .unwrap();
                    total_steps += steps as u64;
                    total_ndc += (p1 + p2) as u64;
                    results[qi] = res;
                }
                let nq = queries.len() as f64;
                let recall = metrics::recall::mean_recall(&results, &dataset.ground_truth, k);
                (total_steps as f64 / nq, total_ndc as f64 / nq, recall)
            };

            // Wrapper that re-binds search list size per call via closure capture.
            let measure_at_l = |sls: usize, thr_use: f32, ee_use: usize| -> (f64, f64, f64) {
                let mut total_steps = 0u64;
                let mut total_ndc = 0u64;
                let mut results = Vec::with_capacity(queries.len());
                for _ in 0..queries.len() { results.push(Vec::new()); }
                for (qi, q) in queries.iter().enumerate() {
                    let (res, _conv, steps, p1, p2) = staged
                        .search_diag(q, k, sls, ws, thr_use, ee_use)
                        .unwrap();
                    total_steps += steps as u64;
                    total_ndc += (p1 + p2) as u64;
                    results[qi] = res;
                }
                let nq = queries.len() as f64;
                let recall = metrics::recall::mean_recall(&results, &dataset.ground_truth, k);
                (total_steps as f64 / nq, total_ndc as f64 / nq, recall)
            };
            let _ = measure; // silence unused warning

            println!(
                "  {:<8} {:<18} {:<18} {:<18} {:<18} {:<10}",
                "L", "Steps(no-ee)", "Steps(staged)", "NDC(no-ee)", "NDC(staged)", "ΔR@10"
            );
            println!("  {}", "─".repeat(94));

            let mut no_ee_steps = Vec::with_capacity(l_values.len());
            let mut staged_steps = Vec::with_capacity(l_values.len());
            let mut no_ee_ndc = Vec::with_capacity(l_values.len());
            let mut staged_ndc = Vec::with_capacity(l_values.len());
            let mut no_ee_recall = Vec::with_capacity(l_values.len());
            let mut staged_recall = Vec::with_capacity(l_values.len());

            for &sls in &l_values {
                // Baseline: convergence-monitoring disabled (runs until L is full).
                let (s_bl, d_bl, r_bl) = measure_at_l(sls, 0.0, usize::MAX);
                // Staged: calibrated convergence + early exit.
                let (s_st, d_st, r_st) = measure_at_l(sls, thr, ee);

                no_ee_steps.push(s_bl);
                staged_steps.push(s_st);
                no_ee_ndc.push(d_bl);
                staged_ndc.push(d_st);
                no_ee_recall.push(r_bl);
                staged_recall.push(r_st);

                println!(
                    "  L={:<6} {:<8.1}({:>+5.1}%)   {:<8.1}            {:<8.0}({:>+5.1}%)  {:<8.0}            {:<+7.4}",
                    sls,
                    s_bl, (s_st / s_bl - 1.0) * 100.0,
                    s_st,
                    d_bl, (d_st / d_bl - 1.0) * 100.0,
                    d_st,
                    r_st - r_bl,
                );
            }

            let json = serde_json::json!({
                "dataset": dim_name,
                "dimension": $N,
                "num_points": num_points,
                "alpha_staged": scfg.alpha,
                "threshold": thr,
                "early_exit_limit": ee,
                "L_values": l_values,
                "no_early_exit": {
                    "steps": no_ee_steps,
                    "ndc": no_ee_ndc,
                    "recall": no_ee_recall,
                },
                "staged": {
                    "steps": staged_steps,
                    "ndc": staged_ndc,
                    "recall": staged_recall,
                },
                "graph": {
                    "avg_degree": avg_deg,
                    "avg_local": avg_local,
                    "avg_extra": avg_extra,
                    "avg_rerank": avg_rerank,
                },
            });
            let path = format!("visualizations/convergence_diag_{}.json", dim_name);
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
                .expect("write json");
            println!("\nSaved {path}");
        }};
    }

    match dimension {
        DIM_32 => run_diag!(32),
        DIM_100 => run_diag!(100),
        DIM_128 => run_diag!(128),
        DIM_960 => run_diag!(960),
        _ => panic!("Unsupported dimension for convergence-diag: {dimension}"),
    }
}

fn run_thread_sweep(dataset: &Dataset, k: usize) {
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let scfg = load_staged_config(dimension);

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    println!(
        "Building DiskANN (α=2.0, R=32, L_build=48) for {} {}pts...",
        dim_name, num_points
    );
    let mut diskann_runner = runner::DiskANNRunner::new(48, 32, 2.0);
    diskann_runner.build(&flat_base, num_points, dimension);

    println!(
        "Building StagedDiskANN (α={:.2}, R={}, L_build={}) for {} {}pts...",
        scfg.alpha, scfg.graph_degree, scfg.build_search_list_size, dim_name, num_points
    );
    let mut staged_runner = runner::StagedDiskANNRunner::new(
        "StagedDiskANN",
        scfg.alpha,
        scfg.graph_degree as usize,
        scfg.build_search_list_size,
        scfg.max_extra,
        scfg.window_size,
    );
    staged_runner.build(&flat_base, num_points, dimension);

    // Fix search L so QPS scaling is comparable across thread counts.
    let search_l: usize = 48;
    diskann_runner.set_search_list_size(search_l);

    let queries = &dataset.queries;
    let thread_counts: [usize; 6] = [1, 2, 4, 6, 8, 16];
    let trials: usize = 9;

    // Pre-warm scratch buffers on enough threads to cover the largest sweep.
    {
        use rayon::prelude::*;
        let max_threads = *thread_counts.iter().max().unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .build()
            .unwrap();
        let warmup: Vec<Vec<f32>> = (0..max_threads * 4 + 20)
            .map(|_| vec![0.0f32; dimension])
            .collect();
        pool.install(|| {
            warmup.par_iter().for_each(|q| {
                let _ = diskann_runner.search(q, k);
                let _ = staged_runner.search(q, k);
            });
        });
    }

    println!(
        "\n─── Thread Sweep [{} {}pts, k={k}, L={search_l}, {} queries, {trials} trials] ───\n",
        dim_name,
        num_points,
        queries.len(),
    );
    println!(
        "  {:<8} {:<14} {:<10} {:<14} {:<10} {:<10}",
        "Threads", "DiskANN QPS", "D_R@10", "Staged QPS", "S_R@10", "Speedup",
    );
    println!("  {}", "─".repeat(72));

    // Collect per-trial QPS so downstream plots can show confidence bands.
    let sample_qps = |runner: &dyn runner::common::AlgorithmRunner, nt: usize| -> (Vec<f64>, f64) {
        let mut samples = Vec::with_capacity(trials);
        let mut recall = 0.0f64;
        for _ in 0..trials {
            let t = Instant::now();
            let results = runner.search_batch_with_threads(queries, k, nt);
            let wall = t.elapsed();
            let qps = queries.len() as f64 / wall.as_secs_f64();
            let ids: Vec<Vec<u32>> = results.into_iter().map(|r| r.neighbors).collect();
            recall = metrics::recall::mean_recall(&ids, &dataset.ground_truth, k);
            samples.push(qps);
        }
        (samples, recall)
    };

    let mut diskann_qps_all: Vec<Vec<f64>> = Vec::new();
    let mut staged_qps_all: Vec<Vec<f64>> = Vec::new();
    let mut diskann_recall: Vec<f64> = Vec::new();
    let mut staged_recall: Vec<f64> = Vec::new();

    for &nt in &thread_counts {
        let (d_samples, d_r) = sample_qps(&diskann_runner, nt);
        let (s_samples, s_r) = sample_qps(&staged_runner, nt);
        let mut d_sorted = d_samples.clone();
        d_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut s_sorted = s_samples.clone();
        s_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d_med = d_sorted[trials / 2];
        let s_med = s_sorted[trials / 2];
        diskann_qps_all.push(d_samples);
        staged_qps_all.push(s_samples);
        diskann_recall.push(d_r);
        staged_recall.push(s_r);
        println!(
            "  {:<8} {:<14.0} {:<10.4} {:<14.0} {:<10.4} {:<10.2}x",
            nt,
            d_med,
            d_r,
            s_med,
            s_r,
            s_med / d_med,
        );
    }

    // Compute median/min/max + speedup + efficiency from per-trial data.
    let sort = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    };
    let diskann_med: Vec<f64> = diskann_qps_all
        .iter()
        .map(|v| sort(v)[trials / 2])
        .collect();
    let staged_med: Vec<f64> = staged_qps_all.iter().map(|v| sort(v)[trials / 2]).collect();
    let diskann_min: Vec<f64> = diskann_qps_all.iter().map(|v| sort(v)[0]).collect();
    let staged_min: Vec<f64> = staged_qps_all.iter().map(|v| sort(v)[0]).collect();
    let diskann_max: Vec<f64> = diskann_qps_all
        .iter()
        .map(|v| sort(v)[trials - 1])
        .collect();
    let staged_max: Vec<f64> = staged_qps_all.iter().map(|v| sort(v)[trials - 1]).collect();

    let d1 = diskann_med[0];
    let s1 = staged_med[0];
    let d_speedup: Vec<f64> = diskann_med.iter().map(|q| q / d1).collect();
    let s_speedup: Vec<f64> = staged_med.iter().map(|q| q / s1).collect();
    let d_eff: Vec<f64> = thread_counts
        .iter()
        .zip(d_speedup.iter())
        .map(|(&nt, sp)| sp / nt as f64)
        .collect();
    let s_eff: Vec<f64> = thread_counts
        .iter()
        .zip(s_speedup.iter())
        .map(|(&nt, sp)| sp / nt as f64)
        .collect();

    let json = serde_json::json!({
        "dataset": dim_name,
        "dimension": dimension,
        "num_points": num_points,
        "num_queries": queries.len(),
        "k": k,
        "search_list_size": search_l,
        "alpha_staged": scfg.alpha,
        "trials": trials,
        "thread_counts": thread_counts,
        "diskann": {
            "qps_median": diskann_med,
            "qps_min": diskann_min,
            "qps_max": diskann_max,
            "qps_per_trial": diskann_qps_all,
            "recall": diskann_recall,
            "speedup": d_speedup,
            "efficiency": d_eff,
        },
        "staged": {
            "qps_median": staged_med,
            "qps_min": staged_min,
            "qps_max": staged_max,
            "qps_per_trial": staged_qps_all,
            "recall": staged_recall,
            "speedup": s_speedup,
            "efficiency": s_eff,
        },
    });
    let path = format!("visualizations/thread_sweep_{}.json", dim_name);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).expect("write json");
    println!("\nSaved {path}");
}

fn run_build_profile(dataset: &Dataset, _k: usize) {
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let scfg = load_staged_config(dimension);

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    const TRIALS: usize = 5;
    let mut diskann_times: Vec<f64> = Vec::with_capacity(TRIALS);
    let mut staged_graph_times: Vec<f64> = Vec::with_capacity(TRIALS);
    let mut staged_overhead_times: Vec<f64> = Vec::with_capacity(TRIALS);

    macro_rules! run_trials {
        ($N:literal) => {{
            for trial in 0..TRIALS {
                println!(
                    "[{}] Trial {}/{} (alpha={:.2}, degree={}, L_build={})",
                    dim_name,
                    trial + 1,
                    TRIALS,
                    scfg.alpha,
                    scfg.graph_degree,
                    scfg.build_search_list_size,
                );

                // DiskANN baseline: alpha=2.0, no candidate sets.
                let t_da = Instant::now();
                let _da = build_diskann_index(
                    &flat_base, num_points, dimension, 2.0, 32, 48, false, None, None, false, 0,
                )
                .expect("diskann build failed");
                let diskann_s = t_da.elapsed().as_secs_f64();
                drop(_da);

                // Staged: graph build with compute_candidate_sets=true.
                let result = build_diskann_index(
                    &flat_base,
                    num_points,
                    dimension,
                    scfg.alpha,
                    scfg.graph_degree,
                    scfg.build_search_list_size as u32,
                    false,
                    None,
                    None,
                    true,
                    scfg.max_extra,
                )
                .expect("staged build failed");
                let staged_graph_s = result.graph_build_time.as_secs_f64();

                // Staged overhead: StagedDiskANN::new (extract + clustering + reorder).
                let t_ov = Instant::now();
                let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                let mut staged = StagedDiskANN::<$N>::new(
                    empty_ds,
                    &result.partitions,
                    result.entry_point,
                    scfg.graph_degree,
                    scfg.max_extra,
                    None,
                    None,
                    None,
                    false,
                );
                staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                let staged_overhead_s = t_ov.elapsed().as_secs_f64();
                drop(staged);
                drop(result.index);

                println!(
                    "  DiskANN: {:.3}s  Staged graph: {:.3}s  Staged overhead: {:.3}s",
                    diskann_s, staged_graph_s, staged_overhead_s,
                );

                diskann_times.push(diskann_s);
                staged_graph_times.push(staged_graph_s);
                staged_overhead_times.push(staged_overhead_s);
            }
        }};
    }

    match dimension {
        DIM_32 => run_trials!(32),
        DIM_100 => run_trials!(100),
        DIM_128 => run_trials!(128),
        DIM_960 => run_trials!(960),
        _ => panic!("Unsupported dimension for build-profile: {dimension}"),
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let stddev = |v: &[f64], m: f64| {
        let variance = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
        variance.sqrt()
    };

    let d_mean = mean(&diskann_times);
    let d_std = stddev(&diskann_times, d_mean);
    let g_mean = mean(&staged_graph_times);
    let g_std = stddev(&staged_graph_times, g_mean);
    let o_mean = mean(&staged_overhead_times);
    let o_std = stddev(&staged_overhead_times, o_mean);

    let json = serde_json::json!({
        "dataset": dim_name,
        "dimension": dimension,
        "num_points": num_points,
        "trials": TRIALS,
        "alpha": scfg.alpha,
        "diskann_s": d_mean,
        "diskann_s_std": d_std,
        "staged_graph_s": g_mean,
        "staged_graph_s_std": g_std,
        "staged_overhead_s": o_mean,
        "staged_overhead_s_std": o_std,
        "staged_total_s": g_mean + o_mean,
        "per_trial": {
            "diskann_s": diskann_times,
            "staged_graph_s": staged_graph_times,
            "staged_overhead_s": staged_overhead_times,
        },
    });

    let path = format!("visualizations/build_profile_{}.json", dim_name);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).expect("write json");
    println!("\nSaved {path}");
    println!("  DiskANN (alpha=2.0):  {:.3}s ± {:.3}s", d_mean, d_std);
    println!("  Staged graph:         {:.3}s ± {:.3}s", g_mean, g_std);
    println!("  Staged overhead:      {:.3}s ± {:.3}s", o_mean, o_std);
    println!(
        "  Staged / DiskANN:     {:.2}x   overhead: {:.1}% of staged total",
        (g_mean + o_mean) / d_mean,
        o_mean / (g_mean + o_mean) * 100.0,
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
                &flat_base,
                num_points,
                dimension,
                alpha,
                scfg.graph_degree,
                scfg.build_search_list_size as u32,
                false,
                None,
                None,
                true,
                scfg.max_extra,
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
    use staged_diskann::{StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let scfg = load_staged_config(dimension);

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    let fmt = |b: usize| -> String { metrics::memory::format_bytes(b) };

    ALLOCATOR.reset_peak();
    let baseline = ALLOCATOR.current_bytes();

    println!(
        "\n═══ Memory Profile [{} {}pts, dim={}, α_staged={:.2}] ═══",
        dim_name, num_points, dimension, scfg.alpha,
    );
    println!("  Baseline (base+queries+GT): {}\n", fmt(baseline));

    let num_threads = rayon::current_num_threads() as u32;

    // ── DiskANN (alpha=2.0, no candidate sets) ──────────────────────────
    ALLOCATOR.reset_peak();
    let diskann_peak_abs = {
        let wp = IndexWriteParametersBuilder::new(48, 32)
            .with_alpha(2.0)
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
        idx_d
            .build_from_data(&flat_base, num_points)
            .expect("build");
        let peak = ALLOCATOR.peak_bytes();
        drop(idx_d);
        peak
    };
    let diskann_peak = diskann_peak_abs.saturating_sub(baseline);

    // ── StagedDiskANN (config α, with candidate sets) ───────────────────
    ALLOCATOR.reset_peak();

    macro_rules! run_staged {
        ($N:literal) => {{
            let wp = IndexWriteParametersBuilder::new(
                scfg.build_search_list_size as u32,
                scfg.graph_degree,
            )
            .with_alpha(scfg.alpha)
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
            idx.build_from_data(&flat_base, num_points).expect("build");
            let entry_point = idx.start_node();
            let partitions = idx
                .extract_graph_and_candidates(scfg.max_extra)
                .expect("extract");
            let staged_peak_abs = ALLOCATOR.peak_bytes();
            drop(idx);

            let empty_ds = InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged = StagedDiskANN::<$N>::new(
                empty_ds,
                &partitions,
                entry_point,
                scfg.graph_degree,
                scfg.max_extra,
                None,
                None,
                None,
                false,
            );
            staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
            let staged_final_abs = ALLOCATOR.current_bytes();
            drop(staged);
            (staged_peak_abs, staged_final_abs)
        }};
    }

    let (staged_peak_abs, staged_final_abs) = match dimension {
        DIM_32 => run_staged!(32),
        DIM_100 => run_staged!(100),
        DIM_128 => run_staged!(128),
        DIM_960 => run_staged!(960),
        _ => panic!("Unsupported dimension for memory-profile: {dimension}"),
    };
    let staged_peak = staged_peak_abs.saturating_sub(baseline);
    let staged_final = staged_final_abs.saturating_sub(baseline);

    println!("  DiskANN (α=2.0) peak:       {}", fmt(diskann_peak));
    println!(
        "  StagedDiskANN (α={:.2}) peak: {}",
        scfg.alpha,
        fmt(staged_peak)
    );
    println!("  StagedDiskANN final:        {}", fmt(staged_final));
    let ratio = staged_peak as f64 / diskann_peak.max(1) as f64;
    println!(
        "  Peak ratio (Staged / DiskANN): {:.2}x    final / DiskANN: {:.2}x",
        ratio,
        staged_final as f64 / diskann_peak.max(1) as f64
    );

    let json = serde_json::json!({
        "dataset": dim_name,
        "dimension": dimension,
        "num_points": num_points,
        "alpha_staged": scfg.alpha,
        "baseline_b": baseline,
        "diskann_peak_b": diskann_peak,
        "staged_peak_b": staged_peak,
        "staged_final_b": staged_final,
    });
    let path = format!("visualizations/memory_profile_{}.json", dim_name);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).expect("write json");
    println!("\nSaved {path}");
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

                // Export zone admission JSON (overwrite each L; last one persists)
                let dim_name = match $N {
                    32 => "glove25", 100 => "glove100", 128 => "sift", 960 => "gist", _ => "unknown",
                };
                let nq = queries.len() as f64;
                let zone_json = serde_json::json!({
                    "dataset": dim_name,
                    "dimension": $N,
                    "alpha": scfg.alpha,
                    "search_list_size": sls,
                    "zones": {
                        "nav_local":     {"unseen": nav_local.unseen as f64 / nq, "admitted": nav_local.admitted as f64 / nq},
                        "nav_remote":    {"unseen": nav_remote.unseen as f64 / nq, "admitted": nav_remote.admitted as f64 / nq},
                        "rerank_local":  {"unseen": rerank_local.unseen as f64 / nq, "admitted": rerank_local.admitted as f64 / nq},
                        "rerank_remote": {"unseen": rerank_remote.unseen as f64 / nq, "admitted": rerank_remote.admitted as f64 / nq},
                        "rerank_extra":  {"unseen": rerank_extra.unseen as f64 / nq, "admitted": rerank_extra.admitted as f64 / nq},
                    },
                });
                let zone_path = format!("visualizations/zone_admission_{}.json", dim_name);
                std::fs::write(&zone_path, serde_json::to_string_pretty(&zone_json).unwrap())
                    .expect("write zone json");
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
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
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
                "topk_coverage_by_step": diag.topk_coverage_by_step,
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

fn run_ablation(dataset: &Dataset, k: usize) {
    use rayon::prelude::*;
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_100, DIM_128, DIM_32, DIM_960};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let scfg = load_staged_config(dimension);
    let num_threads = 8;

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let search_list_sizes: &[usize] =
        &[16, 20, 24, 32, 40, 48, 56, 64, 80, 100, 128, 160, 200, 256];
    let trials = 5;

    macro_rules! run_ablation {
        ($N:literal) => {{
            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            // ── Build full StagedDiskANN ──
            println!("Building StagedDiskANN ({}-dim, alpha={})...", $N, scfg.alpha);
            let result = build_diskann_index(
                &flat_base, num_points, dimension, scfg.alpha,
                scfg.graph_degree, scfg.build_search_list_size as u32,
                false, None, None, true, scfg.max_extra,
            ).expect("build failed");
            drop(result.index);
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged = StagedDiskANN::<$N>::new(
                empty_ds, &result.partitions, result.entry_point,
                scfg.graph_degree, scfg.max_extra, None, None, None, false,
            );
            staged.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

            // Calibrate
            let calib_qs: Vec<[f32; $N]> = queries[..queries.len().min(500)].to_vec();
            let calib = staged.calibrate(&calib_qs, 48, 5).expect("calibrate failed");
            let thr = calib.threshold;
            let ee = calib.early_exit_limit;
            println!("Calibrated: threshold={:.2}, early_exit_limit={}\n", thr, ee);

            // ── Build no-extra variant (max_extra=0) ──
            println!("Building no-extra variant (max_extra=0)...");
            let result_ne = build_diskann_index(
                &flat_base, num_points, dimension, scfg.alpha,
                scfg.graph_degree, scfg.build_search_list_size as u32,
                false, None, None, true, 0,
            ).expect("build failed");
            drop(result_ne.index);
            let empty_ds_ne = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut staged_ne = StagedDiskANN::<$N>::new(
                empty_ds_ne, &result_ne.partitions, result_ne.entry_point,
                scfg.graph_degree, 0, None, None, None, false,
            );
            staged_ne.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
            let calib_ne = staged_ne.calibrate(&calib_qs, 48, 5).expect("calibrate failed");

            // Measure function
            let measure = |search_fn: &(dyn Fn(&[f32; $N], usize) -> Vec<u32> + Sync)| -> Vec<(f64, f64)> {
                let mut data = Vec::new();
                for &sls in search_list_sizes {
                    let mut qps_samples = Vec::with_capacity(trials);
                    let mut recall = 0.0f64;
                    for _ in 0..trials {
                        let t = Instant::now();
                        let results: Vec<Vec<u32>> = pool.install(|| {
                            queries.par_iter().map(|q| search_fn(q, sls)).collect()
                        });
                        let wall = t.elapsed();
                        qps_samples.push(queries.len() as f64 / wall.as_secs_f64());
                        recall = metrics::recall::mean_recall(&results, &dataset.ground_truth, k);
                    }
                    qps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    data.push((recall, qps_samples[trials / 2]));
                }
                data
            };

            // ── DiskANN baseline (alpha=2.0, standard Vamana search) ──
            println!("Building DiskANN baseline ({}-dim, alpha=2.0)...", $N);
            let mut da_runner = crate::runner::DiskANNRunner::new(48, 32, 2.0);
            da_runner.build(&flat_base, num_points, dimension);
            // Pre-expand scratch for max L
            let max_l = *search_list_sizes.last().unwrap();
            da_runner.set_search_list_size(max_l);
            let warmup_qs: Vec<Vec<f32>> = (0..num_threads * 4 + 20)
                .map(|_| vec![0.0f32; dimension]).collect();
            pool.install(|| {
                use rayon::prelude::*;
                warmup_qs.par_iter().for_each(|q| { let _ = da_runner.search(q, k); });
            });

            // ── Ablation variants ──
            println!("Running ablation sweep...\n");

            // 0. DiskANN baseline
            println!("  [diskann] Vamana (alpha=2.0)");
            // DiskANN requires set_search_list_size before search, not thread-safe in closure.
            // Measure each L separately.
            let diskann_data = {
                let mut data = Vec::new();
                for &sls in search_list_sizes {
                    da_runner.set_search_list_size(sls);
                    let mut qps_samples = Vec::with_capacity(trials);
                    let mut recall = 0.0f64;
                    for _ in 0..trials {
                        let t = Instant::now();
                        let results: Vec<Vec<u32>> = pool.install(|| {
                            queries.par_iter()
                                .map(|q| da_runner.search(q.as_slice(), k).neighbors)
                                .collect()
                        });
                        let wall = t.elapsed();
                        qps_samples.push(queries.len() as f64 / wall.as_secs_f64());
                        recall = metrics::recall::mean_recall(&results, &dataset.ground_truth, k);
                    }
                    qps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    data.push((recall, qps_samples[trials / 2]));
                }
                data
            };

            // 1. Full StagedDiskANN
            println!("  [full] convergence + early_exit + extra");
            let full_data = measure(&|q, sls| {
                staged.search(q, k, sls, 5, thr, ee).unwrap_or_default()
            });

            // 2. No early exit (convergence on, ee=MAX)
            println!("  [no-early-exit] convergence on, ee=MAX");
            let no_ee_data = measure(&|q, sls| {
                staged.search(q, k, sls, 5, thr, usize::MAX).unwrap_or_default()
            });

            // 3. No extra candidates (max_extra=0, separate build)
            println!("  [no-extra] max_extra=0, convergence + ee on");
            let no_extra_data = measure(&|q, sls| {
                staged_ne.search(q, k, sls, 5, calib_ne.threshold, calib_ne.early_exit_limit)
                    .unwrap_or_default()
            });

            // Print results
            println!("\n{:<6} {:<22} {:<22} {:<22} {:<22}",
                "L", "DiskANN (R/QPS)", "full (R/QPS)", "no-ee (R/QPS)", "no-extra (R/QPS)");
            println!("{}", "─".repeat(96));
            for (i, &sls) in search_list_sizes.iter().enumerate() {
                println!("L={:<4} {:.3}/{:<12.0}  {:.3}/{:<12.0}  {:.3}/{:<12.0}  {:.3}/{:<12.0}",
                    sls,
                    diskann_data[i].0, diskann_data[i].1,
                    full_data[i].0, full_data[i].1,
                    no_ee_data[i].0, no_ee_data[i].1,
                    no_extra_data[i].0, no_extra_data[i].1,
                );
            }

            // Save JSON
            let to_json = |data: &[(f64, f64)]| -> Vec<(f64, f64)> {
                data.iter().map(|&(r, q)| (r, q)).collect::<Vec<_>>()
            };
            let json = serde_json::json!({
                "dataset": dim_name,
                "dimension": $N,
                "num_points": num_points,
                "threads": num_threads,
                "search_list_sizes": search_list_sizes,
                "diskann": to_json(&diskann_data),
                "full": to_json(&full_data),
                "no_early_exit": to_json(&no_ee_data),
                "no_extra": to_json(&no_extra_data),
            });
            let path = format!("visualizations/ablation_{}.json", dim_name);
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).expect("write json");
            println!("\nSaved {path}");
        }};
    }

    match dimension {
        DIM_32 => run_ablation!(32),
        DIM_100 => run_ablation!(100),
        DIM_128 => run_ablation!(128),
        DIM_960 => run_ablation!(960),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}
