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
use metrics::{measure_qps, LatencyStats};
use report::table::{print_results_table, BenchmarkResult};
use runner::common::AlgorithmRunner;
use runner::{DiskANNRunner, HNSWRunner, NSGRunner, SSDDiskANNRunner, StagedDiskANNRunner};
use std::path::PathBuf;

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

    /// Clustering method for compressed-diskann: cohesive or lpa
    #[arg(long, default_value = "cohesive")]
    clustering: String,

    /// Drop in-memory vectors after enabling mmap search (reduces heap usage)
    #[arg(long)]
    drop_inmem: bool,
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

    log::info!("Loading base vectors from {:?}", base_path);
    let base = dataset::read_fvecs(base_path).expect("Failed to read base vectors");
    let dimension = base.first().map(|v| v.len()).unwrap_or(0);

    log::info!("Loading query vectors from {:?}", query_path);
    let queries = dataset::read_fvecs(query_path).expect("Failed to read query vectors");

    log::info!("Loading ground truth from {:?}", gt_path);
    let ground_truth = dataset::read_ivecs(gt_path).expect("Failed to read ground truth");

    log::info!(
        "Dataset: {} base vectors, {} queries, dim={}",
        base.len(),
        queries.len(),
        dimension
    );

    let truncated = args.max_points > 0 && args.max_points < base.len();
    let base = if truncated {
        log::info!("Truncating base vectors to {} points", args.max_points);
        base.into_iter().take(args.max_points).collect()
    } else {
        base
    };

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
    // that no longer exist. Recompute it against the truncated base.
    if truncated {
        let gt_k = ds.ground_truth.first().map(|v| v.len()).unwrap_or(100);
        println!(
            "Base truncated to {} points — recomputing ground truth (k={})...",
            args.max_points, gt_k
        );
        ds.recompute_ground_truth(gt_k);
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
                let mut runner = DiskANNRunner::new(48, 32, 1.2);
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
            "compressed-diskann" => {
                let clustering_method = match args.clustering.as_str() {
                    "cohesive" => staged_diskann::ClusteringMethod::Cohesive,
                    "lpa" => staged_diskann::ClusteringMethod::LabelPropagation,
                    other => panic!("Unknown clustering method: {other}. Use 'cohesive' or 'lpa'."),
                };
                let mut runner = StagedDiskANNRunner::new(
                    1.2,  // alpha
                    32,   // graph_degree
                    64,   // search_list_size
                    10,   // max_cluster_point_size
                    4,    // max_connection_clusters
                    3,    // max_connection_per_cluster
                    0.7,  // critical_minimum_rate
                    8,    // n_subquantizers
                    8,    // n_bits
                    5,    // window_size
                    0.01, // epsilon
                    clustering_method,
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
            "profile-compressed" => {
                run_compressed_profile(&dataset, args.k);
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
    ALLOCATOR.reset_peak();
    let build_time = runner.build(&flat_base, num_points, dimension);
    let peak_memory = ALLOCATOR.peak_bytes();
    log::info!(
        "[{}] Build complete: {:.2}s",
        runner.name(),
        build_time.as_secs_f64()
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

    // Search
    log::info!(
        "[{}] Running {} queries...",
        runner.name(),
        dataset.num_queries()
    );
    let queries = &dataset.queries;
    let mut all_results: Vec<Vec<u32>> = Vec::with_capacity(queries.len());

    let (qps, _total, durations) = measure_qps(queries.len(), |i| {
        let result = runner.search(&queries[i], k);
        all_results.push(result.neighbors);
        result.duration
    });

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
        build_time,
        recall_at_1,
        recall_at_10,
        recall_at_100,
        qps,
        latency,
        peak_memory,
    });
}

fn run_compressed_profile(dataset: &Dataset, k: usize) {
    use ndarray::Array2;
    use staged_diskann::{build_diskann_index, ClusteringMethod, StagedDiskANN, DIM_128};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128, "Profiling only supports dim=128");

    let flat_base = dataset.base_flat();
    let data_2d = Array2::from_shape_vec((num_points, dimension), flat_base)
        .expect("reshape")
        .to_shared();

    println!("Building DiskANN index ({num_points} points, dim={dimension})...");
    let result = build_diskann_index(&data_2d, 1.2, 32, 64, true, Some(8), Some(8), true);
    println!(
        "DiskANN graph build: {:.2}s, PQ build: {:.2}s\n",
        result.graph_build_time.as_secs_f32(),
        result.pq_build_time.as_secs_f32(),
    );

    let pq = result.pq.expect("PQ");
    let pq_codes = result.pq_codes.expect("PQ codes");

    // Prepare queries
    let num_queries = dataset.queries.len().min(100);
    let queries_arr: Vec<[f32; 128]> = (0..num_queries)
        .map(|i| {
            let mut q = [0.0f32; 128];
            q.copy_from_slice(&dataset.queries[i][..128]);
            q
        })
        .collect();
    let n = num_queries as f64;

    // ─── DiskANN baseline: full-graph-only search ───
    // Build a compressed index with minimal config just to use search_with_metric
    // Use epsilon=0 so convergence never triggers → pure DiskANN full-graph search
    println!("─── DiskANN Baseline (full-graph, degree=32) ───");
    {
        let baseline = StagedDiskANN::<128>::from_graph_ref(
            data_2d.clone(),
            &result.graph,
            result.candidate_sets.clone(),
            result.entry_point,
            Some(pq.clone()),
            Some(pq_codes.clone()),
            10,
            4,
            2,
            0.7,
            None,
            ClusteringMethod::Cohesive,
            true,
        );
        // clustering runs inside new()

        let mut total_ndc = 0u64;
        let mut all_results = Vec::new();
        for query in &queries_arr {
            // epsilon=0 → never converge → always Phase 1 (full graph)
            let (res, full_ndc, _, _) = baseline
                .simulate_ssd_search_with_metric(query, k, 64, 5, 0.0)
                .unwrap();
            total_ndc += full_ndc as u64;
            all_results.push(res);
        }
        let recall = if !dataset.ground_truth.is_empty() {
            metrics::recall::mean_recall(&all_results, &dataset.ground_truth, k)
        } else {
            0.0
        };

        println!("  Avg NDC:   {:>8.1}", total_ndc as f64 / n);
        println!("  R@{k}:      {:>8.4}", recall);
        println!("  Degree:    32 (full Vamana graph)");
        println!();
    }

    // ─── Sweep compressed graph configs: vary (m, n) ───
    // m = max_connection_clusters, n = max_connection_per_cluster
    // pruned_degree = m * n
    let configs: Vec<(usize, usize)> = vec![
        (4, 2),  // 8 edges
        (4, 3),  // 12 edges
        (4, 4),  // 16 edges
        (6, 3),  // 18 edges
        (6, 4),  // 24 edges
        (8, 4),  // 32 edges
        (10, 4), // 40 edges
    ];

    // Table header
    println!("─── Compressed Graph: NDC & Recall vs Edge Count ───\n");
    println!(
        "  {:<10} {:<8} {:<10} {:<12} {:<12} {:<10} {:<10}",
        "Config", "Degree", "Build(s)", "Full NDC", "Compr NDC", "R@10", "NDC Δ%"
    );
    println!("  {}", "─".repeat(72));

    // We need the DiskANN baseline NDC for Δ% computation
    // Re-run baseline quickly (or reuse from above — let's just recompute for clean code)
    let mut diskann_ndc = 0u64;
    let mut diskann_recall = 0.0f64;
    {
        let baseline = StagedDiskANN::<128>::from_graph_ref(
            data_2d.clone(),
            &result.graph,
            result.candidate_sets.clone(),
            result.entry_point,
            Some(pq.clone()),
            Some(pq_codes.clone()),
            10,
            4,
            2,
            0.7,
            None,
            ClusteringMethod::Cohesive,
            true,
        );
        // clustering runs inside new()

        let mut total_ndc = 0u64;
        let mut all_results = Vec::new();
        for query in &queries_arr {
            let (res, full_ndc, _, _) = baseline
                .simulate_ssd_search_with_metric(query, k, 64, 5, 0.0)
                .unwrap();
            total_ndc += full_ndc as u64;
            all_results.push(res);
        }
        diskann_ndc = total_ndc;
        diskann_recall = if !dataset.ground_truth.is_empty() {
            metrics::recall::mean_recall(&all_results, &dataset.ground_truth, k)
        } else {
            0.0
        };
    }

    println!(
        "  {:<10} {:<8} {:<10} {:<12} {:<12} {:<10} {:<10}",
        "DiskANN",
        "32",
        "-",
        format!("{:.1}", diskann_ndc as f64 / n),
        "-",
        format!("{:.4}", diskann_recall),
        "baseline"
    );

    for &(m, per_n) in &configs {
        let degree = m * per_n;
        let config_label = format!("m{}n{}", m, per_n);

        // Clean up saved graph so we build fresh
        let graph_dir = std::path::PathBuf::from("compressed_dskann_graphs");
        if graph_dir.exists() {
            std::fs::remove_dir_all(&graph_dir).ok();
        }

        let build_start = std::time::Instant::now();
        let compressed = StagedDiskANN::<128>::from_graph_ref(
            data_2d.clone(),
            &result.graph,
            result.candidate_sets.clone(),
            result.entry_point,
            Some(pq.clone()),
            Some(pq_codes.clone()),
            10,    // max_cluster_point_size
            m,     // max_connection_clusters
            per_n, // max_connection_per_cluster
            0.7,
            None,
            ClusteringMethod::Cohesive,
            true,
        );
        // clustering runs inside new()
        let build_time = build_start.elapsed();

        // Two-phase search with convergence
        let mut total_full_ndc = 0u64;
        let mut total_comp_ndc = 0u64;
        let mut all_results = Vec::new();
        for query in &queries_arr {
            let (res, full_ndc, comp_ndc, _) = compressed
                .simulate_ssd_search_with_metric(query, k, 64, 5, 0.01)
                .unwrap();
            total_full_ndc += full_ndc as u64;
            total_comp_ndc += comp_ndc as u64;
            all_results.push(res);
        }

        let recall = if !dataset.ground_truth.is_empty() {
            metrics::recall::mean_recall(&all_results, &dataset.ground_truth, k)
        } else {
            0.0
        };

        let avg_full_ndc = total_full_ndc as f64 / n;
        let avg_comp_ndc = total_comp_ndc as f64 / n;
        let ndc_delta = if diskann_ndc > 0 {
            (avg_full_ndc - diskann_ndc as f64 / n) / (diskann_ndc as f64 / n) * 100.0
        } else {
            0.0
        };

        println!(
            "  {:<10} {:<8} {:<10} {:<12} {:<12} {:<10} {:<10}",
            config_label,
            degree,
            format!("{:.2}", build_time.as_secs_f32()),
            format!("{:.1}", avg_full_ndc),
            format!("{:.1}", avg_comp_ndc),
            format!("{:.4}", recall),
            format!("{:+.1}%", ndc_delta),
        );
    }
    println!();

    // Clean up
    let graph_dir = std::path::PathBuf::from("compressed_dskann_graphs");
    if graph_dir.exists() {
        std::fs::remove_dir_all(&graph_dir).ok();
    }
}
