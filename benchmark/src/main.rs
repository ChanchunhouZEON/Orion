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
            "staged-diskann" => {
                let mut runner = StagedDiskANNRunner::new(
                    "StagedDiskANN",
                    1.2,  // alpha
                    32,   // graph_degree
                    48,   // search_list_size (matches DiskANN)
                    10,   // max_cluster_point_size
                    8,    // max_connection_clusters (m)
                    3,    // max_connection_per_cluster (n)
                    0.7,  // critical_minimum_rate
                    5,    // window_size
                    0.01, // epsilon
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
                // staged_diskann in_mem_search with epsilon=0 → pure greedy on full graph,
                // equivalent to DiskANN Vamana. Uses same build params as "diskann".
                let mut runner = StagedDiskANNRunner::new(
                    "InMemStagedDiskANN",
                    1.2, // alpha
                    32,  // graph_degree
                    48,  // search_list_size — matches DiskANN runner
                    10,  // max_cluster_point_size (unused in search)
                    4,   // max_connection_clusters (unused in search)
                    3,   // max_connection_per_cluster (unused in search)
                    0.7, // critical_minimum_rate (unused in search)
                    5,   // window_size
                    0.0, // epsilon=0 → never converge → always full graph
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
            "epsilon-sweep" => {
                run_epsilon_sweep(&dataset, args.k);
            }
            "frontier-sweep" => {
                run_frontier_sweep(&dataset, args.k);
            }
            "full-frontier-sweep" => {
                run_full_frontier_sweep(&dataset, args.k);
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

fn run_compressed_profile(dataset: &Dataset, k: usize) {
    use ndarray::Array2;
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128, "Profiling only supports dim=128");

    let flat_base = dataset.base_flat();
    let data_2d = Array2::from_shape_vec((num_points, dimension), dataset.base_flat())
        .expect("reshape")
        .to_shared();

    println!("Building DiskANN index ({num_points} points, dim={dimension})...");
    let mut result = build_diskann_index(
        &flat_base,
        num_points,
        dimension,
        1.2,
        32,
        64,
        true,
        Some(8),
        Some(8),
        true,
    )
    .expect("build failed");
    println!(
        "DiskANN graph build: {:.2}s, PQ build: {:.2}s\n",
        result.graph_build_time.as_secs_f32(),
        result.pq_build_time.as_secs_f32(),
    );

    let pq = result.pq.expect("PQ");
    let pq_codes = result.pq_codes.expect("PQ codes");

    // Free the index to reduce peak memory; we build InmemDataset from data_2d below.
    {
        let idx = result
            .index
            .as_any_mut()
            .downcast_mut::<diskann::index::InmemIndex<f32, 128>>()
            .expect("downcast to InmemIndex<f32, 128>");
        let _ = std::mem::replace(
            &mut idx.dataset,
            diskann::model::InmemDataset::new(0, 1.0).unwrap(),
        );
    }
    drop(result.index);

    // Helper: build an InmemDataset<f32, 128> from the ndarray data.
    let make_dataset = |data: &ndarray::ArcArray2<f32>| -> diskann::model::InmemDataset<f32, 128> {
        let n = data.nrows();
        let mut ds = diskann::model::InmemDataset::<f32, 128>::new(n, 1.0).unwrap();
        let flat: &[f32] = data.as_slice().expect("contiguous data_2d");
        ds.data.memcpy(&flat[..n * 128]).unwrap();
        ds
    };

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
        let dataset_128 = make_dataset(&data_2d);
        let baseline = StagedDiskANN::<128>::new(
            dataset_128,
            result.graph.clone(),
            result.candidate_sets.clone(),
            result.entry_point,
            Some(pq.clone()),
            Some(pq_codes.clone()),
            10,
            4,
            2,
            0.7,
            None,
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
        let dataset_128 = make_dataset(&data_2d);
        let baseline = StagedDiskANN::<128>::new(
            dataset_128,
            result.graph.clone(),
            result.candidate_sets.clone(),
            result.entry_point,
            Some(pq.clone()),
            Some(pq_codes.clone()),
            10,
            4,
            2,
            0.7,
            None,
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
        let dataset_128 = make_dataset(&data_2d);
        let compressed = StagedDiskANN::<128>::new(
            dataset_128,
            result.graph.clone(),
            result.candidate_sets.clone(),
            result.entry_point,
            Some(pq.clone()),
            Some(pq_codes.clone()),
            10,    // max_cluster_point_size
            m,     // max_connection_clusters
            per_n, // max_connection_per_cluster
            0.7,
            None,
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

/// Sweep epsilon values for StagedDiskANN to show the QPS / recall trade-off.
///
/// Builds the DiskANN graph and StagedDiskANN index once, then reruns search
/// with each epsilon — no per-epsilon rebuild needed.
///
/// epsilon = 0.0 → convergence never triggers → pure greedy (DiskANN-equivalent).
/// epsilon > 0.0 → switches to compressed graph neighbors once the sliding window
///                 of best distances converges, reducing distance computations.
fn run_epsilon_sweep(dataset: &Dataset, k: usize) {
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(
        dimension, DIM_128,
        "epsilon-sweep currently requires dim=128 (SIFT); got {}",
        dimension
    );

    let flat_base = dataset.base_flat();

    // ── Build: one graph, one staged index ──────────────────────────────────
    println!(
        "Building DiskANN + StagedDiskANN ({} pts, dim={})...",
        num_points, dimension
    );
    let mut result = build_diskann_index(
        &flat_base, num_points, dimension, 1.2, 32, 48, false, None, None, true,
    )
    .expect("build failed");
    println!(
        "  Graph build: {:.2}s",
        result.graph_build_time.as_secs_f32()
    );

    // Remove cached compressed graph so we always build fresh.
    let cache_dir = std::path::PathBuf::from("compressed_dskann_graphs");
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    let dataset_128 = {
        let idx = result
            .index
            .as_any_mut()
            .downcast_mut::<diskann::index::InmemIndex<f32, 128>>()
            .expect("downcast to InmemIndex<f32, 128>");
        std::mem::replace(
            &mut idx.dataset,
            diskann::model::InmemDataset::new(0, 1.0).unwrap(),
        )
    };
    drop(result.index);

    let staged = StagedDiskANN::<128>::new(
        dataset_128,
        result.graph,
        result.candidate_sets,
        result.entry_point,
        None, // no PQ
        None,
        10,  // max_cluster_point_size
        4,   // max_connection_clusters
        3,   // max_connection_per_cluster
        0.7, // critical_minimum_rate
        None,
        false, // don't save to disk
    );

    // ── Prepare queries ──────────────────────────────────────────────────────
    let num_queries = dataset.queries.len();
    let queries_arr: Vec<[f32; 128]> = dataset
        .queries
        .iter()
        .map(|q| {
            let mut arr = [0.0f32; 128];
            arr.copy_from_slice(&q[..128]);
            arr
        })
        .collect();

    // ── Sweep: outer = window_size, inner = epsilon ──────────────────────────
    const SEARCH_LIST_SIZE: usize = 48;

    let window_sizes: &[usize] = &[5, 10, 20, 50];
    let epsilons: &[f32] = &[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];

    println!(
        "\n─── Window × Epsilon Sweep  (search_list_size={}, k={}) ───",
        SEARCH_LIST_SIZE, k
    );
    println!("  Note: epsilon=0.0 → convergence never fires → pure greedy (≡ DiskANN Vamana)\n");

    for &ws in window_sizes {
        println!(
            "  window_size = {}\n  {:<10}  {:<12}  {:<8}  {:<8}  {:<12}  {:<12}",
            ws, "Epsilon", "QPS", "R@1", "R@10", "Mean(ms)", "P99(ms)"
        );
        println!("  {}", "─".repeat(70));

        for &eps in epsilons {
            let mut all_results: Vec<Vec<u32>> = Vec::with_capacity(num_queries);

            let (qps, _, durations) = measure_qps(num_queries, |i| {
                let t = Instant::now();
                let res = staged
                    .search(&queries_arr[i], k, SEARCH_LIST_SIZE, ws, eps)
                    .expect("search failed");
                let dur = t.elapsed();
                all_results.push(res);
                dur
            });

            let recall_1 = metrics::recall::mean_recall(&all_results, &dataset.ground_truth, 1);
            let recall_10 = metrics::recall::mean_recall(&all_results, &dataset.ground_truth, 10);
            let latency = LatencyStats::from_durations(durations);

            println!(
                "  {:<10.4}  {:<12.1}  {:<8.4}  {:<8.4}  {:<12.3}  {:<12.3}",
                eps,
                qps,
                recall_1,
                recall_10,
                latency.mean.as_secs_f64() * 1000.0,
                latency.p99.as_secs_f64() * 1000.0,
            );
        }
        println!();
    }

    // Clean up
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }
}

/// Sweep search_list_size × epsilon to build full Pareto frontiers for
/// DiskANN baseline and StagedDiskANN, writing results to
/// `visualizations/frontier_data.json` for plotting.
fn run_frontier_sweep(dataset: &Dataset, k: usize) {
    use serde_json::json;
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(
        dimension, DIM_128,
        "frontier-sweep requires dim=128; got {}",
        dimension
    );

    let flat_base = dataset.base_flat();

    println!(
        "Building DiskANN + StagedDiskANN ({} pts, dim={})...",
        num_points, dimension
    );
    let mut result = build_diskann_index(
        &flat_base, num_points, dimension, 1.2, 32, 64, false, None, None, true,
    )
    .expect("build failed");
    println!(
        "  Graph build: {:.2}s\n",
        result.graph_build_time.as_secs_f32()
    );

    let cache_dir = std::path::PathBuf::from("compressed_dskann_graphs");
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    let dataset_128 = {
        let idx = result
            .index
            .as_any_mut()
            .downcast_mut::<diskann::index::InmemIndex<f32, 128>>()
            .expect("downcast to InmemIndex<f32, 128>");
        std::mem::replace(
            &mut idx.dataset,
            diskann::model::InmemDataset::new(0, 1.0).unwrap(),
        )
    };
    drop(result.index);

    let staged = StagedDiskANN::<128>::new(
        dataset_128,
        result.graph,
        result.candidate_sets,
        result.entry_point,
        None,
        None,
        10,  // max_cluster_point_size
        4,   // max_connection_clusters
        3,   // max_connection_per_cluster
        0.7, // critical_minimum_rate
        None,
        false,
    );

    let queries_arr: Vec<[f32; 128]> = dataset
        .queries
        .iter()
        .map(|q| {
            let mut arr = [0.0f32; 128];
            arr.copy_from_slice(&q[..128]);
            arr
        })
        .collect();
    let num_queries = queries_arr.len();

    // search_list_size values spanning low → high recall for DiskANN
    let search_list_sizes: &[usize] = &[10, 14, 18, 24, 30, 40, 48, 64, 80, 100];
    // Best window_size from previous sweep
    let window_size: usize = 10;
    let epsilons: &[f32] = &[0.0, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0];

    let mut diskann_pts: Vec<serde_json::Value> = Vec::new();
    let mut staged_pts: Vec<serde_json::Value> = Vec::new();

    println!(
        "{:<6}  {:<8}  {:<10}  {:<10}  {:<8}",
        "SLS", "Epsilon", "QPS", "R@10", "Source"
    );
    println!("{}", "─".repeat(50));

    for &sls in search_list_sizes {
        for &eps in epsilons {
            let mut all_results: Vec<Vec<u32>> = Vec::with_capacity(num_queries);
            let (qps, _, durations) = measure_qps(num_queries, |i| {
                let t = Instant::now();
                let res = staged
                    .search(&queries_arr[i], k, sls, window_size, eps)
                    .expect("search failed");
                let dur = t.elapsed();
                all_results.push(res);
                dur
            });

            let recall_10 = metrics::recall::mean_recall(&all_results, &dataset.ground_truth, 10);
            let latency = LatencyStats::from_durations(durations);

            let point = json!({
                "search_list_size": sls,
                "window_size": window_size,
                "epsilon": eps,
                "qps": (qps * 10.0).round() / 10.0,
                "recall_10": (recall_10 * 100000.0).round() / 100000.0,
                "mean_ms": (latency.mean.as_secs_f64() * 1e6).round() / 1e3,
                "p99_ms": (latency.p99.as_secs_f64() * 1e6).round() / 1e3,
            });

            let label = if eps == 0.0 { "DiskANN" } else { "Staged" };
            println!(
                "{:<6}  {:<8.4}  {:<10.1}  {:<10.4}  {}",
                sls, eps, qps, recall_10, label
            );

            if eps == 0.0 {
                diskann_pts.push(point.clone());
            }
            staged_pts.push(point);
        }
    }

    let output = json!({
        "meta": {
            "dataset": dataset.name,
            "num_points": num_points,
            "k": k,
            "window_size": window_size,
        },
        "diskann": diskann_pts,
        "staged_diskann": staged_pts,
    });

    let out_path = "visualizations/frontier_data.json";
    std::fs::write(out_path, serde_json::to_string_pretty(&output).unwrap())
        .expect("Failed to write frontier_data.json");
    println!("\nSaved → {out_path}");

    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }
}

/// Full three-dimensional sweep: search_list_size × window_size × epsilon.
///
/// For each (sls, ws, eps) triple, measures QPS and R@10.
/// eps=0 rows serve as the DiskANN baseline for that SLS.
/// Results written to `visualizations/full_frontier_data.json`.
fn run_full_frontier_sweep(dataset: &Dataset, k: usize) {
    use serde_json::json;
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(
        dimension, DIM_128,
        "full-frontier-sweep requires dim=128; got {}",
        dimension
    );

    let flat_base = dataset.base_flat();

    println!(
        "Building DiskANN + StagedDiskANN ({} pts, dim={})...",
        num_points, dimension
    );
    let mut result = build_diskann_index(
        &flat_base, num_points, dimension, 1.2, 32, 64, false, None, None, true,
    )
    .expect("build failed");
    println!(
        "  Graph build: {:.2}s\n",
        result.graph_build_time.as_secs_f32()
    );

    let cache_dir2 = std::path::PathBuf::from("compressed_dskann_graphs_full");
    if cache_dir2.exists() {
        std::fs::remove_dir_all(&cache_dir2).ok();
    }

    let dataset_128 = {
        let idx = result
            .index
            .as_any_mut()
            .downcast_mut::<diskann::index::InmemIndex<f32, 128>>()
            .expect("downcast to InmemIndex<f32, 128>");
        std::mem::replace(
            &mut idx.dataset,
            diskann::model::InmemDataset::new(0, 1.0).unwrap(),
        )
    };
    drop(result.index);

    let staged = StagedDiskANN::<128>::new(
        dataset_128,
        result.graph,
        result.candidate_sets,
        result.entry_point,
        None,
        None,
        10,
        4,
        3,
        0.7,
        None,
        false,
    );

    let queries_arr: Vec<[f32; 128]> = dataset
        .queries
        .iter()
        .map(|q| {
            let mut arr = [0.0f32; 128];
            arr.copy_from_slice(&q[..128]);
            arr
        })
        .collect();
    let num_queries = queries_arr.len();

    let search_list_sizes: &[usize] = &[10, 14, 18, 24, 30, 40, 48, 64, 80, 100];
    let window_sizes: &[usize] = &[5, 10, 20];
    let epsilons: &[f32] = &[0.0, 0.01, 0.05, 0.1, 0.3, 0.5];

    let total = search_list_sizes.len() * window_sizes.len() * epsilons.len();
    println!(
        "Running {} combinations ({} SLS × {} WS × {} eps)...\n",
        total,
        search_list_sizes.len(),
        window_sizes.len(),
        epsilons.len()
    );

    let mut all_pts: Vec<serde_json::Value> = Vec::with_capacity(total);
    let mut done = 0usize;

    for &sls in search_list_sizes {
        for &ws in window_sizes {
            for &eps in epsilons {
                let mut all_results: Vec<Vec<u32>> = Vec::with_capacity(num_queries);
                let (qps, _, _) = measure_qps(num_queries, |i| {
                    let t = Instant::now();
                    let res = staged
                        .search(&queries_arr[i], k, sls, ws, eps)
                        .expect("search failed");
                    let dur = t.elapsed();
                    all_results.push(res);
                    dur
                });
                let recall_10 =
                    metrics::recall::mean_recall(&all_results, &dataset.ground_truth, 10);

                all_pts.push(json!({
                    "search_list_size": sls,
                    "window_size": ws,
                    "epsilon": eps,
                    "qps": (qps * 10.0).round() / 10.0,
                    "recall_10": (recall_10 * 100000.0).round() / 100000.0,
                }));

                done += 1;
                println!(
                    "  [{:>3}/{}] SLS={:>3} WS={:>2} eps={:.3}  QPS={:>8.0}  R@10={:.4}",
                    done, total, sls, ws, eps, qps, recall_10
                );
            }
        }
    }

    let output = json!({
        "meta": {
            "dataset": dataset.name,
            "num_points": num_points,
            "k": k,
        },
        "points": all_pts,
    });

    let out_path = "visualizations/full_frontier_data.json";
    std::fs::write(out_path, serde_json::to_string_pretty(&output).unwrap()).expect("write failed");
    println!("\nSaved → {out_path}");

    if cache_dir2.exists() {
        std::fs::remove_dir_all(&cache_dir2).ok();
    }
}

/// Profile and compare DiskANN build time vs StagedDiskANN build time.
///
/// Measures:
///   - DiskANN Vamana graph build time
///   - StagedDiskANN clustering + compression overhead
/// Runs 3 trials each and reports mean ± stddev.
/// Diagnose convergence behavior and compressed graph quality.
///
/// Sweeps over m (max_connection_clusters) values to compare compressed
/// degree distribution and convergence/recall trade-offs.
fn run_convergence_diag(dataset: &Dataset, k: usize) {
    use ndarray::Array2;
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128);

    let flat_base = dataset.base_flat();
    let data_2d = Array2::from_shape_vec((num_points, dimension), dataset.base_flat())
        .expect("reshape")
        .to_shared();

    // Remove cached compressed graph
    let cache_dir = std::path::PathBuf::from("compressed_dskann_graphs");
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    println!("Building DiskANN graph ({num_points} pts, dim={dimension})...");
    let mut result = build_diskann_index(
        &flat_base, num_points, dimension, 1.2, 32, 48, false, None, None, true,
    )
    .expect("build failed");
    println!(
        "  Graph build: {:.2}s\n",
        result.graph_build_time.as_secs_f32()
    );

    // Free the index to reduce peak memory; we build InmemDataset from data_2d below.
    {
        let idx = result
            .index
            .as_any_mut()
            .downcast_mut::<diskann::index::InmemIndex<f32, 128>>()
            .expect("downcast to InmemIndex<f32, 128>");
        let _ = std::mem::replace(
            &mut idx.dataset,
            diskann::model::InmemDataset::new(0, 1.0).unwrap(),
        );
    }
    drop(result.index);

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

    // ── Sweep m values (n fixed at 3) ───────────────────────────────
    let m_values: &[usize] = &[4, 6, 8, 10];
    let n_per_cluster = 3usize;
    let epsilons: &[f32] = &[0.0, 0.005, 0.01, 0.05, 0.1];
    let ws = 5usize;

    for &m in m_values {
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir).ok();
        }

        let t = Instant::now();
        let dataset_128 = make_dataset(&data_2d);
        let staged = StagedDiskANN::<128>::new(
            dataset_128,
            result.graph.clone(),
            result.candidate_sets.clone(),
            result.entry_point,
            None,
            None,
            10,            // max_cluster_point_size
            m,             // max_connection_clusters
            n_per_cluster, // max_connection_per_cluster
            0.7,           // critical_minimum_rate
            None,
            false,
        );
        let build_t = t.elapsed();

        // ── Degree distribution ──────────────────────────────────────
        let degree_stats = staged.csr_degree_stats();
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
            "═══ m={m}, n={n_per_cluster}  (max pruned degree = {})  build={:.3}s ═══",
            m * n_per_cluster,
            build_t.as_secs_f32()
        );
        println!("  Avg full degree: {avg_full:.1}  |  Avg compressed: {avg_comp:.1}  |  Max compressed: {max_comp}  |  Zero: {zero_compressed}");

        // Histogram (compact)
        let max_bucket = (m * n_per_cluster).min(32) + 1;
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
                let (res, conv_at, steps, p1_ndc, p2_ndc) =
                    staged.search_diag(q, k, 48, ws, eps).unwrap();
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
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};
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
        runner::StagedDiskANNRunner::new("StagedDiskANN", 1.2, 32, 48, 10, 8, 3, 0.7, 5, 0.01);
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

        let cache_dir = std::path::PathBuf::from("compressed_dskann_graphs");
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
            &flat_base, num_points, dimension, 1.2, 32, 48, false, None, None, true,
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
            result.graph,
            result.candidate_sets,
            result.entry_point,
            None,
            None,
            8,
            8,
            3,
            0.7,
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
    use diskann::index::InmemIndex;
    use staged_diskann::{build_diskann_index, StagedDiskANN, DIM_128};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128);

    let flat_base = dataset.base_flat();

    println!("Building StagedDiskANN index ({num_points} pts)...");
    let mut result = build_diskann_index(
        &flat_base, num_points, dimension, 1.2, 32, 48, false, None, None, true,
    )
    .expect("build failed");

    let dataset_taken = {
        let idx = result
            .index
            .as_any_mut()
            .downcast_mut::<InmemIndex<f32, 128>>()
            .expect("downcast to InmemIndex<f32, 128>");
        std::mem::replace(
            &mut idx.dataset,
            diskann::model::InmemDataset::new(0, 1.0).unwrap(),
        )
    };
    drop(result.index);

    let staged = StagedDiskANN::<128>::new(
        dataset_taken,
        result.graph,
        result.candidate_sets,
        result.entry_point,
        None,
        None,
        10,  // max_cluster_point_size
        8,   // max_connection_clusters
        3,   // max_connection_per_cluster
        0.7, // critical_minimum_rate
        None,
        false,
    );

    let queries_128: Vec<[f32; 128]> = dataset
        .queries
        .iter()
        .map(|q| {
            let mut arr = [0f32; 128];
            arr.copy_from_slice(q);
            arr
        })
        .collect();

    // Warm-up run
    println!("Warm-up run (1000 queries)...");
    let warmup_n = queries_128.len().min(1000);
    for q in &queries_128[..warmup_n] {
        staged.search(q, k, 48, 5, 0.01).ok();
    }

    // Profile run — all queries, single-threaded
    println!(
        "Profiling {} queries (single-threaded)...",
        queries_128.len()
    );
    let stats = staged
        .search_profile(&queries_128, k, 48, 5, 0.01)
        .expect("search_profile failed");
    stats.print_report();
}

fn run_memory_profile(dataset: &Dataset) {
    use diskann::index::{ANNInmemIndex, InmemIndex, create_inmem_index};
    use diskann::model::{CsrGraph, IndexConfiguration, InmemDataset};
    use diskann::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
    use staged_diskann::{StagedDiskANN, DIM_128};
    use std::sync::Arc;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    assert_eq!(dimension, DIM_128);
    let flat_base = dataset.base_flat();

    let fmt = |b: usize| -> String { metrics::memory::format_bytes(b) };

    ALLOCATOR.reset_peak();
    println!("\n═══ Memory Profile (SIFT 100K) ═══\n");
    println!("  {:<50} {:>12} {:>12}", "Step", "Current", "Peak");
    println!("  {}", "─".repeat(75));

    let m = |label: &str| {
        println!("  {:<50} {:>12} {:>12}",
            label,
            metrics::memory::format_bytes(ALLOCATOR.current_bytes()),
            metrics::memory::format_bytes(ALLOCATOR.peak_bytes()),
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
            vector::Metric::L2, dimension, dimension, num_points,
            false, 0, false, 0, 1.0, wp,
        );
        let mut idx_d: Box<dyn ANNInmemIndex<f32>> =
            create_inmem_index::<f32>(config).expect("create index");
        m("D1. create_inmem_index (no CS)");

        idx_d.build_from_data(&flat_base, num_points).expect("build");
        m("D2. build_from_data (no CS)");
    }
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
        vector::Metric::L2, dimension, dimension, num_points,
        false, 0, false, 0, 1.0, wp,
    );
    let mut idx: Box<dyn ANNInmemIndex<f32>> =
        create_inmem_index::<f32>(config).expect("create index");
    m("S1. create_inmem_index (with CS)");

    idx.build_from_data(&flat_base, num_points).expect("build");
    m("S2. build_from_data (Vamana + anchor sets)");

    let entry_point = idx.start_node();

    ALLOCATOR.reset_peak();
    let peak_before_extract = ALLOCATOR.current_bytes();
    let (csr, candidate_sets) = idx.extract_graph_and_candidates(num_points, 32)
        .expect("extract");
    let peak_during_extract = ALLOCATOR.peak_bytes();
    m("S3. extract_graph_and_candidates");
    println!("     ↳ peak DURING extract: {}", fmt(peak_during_extract));
    println!("     ↳ delta from pre-extract: {}", fmt(peak_during_extract - peak_before_extract));

    let candidate_sets = Arc::new(candidate_sets);
    m("S4. Arc::new(candidate_sets)");

    let dataset_128 = {
        let inmem: &mut InmemIndex<f32, 128> = idx.as_any_mut()
            .downcast_mut().expect("downcast");
        std::mem::replace(
            &mut inmem.dataset,
            InmemDataset::new(0, 1.0).unwrap(),
        )
    };
    m("S5. take dataset (mem::replace)");

    drop(idx);
    m("S6. drop(InmemIndex)");

    ALLOCATOR.reset_peak();
    let peak_before_staged = ALLOCATOR.current_bytes();
    let _staged = StagedDiskANN::<128>::new(
        dataset_128, csr, candidate_sets, entry_point,
        None, None, 10, 8, 3, 0.7, None, false,
    );
    let peak_during_staged = ALLOCATOR.peak_bytes();
    m("S7. StagedDiskANN::new (cluster+reorder)");
    println!("     ↳ peak DURING staged: {}", fmt(peak_during_staged));
    println!("     ↳ delta from pre-staged: {}", fmt(peak_during_staged - peak_before_staged));

    println!("\n  {}", "─".repeat(75));
    m("FINAL");
    println!();
}
