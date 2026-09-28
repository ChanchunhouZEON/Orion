/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

mod config;
mod dataset;
mod metrics;
mod report;
mod runner;
mod utils;
#[allow(dead_code)]
mod cli;
use runner::parlayann_bridge;

use clap::Parser;
use dataset::Dataset;
use runner::common::AlgorithmRunner;
use runner::cascade;
use std::path::PathBuf;

// `OrionConfig` + `load_orion_config` used to live here; they have
// been promoted to `benchmark/src/config.rs` as a proper module with
// richer per-dataset resolution (paths, metric, base-graph source).
// Use `config::load_dataset_config_by_dim(dim).staged` at call sites.

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
    #[arg(long, default_value_t = config::load_sweep_config().k)]
    k: usize,

    #[command(flatten)]
    sweep: config::SweepOverrides,

    /// Print resolved settings without loading vectors or building indexes.
    #[arg(long)]
    print_config: bool,

    /// Algorithms to benchmark (comma-separated; see `match *algo` below
    /// for the live list — `ablation`, `cascade-ablation`, `build-profile`,
    /// `thread-sweep`, …)
    #[arg(long, default_value = "ablation,cascade-ablation")]
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

fn calibration_samples() -> usize {
    config::load_sweep_config().calibration.samples
}

fn calibration_l(k: usize) -> usize {
    config::load_sweep_config().calibration.search_list_size(k)
}

fn experiment(profile: &str, k: usize, overrides: &config::SweepOverrides) -> config::ResolvedSweep {
    config::load_sweep_config().resolve(profile, k, overrides)
        .unwrap_or_else(|e| panic!("Invalid benchmark settings: {e}"))
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

    let selected: Vec<&str> = args.algorithms.split(',').map(|s| s.trim()).collect();
    let resolved: Vec<_> = selected.iter().filter_map(|algo| {
        let profile = match *algo {
            "ablation" | "cascade-ablation" => "ablation",
            "ads-comparison" => "ads",
            "convergence-diag" | "search-profile" => "diagnostic",
            "thread-sweep" | "extra-profile" => "single",
            _ => return None,
        };
        Some((algo, experiment(profile, args.k, &args.sweep)))
    }).collect();
    assert!(args.k > 0, "k must be positive");
    println!("{}", serde_json::to_string_pretty(&resolved).unwrap());
    if args.print_config { return; }

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

    utils::validate_ground_truth(&dataset.ground_truth, dataset.queries.len(), args.k)
        .expect("invalid ground truth");

    for algo in &selected {
        match *algo {
            "build-profile" => {
                run_build_profile(&dataset, args.k);
            }
            "convergence-diag" => {
                run_convergence_diag(&dataset, args.k, &args.sweep);
            }
            "thread-sweep" => {
                run_thread_sweep(&dataset, args.k, &args.sweep);
            }
            "search-profile" => {
                run_search_profile(&dataset, args.k, &args.sweep);
            }
            "memory-profile" => {
                run_memory_profile(&dataset);
            }
            "cliff-profile" => {
                run_cliff_profile(&dataset);
            }
            "neighbor-contribution" => {
                run_neighbor_contribution_profile(&dataset);
            }
            "extra-profile" => {
                run_extra_profile(&dataset, args.k, &args.sweep);
            }
            "calibration-diag" => {
                run_calibration_diag(&dataset, args.k);
            }
            "ablation" => {
                run_ablation(&dataset, args.k, &args.sweep);
            }
            "cascade-ablation" => {
                run_cascade_ablation(&dataset, args.k, &args.sweep);
            }
            "ads-comparison" => {
                run_ads_comparison(&dataset, args.k, &args.sweep);
            }
            other => {
                log::warn!("Unknown algorithm: {other}");
            }
        }
    }
}

fn run_convergence_diag(dataset: &Dataset, k: usize, overrides: &config::SweepOverrides) {
    use orion::{build_diskann_index, Orion, DIM_100, DIM_128, DIM_32, DIM_960};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let ocfg = crate::config::load_dataset_config_by_dim(dimension).orion;

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    let l_values = experiment("diagnostic", k, overrides).search_list_sizes;
    let ws = ocfg.window_size;

    macro_rules! run_diag {
        ($N:literal) => {{
            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            println!(
                "Building Orion ({}-dim, alpha={:.2}, R={}, L_build={})...",
                $N, ocfg.alpha, ocfg.graph_degree, ocfg.build_search_list_size,
            );
            let result = build_diskann_index(
                &flat_base, num_points, dimension, ocfg.alpha,
                ocfg.graph_degree, ocfg.build_search_list_size as u32,
                false, None, None, true, ocfg.max_extra,
            ).expect("build failed");
            drop(result.index);

            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = Orion::<$N>::new(
                empty_ds, &result.partitions, result.entry_point,
                ocfg.graph_degree, ocfg.max_extra, None, None, None, false,
            );
            idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

            // Calibrate once on the first 200 queries.
            let calib_qs: Vec<[f32; $N]> = queries[..queries.len().min(calibration_samples())].to_vec();
            let calib_l = calibration_l(k);
            let calib = idx.calibrate(&calib_qs, calib_l, ws, orion::CalibrationConfig { k, ..Default::default() }).expect("calibrate failed");
            let thr = calib.threshold;
            let ee = calib.early_exit_limit;
            println!("Calibrated: threshold={:.2}, early_exit_limit={}\n", thr, ee);

            // Graph structure stats (same graph across all L).
            let n = idx.graph.num_nodes();
            let (mut sum_deg, mut sum_local, mut sum_extra) = (0usize, 0usize, 0usize);
            for i in 0..n {
                sum_deg += idx.graph.degree(i);
                sum_local += idx.graph.local_count(i);
                sum_extra += idx.graph.extra_count(i);
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
                    let (res, _conv, steps, p1, p2) = idx
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
                    let (res, _conv, steps, p1, p2) = idx
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
                "L", "Steps(no-ee)", "Steps(orion)", "NDC(no-ee)", "NDC(orion)", format!("ΔR@{k}")
            );
            println!("  {}", "─".repeat(94));

            let mut no_ee_steps = Vec::with_capacity(l_values.len());
            let mut orion_steps = Vec::with_capacity(l_values.len());
            let mut no_ee_ndc = Vec::with_capacity(l_values.len());
            let mut orion_ndc = Vec::with_capacity(l_values.len());
            let mut no_ee_recall = Vec::with_capacity(l_values.len());
            let mut orion_recall = Vec::with_capacity(l_values.len());

            for &sls in &l_values {
                // Baseline: convergence-monitoring disabled (runs until L is full).
                let (s_bl, d_bl, r_bl) = measure_at_l(sls, 0.0, usize::MAX);
                // Orion: calibrated convergence + early exit.
                let (s_st, d_st, r_st) = measure_at_l(sls, thr, ee);

                no_ee_steps.push(s_bl);
                orion_steps.push(s_st);
                no_ee_ndc.push(d_bl);
                orion_ndc.push(d_st);
                no_ee_recall.push(r_bl);
                orion_recall.push(r_st);

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
                "alpha_orion": ocfg.alpha,
                "threshold": thr,
                "early_exit_limit": ee,
                "L_values": l_values,
                "no_early_exit": {
                    "steps": no_ee_steps,
                    "ndc": no_ee_ndc,
                    "recall": no_ee_recall,
                },
                "orion": {
                    "steps": orion_steps,
                    "ndc": orion_ndc,
                    "recall": orion_recall,
                },
                "graph": {
                    "avg_degree": avg_deg,
                    "avg_local": avg_local,
                    "avg_extra": avg_extra,
                    "avg_rerank": avg_rerank,
                },
            });
            let path = utils::result_path("convergence_diag", dim_name, k);
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

fn run_thread_sweep(dataset: &Dataset, k: usize, overrides: &config::SweepOverrides) {
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let ocfg = crate::config::load_dataset_config_by_dim(dimension).orion;

    let dim_name = match dimension {
        128 => "sift",
        960 => "gist",
        _ => "unknown/unsupported dataset(current version only support GIST & SIFT which use L2Q metric as default)",
    };

    // DiskANN and Orion both build at the dataset's orion params
    // (sweep.yaml `datasets.<name>.orion` block) so the comparison
    // isolates the search algorithm, not the build hyperparams. The
    // legacy hardcoded "α=2.0 R=32 L=48" label predated the
    // DiskANN-matches-orion change in `run_qps_recall_sweep` and
    // was lying about what got built.
    println!(
        "Building DiskANN (α={:.2}, R={}, L_build={}) for {} {}pts...",
        ocfg.alpha, ocfg.graph_degree, ocfg.build_search_list_size, dim_name, num_points
    );
    let mut diskann_runner =
        runner::DiskANNRunner::new(ocfg.build_search_list_size, ocfg.graph_degree, ocfg.alpha);

    // Cache paths mirror `orion`'s naming so a single `orion`
    // build populates both. Both algorithms here use the same R / L /
    // α / max_extra triplet (the `ocfg` block), so the same cache slot
    // works for both. When the cache files exist, build() short-circuits
    // straight to load — saves ~70s of Vamana construction per run and
    // eliminates graph-topology jitter between thread-sweep runs.
    let alpha_tag = format!("{:.2}", ocfg.alpha).replace('.', "_");
    // `_l2` suffix matches `diskann_sweep`'s key shape so the cached
    // Vamana graph built by the full sweep can be reused here.
    let diskann_cache = std::path::PathBuf::from("cache/diskann").join(format!(
        "{}_n{}_r{}_l{}_a{}_l2.bin",
        dim_name, num_points, ocfg.graph_degree, ocfg.build_search_list_size, alpha_tag,
    ));
    let orion_cache = std::path::PathBuf::from("cache/orion").join(format!(
        "{}_n{}_r{}_l{}_a{}_ex{}.bin",
        dim_name,
        num_points,
        ocfg.graph_degree,
        ocfg.build_search_list_size,
        alpha_tag,
        ocfg.max_extra,
    ));
    diskann_runner.set_cache_path(&diskann_cache);
    diskann_runner.build(&flat_base, num_points, dimension);

    println!(
        "Building Orion (α={:.2}, R={}, L_build={}) for {} {}pts...",
        ocfg.alpha, ocfg.graph_degree, ocfg.build_search_list_size, dim_name, num_points
    );
    // thread-sweep uses the dataset's `sweep.yaml` cascade triple —
    // unified pipeline drives whichever (prefilter, admission, rerank)
    // the workload's PA-aligned recipe specifies.
    let mut orion_runner = runner::OrionRunner::new(
        "Orion",
        ocfg.alpha,
        ocfg.graph_degree as usize,
        ocfg.build_search_list_size,
        ocfg.max_extra,
        ocfg.window_size,
        ocfg.prefilter,
        ocfg.admission,
        ocfg.rerank,
    );
    orion_runner.set_cache_path(&orion_cache);
    orion_runner.build(&flat_base, num_points, dimension);

    // Fix search L on both runners so QPS scaling is comparable across
    // thread counts AND the two algorithms operate at the same recall
    // target. Orion's `new()` defaults `search_list_size` to the build
    // L (ocfg.build_search_list_size, 128 for SIFT) — without this
    // override orion would search at L=128 while DiskANN searches at
    // L=48, conflating algorithm gap with hyperparameter gap and
    // making the orion numbers diverge from `orion`'s output.
    let run = experiment("single", k, overrides);
    assert_eq!(run.search_list_sizes.len(), 1, "thread-sweep requires one search L");
    let search_l = run.search_list_sizes[0];
    diskann_runner.set_search_list_size(search_l);
    orion_runner.set_search_list_size(search_l);

    // Re-calibrate `threshold` + `early_exit_limit` at the **search L**
    // (48) using **real test queries** — the build-time calibration ran
    // at L=128 (the build L) over the first 500 base vectors as warmup,
    // both of which diverge from how `orion` calibrates
    // (CALIB_L=48, 200 real test queries). Different calibration →
    // different convergence + early-exit behavior → ~10-20% QPS gap.
    // Match `orion`'s recipe exactly.
    orion_runner.recalibrate(&dataset.queries, calibration_samples(), k);
    let (thr, ee) = orion_runner.calibrated_params();
    println!(
        "Recalibrated at L={search_l}: threshold={thr:.2}, early_exit_limit={ee} (reference orion: threshold=0.15, early_exit_limit=15)",
    );

    let queries = &dataset.queries;
    // Sweep covers the full P/E topology curve on M4 Max (10P + 4E):
    //   T ≤ 10  — all on P-cores (peak per-thread throughput)
    //   T = 12  — 10 P + 2 E (first E-core spill)
    //   T = 14  — all physical cores
    //   T ≥ 16  — over-subscribed; scheduler swaps threads on/off cores
    // The shape across these thresholds is the whole story — going
    // straight from 8 → 16 hides the P-core saturation knee.
    let thread_counts: [usize; 9] = [1, 2, 4, 6, 8, 10, 12, 14, 16];
    let trials = run.trials;

    // Pin the driver thread on P-cores; rayon workers get the same
    // bump via `start_handler` below. Mirrors `orion`'s setup
    // — without it the macOS scheduler may demote workers to E-cores
    // (3-4× slower per op) under sustained pressure or thermal load.
    utils::set_thread_qos_user_interactive();

    // Pin the hot regions into RAM so warmup + timed trials see the
    // same resident pages — no page-in cost on the timed-side critical
    // path. Mirrors `orion`'s mlock setup.
    //
    // We pin: orion's f32 dataset, quantized sidecar (L2Q for SIFT/GIST),
    // PhasedGraph slab, plus the query batch. DiskANN's internal graph
    // sits behind `Box<dyn ANNInmemIndex>` and isn't directly addressable
    // from here — its first-trial pages are still warmed by the per-pool
    // warmup pass below, just not formally mlocked. If `RLIMIT_MEMLOCK`
    // is too low the calls log a warning and fall back to ordinary
    // paging without aborting.
    orion_runner.pin_hot_regions();
    {
        // Queries are a `Vec<Vec<f32>>`. The outer Vec's heap is small
        // (one `Vec<f32>` header per query); the actual f32 payload
        // lives in each inner Vec's heap allocation. Pinning the outer
        // Vec's contiguous header array is cheap and mostly symbolic
        // — the per-query payload pages are walked during search and
        // brought in naturally by the warmup pass. For a future
        // refactor that flattens `queries` to a single `Vec<f32>`,
        // this single mlock would also cover the payload.
        let q_ptr = queries.as_ptr() as *const u8;
        let q_len = queries.len() * std::mem::size_of::<Vec<f32>>();
        utils::mlock_bytes("queries (headers)", q_ptr, q_len);
    }

    println!(
        "\n─── Thread Sweep [{} {}pts, k={k}, L={search_l}, {} queries, {trials} trials] ───\n",
        dim_name,
        num_points,
        queries.len(),
    );
    println!(
        "  {:<8} {:<14} {:<10} {:<14} {:<10} {:<10}",
        "Threads", "DiskANN QPS", format!("D_R@{k}"), "Orion QPS", format!("O_R@{k}"), "Speedup",
    );
    println!("  {}", "─".repeat(72));

    // ── One-time scratch-pool sizing ──────────────────────────────────
    // `Orion::inmem_scratch_pool` is `get_or_init` — initialised
    // ONCE on first search with size `current_num_threads() + 5`. If we
    // let the T=1 pool's warmup initialise it (sized for 6 scratches),
    // every later thread count (T=8, T=16) reuses that under-sized pool
    // and threads block waiting for a free scratch. At T=16, 10 of 16
    // workers stall on the scratch queue → most of the QPS gap vs
    // `orion` came from this.
    //
    // Pre-warm at the max thread count so the scratch pool is sized for
    // the worst-case (max_threads + 5). Subsequent per-thread-count
    // pools all hit the cached pool with adequate scratch.
    {
        let max_threads = *thread_counts.iter().max().unwrap();
        let init_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .start_handler(|_| utils::set_thread_qos_user_interactive())
            .build()
            .expect("failed to build init pool");
        let _ = init_pool.install(|| orion_runner.search_batch(queries, k));
        let _ = init_pool.install(|| diskann_runner.search_batch(queries, k));
    }

    // Per-thread-count timed-region protocol — one pool built **once**
    // per `nt`, kept alive for warmup + all trials. Old code routed
    // through `search_batch_with_threads` which spawned a fresh
    // `ThreadPool` on every trial; that triggered OS thread churn
    // and fresh worker placement each call, costing ~5-10ms per trial
    // and adding huge variance. The pool also installs the QoS bump
    // on each worker so they all stay on P-cores.
    let sample_qps = |runner: &dyn runner::common::AlgorithmRunner,
                      pool: &rayon::ThreadPool|
     -> (Vec<f64>, f64) {
        let mut samples = Vec::with_capacity(trials);
        let mut recall = 0.0f64;
        // Real-query warmup inside the pool — primes prefetchers,
        // drives DVFS to peak P-state, gets the rayon work-stealing
        // queues into steady state. Single pass; trials are timed
        // separately.
        let _ = pool.install(|| runner.search_batch(queries, k));
        for _ in 0..trials {
            utils::flush_cache();
            let t = Instant::now();
            let results = pool.install(|| runner.search_batch(queries, k));
            let wall = t.elapsed();
            let qps = queries.len() as f64 / wall.as_secs_f64();
            let ids: Vec<Vec<u32>> = results.into_iter().map(|r| r.neighbors).collect();
            recall = metrics::recall::mean_recall(&ids, &dataset.ground_truth, k);
            samples.push(qps);
        }
        (samples, recall)
    };

    let mut diskann_qps_all: Vec<Vec<f64>> = Vec::new();
    let mut orion_qps_all: Vec<Vec<f64>> = Vec::new();
    let mut diskann_recall: Vec<f64> = Vec::new();
    let mut orion_recall: Vec<f64> = Vec::new();

    for &nt in &thread_counts {
        // Build one pool for this thread count; both runners reuse it
        // across warmup + all trials. `start_handler` runs on every
        // worker thread when rayon spawns it, applying the QoS bump
        // once per worker lifetime.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .start_handler(|_| utils::set_thread_qos_user_interactive())
            .build()
            .expect("failed to build thread-sweep pool");
        let (d_samples, d_r) = sample_qps(&diskann_runner, &pool);
        let (s_samples, s_r) = sample_qps(&orion_runner, &pool);
        let mut d_sorted = d_samples.clone();
        d_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut s_sorted = s_samples.clone();
        s_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d_med = d_sorted[trials / 2];
        let s_med = s_sorted[trials / 2];
        diskann_qps_all.push(d_samples);
        orion_qps_all.push(s_samples);
        diskann_recall.push(d_r);
        orion_recall.push(s_r);
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
    let orion_med: Vec<f64> = orion_qps_all.iter().map(|v| sort(v)[trials / 2]).collect();
    let diskann_min: Vec<f64> = diskann_qps_all.iter().map(|v| sort(v)[0]).collect();
    let orion_min: Vec<f64> = orion_qps_all.iter().map(|v| sort(v)[0]).collect();
    let diskann_max: Vec<f64> = diskann_qps_all
        .iter()
        .map(|v| sort(v)[trials - 1])
        .collect();
    let orion_max: Vec<f64> = orion_qps_all.iter().map(|v| sort(v)[trials - 1]).collect();

    let d1 = diskann_med[0];
    let s1 = orion_med[0];
    let d_speedup: Vec<f64> = diskann_med.iter().map(|q| q / d1).collect();
    let s_speedup: Vec<f64> = orion_med.iter().map(|q| q / s1).collect();
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
        "alpha_orion": ocfg.alpha,
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
        "orion": {
            "qps_median": orion_med,
            "qps_min": orion_min,
            "qps_max": orion_max,
            "qps_per_trial": orion_qps_all,
            "recall": orion_recall,
            "speedup": s_speedup,
            "efficiency": s_eff,
        },
    });
    let path = utils::result_path("thread_sweep", dim_name, k);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).expect("write json");
    println!("\nSaved {path}");
}

fn run_build_profile(dataset: &Dataset, _k: usize) {
    use orion::{build_diskann_index, Orion, DIM_100, DIM_128, DIM_32, DIM_960};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let ocfg = crate::config::load_dataset_config_by_dim(dimension).orion;

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    // 3 trials at full dataset size — variance on Vamana build time
    // is consistently < 5% across runs (R/L/α are the dominant
    // factors), so 3 produces a tight median without burning the
    // ~15 min/trial gist budget.
    const TRIALS: usize = 3;
    let mut diskann_times: Vec<f64> = Vec::with_capacity(TRIALS);
    let mut orion_graph_times: Vec<f64> = Vec::with_capacity(TRIALS);
    let mut orion_overhead_times: Vec<f64> = Vec::with_capacity(TRIALS);

    macro_rules! run_trials {
        ($N:literal) => {{
            for trial in 0..TRIALS {
                println!(
                    "[{}] Trial {}/{} (alpha={:.2}, degree={}, L_build={})",
                    dim_name,
                    trial + 1,
                    TRIALS,
                    ocfg.alpha,
                    ocfg.graph_degree,
                    ocfg.build_search_list_size,
                );

                // DiskANN baseline — **same α as Orion** (PA-aligned
                // per-dataset; e.g. SIFT α=1.15, GIST α=1.1). The α=2.0
                // recipe is never used in production, so the honest
                // overhead delta is α-matched: every build param
                // (R / L_build / α / num_threads / metric) matches
                // Orion, and the only difference is the
                // `compute_candidate_sets` toggle that drives the
                // PhasedGraph extras pass.
                let t_da = Instant::now();
                let _da = build_diskann_index(
                    &flat_base,
                    num_points,
                    dimension,
                    ocfg.alpha,
                    ocfg.graph_degree,
                    ocfg.build_search_list_size as u32,
                    false,
                    None,
                    None,
                    false,
                    0,
                )
                .expect("diskann build failed");
                let diskann_s = t_da.elapsed().as_secs_f64();
                drop(_da);

                // Orion: graph build with compute_candidate_sets=true.
                let result = build_diskann_index(
                    &flat_base,
                    num_points,
                    dimension,
                    ocfg.alpha,
                    ocfg.graph_degree,
                    ocfg.build_search_list_size as u32,
                    false,
                    None,
                    None,
                    true,
                    ocfg.max_extra,
                )
                .expect("orion build failed");
                let orion_graph_s = result.graph_build_time.as_secs_f64();

                // Orion overhead: Orion::new (extract + clustering + reorder).
                let t_ov = Instant::now();
                let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                let mut idx = Orion::<$N>::new(
                    empty_ds,
                    &result.partitions,
                    result.entry_point,
                    ocfg.graph_degree,
                    ocfg.max_extra,
                    None,
                    None,
                    None,
                    false,
                );
                idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                let orion_overhead_s = t_ov.elapsed().as_secs_f64();
                drop(idx);
                drop(result.index);

                println!(
                    "  DiskANN: {:.3}s  Orion graph: {:.3}s  Orion overhead: {:.3}s",
                    diskann_s, orion_graph_s, orion_overhead_s,
                );

                diskann_times.push(diskann_s);
                orion_graph_times.push(orion_graph_s);
                orion_overhead_times.push(orion_overhead_s);
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
    let g_mean = mean(&orion_graph_times);
    let g_std = stddev(&orion_graph_times, g_mean);
    let o_mean = mean(&orion_overhead_times);
    let o_std = stddev(&orion_overhead_times, o_mean);

    let json = serde_json::json!({
        "dataset": dim_name,
        "dimension": dimension,
        "num_points": num_points,
        "trials": TRIALS,
        "alpha": ocfg.alpha,
        "alpha_diskann": ocfg.alpha,
        "alpha_orion": ocfg.alpha,
        "diskann_s": d_mean,
        "diskann_s_std": d_std,
        "orion_graph_s": g_mean,
        "orion_graph_s_std": g_std,
        "orion_overhead_s": o_mean,
        "orion_overhead_s_std": o_std,
        "orion_total_s": g_mean + o_mean,
        "per_trial": {
            "diskann_s": diskann_times,
            "orion_graph_s": orion_graph_times,
            "orion_overhead_s": orion_overhead_times,
        },
    });

    let path = format!("visualizations/build_profile_{}.json", dim_name);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).expect("write json");
    println!("\nSaved {path}");
    println!(
        "  DiskANN (alpha={:.2}, no candidates): {:.3}s ± {:.3}s",
        ocfg.alpha, d_mean, d_std,
    );
    println!(
        "  Orion graph:                        {:.3}s ± {:.3}s",
        g_mean, g_std
    );
    println!(
        "  Orion overhead:                     {:.3}s ± {:.3}s",
        o_mean, o_std
    );
    println!(
        "  Orion / DiskANN:     {:.2}x   overhead: {:.1}% of orion total",
        (g_mean + o_mean) / d_mean,
        o_mean / (g_mean + o_mean) * 100.0,
    );
}

fn run_search_profile(dataset: &Dataset, k: usize, overrides: &config::SweepOverrides) {
    use orion::{build_diskann_index, Orion, DIM_100, DIM_128, DIM_32, DIM_960};

    let run = experiment("diagnostic", k, overrides);

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let ocfg = crate::config::load_dataset_config_by_dim(dimension).orion;

    macro_rules! profile {
        ($N:literal) => {{
            let alpha = ocfg.alpha;
            println!("Building Orion ({}-dim, alpha={})...", $N, alpha);
            let result = build_diskann_index(
                &flat_base,
                num_points,
                dimension,
                alpha,
                ocfg.graph_degree,
                ocfg.build_search_list_size as u32,
                false,
                None,
                None,
                true,
                ocfg.max_extra,
            )
            .expect("build failed");
            drop(result.index);

            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = Orion::<$N>::new(
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
            idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

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
            let _ = idx.ensure_quantized_dataset();
            for q in &queries[..queries.len().min(1000)] {
                crate::runner::cascade::search_compose::<$N>(
                    &idx,
                    q,
                    k,
                    run.search_list_sizes[0],
                    ocfg.window_size,
                    0.15,
                    1000,
                    crate::runner::cascade::PrefilterChoice::None,
                    crate::runner::cascade::AdmissionChoice::L2U8,
                    crate::runner::cascade::RerankChoice::F32,
                )
                .ok();
            }

            // Early-exit sweep: simulate search with different early_exit_count.
            // Uses search_diag which returns (results, converge_step, total_steps, p1_ndc, p2_ndc).
            println!(
                "\n─── Early Exit Sweep ({} queries) ───\n",
                queries.len()
            );
            println!(
                "  {:<12} {:<12} {:<14} {:<10} {:<10}",
                "EarlyExit", "Iter/query", "DistCalls/q", format!("R@{k}"), "QPS_est"
            );
            println!("  {}", "─".repeat(60));

            let ws = ocfg.window_size;

            // Auto-calibrate.
            let calib_sample: Vec<[f32; $N]> = queries[..queries.len().min(calibration_samples())].to_vec();
            let calib = idx
                .calibrate(&calib_sample, calibration_l(k), ws, orion::CalibrationConfig { k, ..Default::default() })
                .expect("calibrate failed");
            println!(
                "  Calibrated: threshold={:.2}, early_exit_limit={}",
                calib.threshold, calib.early_exit_limit
            );

            let thr = calib.threshold;
            let ee = calib.early_exit_limit;

            for &sls in &run.search_list_sizes {
                println!(
                    "\n─── L={}, ws={}, thr={:.2}, ee={} ───\n",
                    sls, ws, thr, ee
                );
                println!(
                    "  {:<12} {:<12} {:<14} {:<10} {:<10}",
                    "Mode", "Iter/query", "DistCalls/q", format!("R@{k}"), "QPS_est"
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
                            idx.search_diag(q, k, sls, ws, 0.0, 0).unwrap();
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

                // Orion with early exit = 0 (no early exit, but with convergence)
                {
                    let mut total_iter = 0u64;
                    let mut total_dist = 0u64;
                    let mut recall_sum = 0.0f64;
                    let t_start = std::time::Instant::now();
                    for (qi, q) in queries.iter().enumerate() {
                        let (res, _conv, steps, p1, p2) =
                            idx.search_diag(q, k, sls, ws, thr, ee).unwrap();
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
                        "orion",
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
    use orion::{Orion, DIM_100, DIM_128, DIM_32, DIM_960};

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let ocfg = crate::config::load_dataset_config_by_dim(dimension).orion;

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
        "\n═══ Memory Profile [{} {}pts, dim={}, α_orion={:.2}] ═══",
        dim_name, num_points, dimension, ocfg.alpha,
    );
    println!("  Baseline (base+queries+GT): {}\n", fmt(baseline));

    let num_threads = rayon::current_num_threads() as u32;

    // ── DiskANN (same α as Orion, no candidate sets) ──────────────────
    // α-matched against Orion below — every build param
    // (R / L_build / α / num_threads / metric) is identical and the
    // only difference is the `compute_candidate_sets` flag, so the
    // peak delta isolates the PhasedGraph extras cost.
    ALLOCATOR.reset_peak();
    let diskann_peak_abs = {
        let wp =
            IndexWriteParametersBuilder::new(ocfg.build_search_list_size as u32, ocfg.graph_degree)
                .with_alpha(ocfg.alpha)
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

    // ── Orion (config α, with candidate sets) ───────────────────
    ALLOCATOR.reset_peak();

    macro_rules! run_orion {
        ($N:literal) => {{
            let wp = IndexWriteParametersBuilder::new(
                ocfg.build_search_list_size as u32,
                ocfg.graph_degree,
            )
            .with_alpha(ocfg.alpha)
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
                .extract_graph_and_candidates(ocfg.max_extra)
                .expect("extract");
            let orion_peak_abs = ALLOCATOR.peak_bytes();
            drop(idx);

            let empty_ds = InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = Orion::<$N>::new(
                empty_ds,
                &partitions,
                entry_point,
                ocfg.graph_degree,
                ocfg.max_extra,
                None,
                None,
                None,
                false,
            );
            idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
            let orion_final_abs = ALLOCATOR.current_bytes();
            drop(idx);
            (orion_peak_abs, orion_final_abs)
        }};
    }

    let (orion_peak_abs, orion_final_abs) = match dimension {
        DIM_32 => run_orion!(32),
        DIM_100 => run_orion!(100),
        DIM_128 => run_orion!(128),
        DIM_960 => run_orion!(960),
        _ => panic!("Unsupported dimension for memory-profile: {dimension}"),
    };
    let orion_peak = orion_peak_abs.saturating_sub(baseline);
    let orion_final = orion_final_abs.saturating_sub(baseline);

    println!(
        "  DiskANN (α={:.2}, no candidates) peak: {}",
        ocfg.alpha,
        fmt(diskann_peak),
    );
    println!(
        "  Orion (α={:.2}) peak:         {}",
        ocfg.alpha,
        fmt(orion_peak)
    );
    println!("  Orion final:        {}", fmt(orion_final));
    let ratio = orion_peak as f64 / diskann_peak.max(1) as f64;
    println!(
        "  Peak ratio (Orion / DiskANN): {:.2}x    final / DiskANN: {:.2}x",
        ratio,
        orion_final as f64 / diskann_peak.max(1) as f64
    );

    let json = serde_json::json!({
        "dataset": dim_name,
        "dimension": dimension,
        "num_points": num_points,
        "alpha_orion": ocfg.alpha,
        "baseline_b": baseline,
        "diskann_peak_b": diskann_peak,
        "orion_peak_b": orion_peak,
        "orion_final_b": orion_final,
    });
    let path = format!("visualizations/memory_profile_{}.json", dim_name);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).expect("write json");
    println!("\nSaved {path}");
}

fn run_ads_comparison(dataset: &Dataset, k: usize, overrides: &config::SweepOverrides) {
    use crate::runner::common::{AlgorithmRunner, SearchResult};
    use crate::runner::{
        DiskANNAdsRunner, DiskANNRunner, OrionAdsRunner, OrionRunner,
    };
    use rayon::prelude::*;
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();

    let run = experiment("ads", k, overrides);
    let search_list_sizes = &run.search_list_sizes;
    let num_threads = run.threads;
    let trials = run.trials;

    let dim_name = match dimension {
        32 => "glove25",
        100 => "glove100",
        128 => "sift",
        960 => "gist",
        _ => "unknown",
    };

    let ocfg = config::load_dataset_config(dim_name).orion;
    let orion_alpha = ocfg.alpha;
    let st_degree = ocfg.graph_degree as usize;
    let orion_build_l = ocfg.build_search_list_size;
    let st_max_extra = ocfg.max_extra;
    let orion_ws = ocfg.window_size;

    // α-match DiskANN baselines to the per-dataset Orion topology.
    // The global `defaults.diskann` α=2.0 R=64 L=100 recipe is never
    // used in production and would conflate algorithm noise with
    // build-config noise (the same conflation we already fixed in
    // `run_build_profile` and `run_memory_profile`).
    let da_alpha = orion_alpha;
    let da_degree = st_degree as u32;
    let da_build_l = orion_build_l;

    // ADSampling ε: higher = tighter confidence, less speedup but safer.
    let ads_epsilon = 2.1_f32;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("thread pool");

    println!(
        "\n═══ ADSampling 4-way ({dim_name}, {num_points} pts, {num_threads}T, k={k}, ε_ads={ads_epsilon}) ═══\n"
    );

    // Per-variant cache paths — keyed by (dataset, n, R, L_build, α)
    // so cache slots are valid as long as the build params don't
    // change. ADS variants use the `_ads` suffix to distinguish from
    // the matched-α DiskANN / Orion caches produced by other binaries
    // (`diskann_sweep`, `orion`, the head-to-head harness).
    // First invocation pays the full build cost (~3 min per variant on
    // GIST PA-aligned); subsequent runs of the same config hit the
    // cache and skip the Vamana build entirely.
    let alpha_tag = format!("{:.2}", orion_alpha).replace('.', "_");
    let da_alpha_tag = format!("{:.2}", da_alpha).replace('.', "_");
    let diskann_cache = std::path::PathBuf::from("cache/diskann").join(format!(
        "{dim_name}_n{num_points}_r{da_degree}_l{da_build_l}_a{da_alpha_tag}_l2.bin"
    ));
    let diskann_ads_cache = std::path::PathBuf::from("cache/diskann").join(format!(
        "{dim_name}_n{num_points}_r{da_degree}_l{da_build_l}_a{da_alpha_tag}_l2_ads.bin"
    ));
    let orion_cache = std::path::PathBuf::from("cache/orion").join(format!(
        "{dim_name}_n{num_points}_r{st_degree}_l{orion_build_l}_a{alpha_tag}_ex{st_max_extra}.bin"
    ));
    let orion_ads_cache = std::path::PathBuf::from("cache/orion").join(format!(
        "{dim_name}_n{num_points}_r{st_degree}_l{orion_build_l}_a{alpha_tag}_ex{st_max_extra}_ads.bin"
    ));

    // Build the 4 runners. Each holds its own index — rotation is isolated per variant.
    println!("Building DiskANN (α={da_alpha}, R={da_degree})...");
    let mut da_runner = DiskANNRunner::new(da_build_l, da_degree, da_alpha);
    da_runner.set_cache_path(&diskann_cache);
    da_runner.build(&flat_base, num_points, dimension);

    println!("Building DiskANN+ADS (α={da_alpha}, R={da_degree})...");
    let mut da_ads_runner = DiskANNAdsRunner::new(da_build_l, da_degree, da_alpha, ads_epsilon);
    da_ads_runner.set_cache_path(&diskann_ads_cache);
    da_ads_runner.build(&flat_base, num_points, dimension);

    println!("Building Orion (α={orion_alpha}, R={st_degree}, extra={st_max_extra})...");
    let mut st_runner = OrionRunner::new(
        "Orion",
        orion_alpha,
        st_degree,
        orion_build_l,
        st_max_extra,
        orion_ws,
        crate::runner::cascade::PrefilterChoice::None,
        crate::runner::cascade::AdmissionChoice::L2U8,
        crate::runner::cascade::RerankChoice::F32,
    );
    st_runner.set_cache_path(&orion_cache);
    st_runner.build(&flat_base, num_points, dimension);

    println!("Building Orion+ADS (α={orion_alpha}, R={st_degree}, extra={st_max_extra})...");
    let mut st_ads_runner = OrionAdsRunner::new(
        "Orion+ADS",
        orion_alpha,
        st_degree,
        orion_build_l,
        st_max_extra,
        orion_ws,
    );
    st_ads_runner.set_cache_path(&orion_ads_cache);
    st_ads_runner.build(&flat_base, num_points, dimension);

    // Sweep helper: swap L on the runner, run batch search `trials` times, take median QPS.
    let sweep_runner = |runner: &mut dyn RunnerWithL, label: &str| -> Vec<(f64, f64)> {
        println!("\n{label}:");

        // ── Warm-up pass ─────────────────────────────────────────────
        // Drive everything that gets lazily set up on the first search
        // call so the L = first-sls measurement doesn't pay the cold-
        // start tax (previously the L = 16 row showed a 10-20× QPS
        // dip vs L = 20 on the Orion variants):
        //
        //   * `InMemScratchPool::get_or_init` (allocates
        //     `current_num_threads + 5` scratch buffers on first hit)
        //   * `ensure_quantized_dataset` lazy `.qds` / `.qdsl2kt`
        //     sidecar build for the admission tier
        //   * Cascade adapter construction inside
        //     `search_batch_compose` (boxes the dyn PrefilterStage /
        //     AdmissionStage / RerankStage trait objects)
        //   * `search_unified` monomorphisation hot path → fills the
        //     CPU's branch predictor + code cache for the (P, A, R)
        //     triple this runner picks
        //   * For the ADS variants: the scaled-partial-sum lookup
        //     table and rotator code path
        //   * DVFS ramps to peak P-state; rayon worker placement
        //     stabilises; the dataset / sidecar pages are dragged
        //     into L2/SLC so the next pass is true steady-state.
        //
        // Warm at the smallest sweep L so the warmup itself is cheap
        // (≈100 ms on GIST) and we don't burn a real trial's worth of
        // work on something we discard.
        let warmup_l = *search_list_sizes.first().unwrap_or(&16);
        runner.set_l(warmup_l);
        let _ = pool.install(|| {
            dataset
                .queries
                .par_iter()
                .map(|q| runner.search_ref().search(q, k))
                .collect::<Vec<SearchResult>>()
        });

        let mut rows: Vec<(f64, f64)> = Vec::new();
        for &sls in search_list_sizes {
            runner.set_l(sls);
            let mut qps_samples = Vec::with_capacity(trials);
            let mut recall = 0.0f64;
            for _ in 0..trials {
                let t = Instant::now();
                let results: Vec<SearchResult> = pool.install(|| {
                    dataset
                        .queries
                        .par_iter()
                        .map(|q| runner.search_ref().search(q, k))
                        .collect()
                });
                let wall = t.elapsed();
                qps_samples.push(dataset.queries.len() as f64 / wall.as_secs_f64());
                let ids: Vec<Vec<u32>> = results.into_iter().map(|r| r.neighbors).collect();
                recall = metrics::recall::mean_recall(&ids, &dataset.ground_truth, k);
            }
            qps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let qps = qps_samples[trials / 2];
            rows.push((recall, qps));
            println!("  L={sls:>4}  R@{k}={recall:.4}  QPS={qps:.0}");
        }
        rows
    };

    // Trait-object shim so the sweep helper can accept heterogeneous runners.
    trait RunnerWithL: Sync {
        fn set_l(&mut self, sls: usize);
        fn search_ref(&self) -> &dyn AlgorithmRunner;
    }
    impl RunnerWithL for DiskANNRunner {
        fn set_l(&mut self, sls: usize) {
            self.set_search_list_size(sls);
        }
        fn search_ref(&self) -> &dyn AlgorithmRunner {
            self
        }
    }
    impl RunnerWithL for DiskANNAdsRunner {
        fn set_l(&mut self, sls: usize) {
            self.set_search_list_size(sls);
        }
        fn search_ref(&self) -> &dyn AlgorithmRunner {
            self
        }
    }
    impl RunnerWithL for OrionRunner {
        fn set_l(&mut self, sls: usize) {
            self.set_search_list_size(sls);
        }
        fn search_ref(&self) -> &dyn AlgorithmRunner {
            self
        }
    }
    impl RunnerWithL for OrionAdsRunner {
        fn set_l(&mut self, sls: usize) {
            self.set_search_list_size(sls);
        }
        fn search_ref(&self) -> &dyn AlgorithmRunner {
            self
        }
    }

    let da_rows = sweep_runner(&mut da_runner, "DiskANN");
    let da_ads_rows = sweep_runner(&mut da_ads_runner, "DiskANN+ADS");
    let st_rows = sweep_runner(&mut st_runner, "Orion");
    let st_ads_rows = sweep_runner(&mut st_ads_runner, "Orion+ADS");

    // Emit JSON for the plot script.
    let json_path = utils::result_path("ads", dim_name, k);
    std::fs::create_dir_all("visualizations").ok();
    let fmt = |rows: &[(f64, f64)]| {
        rows.iter()
            .map(|(r, q)| format!("[{:.4}, {:.0}]", r, q))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let json = format!(
        "{{\n  \"dataset\": \"{dim_name}\",\n  \"dimension\": {dimension},\n  \"num_points\": {num_points},\n  \"threads\": {num_threads},\n  \"ads_epsilon\": {ads_epsilon},\n  \"diskann\": [{}],\n  \"diskann_ads\": [{}],\n  \"orion\": [{}],\n  \"orion_ads\": [{}]\n}}",
        fmt(&da_rows), fmt(&da_ads_rows), fmt(&st_rows), fmt(&st_ads_rows),
    );
    std::fs::write(&json_path, &json).expect("write json");
    println!("\nSaved to {json_path}");
}

fn run_cliff_profile(dataset: &Dataset) {
    use orion::algorithm::analysis::{
        annotate_bf_ranks, compute_cliff_stats, print_node_cliff_detail, summarize_cliff_ranks,
        summarize_cliff_stats,
    };
    use orion::{build_diskann_index, PhasedGraph};

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
    use orion::algorithm::analysis::{
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
    use orion::algorithm::search::convergence::SearchConvergenceChecker;
    use orion::{build_diskann_index, Orion};
    use vector::Metric;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let _k = 10;
    let ocfg = crate::config::load_dataset_config_by_dim(dimension).orion;

    macro_rules! run_profile {
        ($N:literal) => {{
            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            println!("Building Orion ({}-dim, alpha={})...", $N, ocfg.alpha);
            let result = build_diskann_index(
                &flat_base, num_points, dimension, ocfg.alpha, ocfg.graph_degree,
                ocfg.build_search_list_size as u32, false, None, None, true, ocfg.max_extra,
            ).expect("build failed");
            let entry = result.entry_point;
            drop(result.index);
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = Orion::<$N>::new(
                empty_ds, &result.partitions, entry,
                ocfg.graph_degree, ocfg.max_extra, None, None, None, false,
            );
            idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

            let graph = &idx.graph;
            let ds = &idx.dataset;

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
                    let mut scc = SearchConvergenceChecker::new(ws, eps);
                    let mut seen = vec![false; num_points];
                    let mut pq = diskann::model::NeighborPriorityQueue::with_capacity(sls);

                    seen[entry as usize] = true;
                    let ed = ds.get_vertex(entry).unwrap().compare(&query_vertex, Metric::L2);
                    pq.insert(DNeighbor::new(entry, ed));

                    let mut prev_admitted: usize = 1; // optimistic start

                    while pq.has_notvisited_node() {
                        let cur = pq.closest_notvisited();
                        let id = cur.id as usize;
                        let converged = scc.update(prev_admitted);

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
                    "alpha": ocfg.alpha,
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
            let max_deg = ocfg.graph_degree as usize;
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
                "alpha": ocfg.alpha,
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

fn run_extra_profile(dataset: &Dataset, k: usize, overrides: &config::SweepOverrides) {
    use orion::{build_diskann_index, Orion, DIM_100, DIM_128, DIM_32, DIM_960};
    let ocfg = crate::config::load_dataset_config_by_dim(dataset.dimension).orion;
    let run = experiment("single", k, overrides);
    assert_eq!(run.search_list_sizes.len(), 1, "extra-profile requires one search L");
    let search_l = run.search_list_sizes[0];

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

            let alpha = ocfg.alpha;

            // Sweep max_extra = 0, 2, 4, 8, 16, 32.
            println!(
                "\n═══ Extra Candidate Profile ({}-dim, alpha={}, {} queries) ═══\n",
                $N,
                alpha,
                queries.len()
            );
            println!(
                "  {:<10} {:<10} {:<10} {:<12} {:<14} {:<10} {:<10}",
                "MaxExtra", "AvgExtra", "AvgLocal", "Iter/query", "DistCalls/q", format!("R@{k}"), "QPS"
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
                let mut idx = Orion::<$N>::new(
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
                idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);

                let avg_extra = (0..num_points)
                    .map(|i| idx.graph.extra_count(i))
                    .sum::<usize>() as f64
                    / num_points as f64;
                let avg_local = (0..num_points)
                    .map(|i| idx.graph.local_count(i))
                    .sum::<usize>() as f64
                    / num_points as f64;

                // Calibrate and search.
                let calib = idx
                    .calibrate(&queries[..queries.len().min(calibration_samples())], calibration_l(k), ocfg.window_size, orion::CalibrationConfig { k, ..Default::default() })
                    .expect("calibrate");

                // Run search_diag for detailed stats.
                let mut total_iter = 0u64;
                let mut total_dist = 0u64;
                let mut recall_sum = 0.0f64;
                let t_start = std::time::Instant::now();
                for (qi, q) in queries.iter().enumerate() {
                    let (res, _, steps, p1, p2) = idx
                        .search_diag(q, k, search_l, ocfg.window_size, calib.threshold, calib.early_exit_limit)
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
                orion::PhasedGraph::build_from_partitions(&result_cov.partitions, 32, 32);

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

fn run_calibration_diag(dataset: &Dataset, k: usize) {
    use orion::{build_diskann_index, Orion, DIM_100, DIM_128, DIM_32, DIM_960};
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    let ocfg = crate::config::load_dataset_config_by_dim(dimension).orion;

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

            // Build Orion
            let t_build = Instant::now();
            let result = build_diskann_index(
                &flat_base, num_points, dimension, ocfg.alpha,
                ocfg.graph_degree, ocfg.build_search_list_size as u32,
                false, None, None, true, ocfg.max_extra,
            ).expect("build failed");
            let graph_build_s = result.graph_build_time.as_secs_f64();
            drop(result.index);

            let t_staged = Instant::now();
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = Orion::<$N>::new(
                empty_ds, &result.partitions, result.entry_point,
                ocfg.graph_degree, ocfg.max_extra, None, None, None, false,
            );
            idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
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
            let calib_qs: Vec<[f32; $N]> = queries[..queries.len().min(calibration_samples())].to_vec();
            let diag = idx.calibrate_with_diagnostics(&calib_qs, config::load_sweep_config().calibration.diagnostic_search_list_size(k), ocfg.window_size, orion::CalibrationConfig {
                k, metric: ocfg.admission.calibration_metric(),
            })
                .expect("calibrate failed");

            // Write JSON
            let json = serde_json::json!({
                "dataset": dim_name,
                "dimension": $N,
                "num_points": num_points,
                "alpha": ocfg.alpha,
                "admission_rates": diag.admission_rates,
                "useful_gaps": diag.useful_gaps,
                "tail_gaps": diag.tail_gaps,
                "topk_coverage_by_step": diag.topk_coverage_by_step,
                "threshold": diag.params.threshold,
                "early_exit_limit": diag.params.early_exit_limit,
                "build": {
                    "diskann_s": diskann_build_s,
                    "orion_graph_s": graph_build_s,
                    "orion_overhead_s": staged_overhead_s,
                    "orion_total_s": total_build_s,
                }
            });

            let path = utils::result_path("calibration_diag", dim_name, k);
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
                .expect("write json");
            println!("Saved {path}");
            println!("  threshold={:.2}, early_exit_limit={}", diag.params.threshold, diag.params.early_exit_limit);
            println!("  DiskANN build: {:.2}s", diskann_build_s);
            println!("  Orion graph build: {:.2}s  overhead: {:.2}s  total: {:.2}s",
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

/// Convergence-side ablation: isolates the contributions of the
/// **early-exit** and **extras** machinery on top of the per-dataset
/// default cascade. Emits a 4-series QPS-vs-Recall curve per dataset
/// (companion panel to `run_cascade_ablation`, which slices the
/// cascade *tiers* instead).
///
/// Variants — a clean 2×2 factorial on the convergence-side knobs,
/// **all four sharing the same cascade backbone, base graph, and
/// calibration**:
///
///   * `origin`        — cascade + convergence ON, **no extras + no
///                       early-stop** (the bare-cascade reference).
///                       This is NOT Microsoft Vamana — that headline
///                       gain lives in the 3-engine sweep panel. Here
///                       origin isolates "what does the cascade alone
///                       achieve without either convergence-side
///                       optimization?", so deltas vs `full`/`no_ee`/
///                       `no_extra` cleanly attribute to each knob.
///   * `no_extra`      — cascade + early-stop, **extras off**.
///                       Isolates the marginal value of early-stop
///                       on top of bare-cascade.
///   * `no_early_stop` — cascade + extras, **`ee = MAX`**.
///                       Isolates the marginal value of extras on top
///                       of bare-cascade.
///   * `full`          — cascade + early-stop + extras (production).
///
/// Earlier versions of this ablation used Microsoft Vamana as origin
/// (which conflates "no cascade backbone" with "no convergence
/// machinery" — the 4-10× gap was mostly the cascade kernels, not the
/// convergence knobs being ablated) and built a separate `max_extra=0`
/// graph for `no_extra` (which conflated build-time topology with the
/// search-time extras contribution). This design eliminates both
/// confounds: one graph, one calibration, all variants on the same
/// search path with only the two convergence-side flags toggled.
fn run_ablation(dataset: &Dataset, k: usize, overrides: &config::SweepOverrides) {
    use rayon::prelude::*;
    use orion::{
        build_diskann_index, Orion, DIM_100, DIM_128, DIM_1536, DIM_32, DIM_768, DIM_960,
    };
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    // Disambiguate dim=128 by num_points: sift (1M) vs deep10m (10M).
    // Other dimensions are unambiguous across the panel set.
    let dim_name = match (dimension, num_points) {
        (32, _) => "glove25",
        (100, _) => "glove100",
        (128, n) if n >= 5_000_000 => "deep10m",
        (128, _) => "sift",
        (768, _) => "msmarco_bert_1M",
        (960, _) => "gist",
        (1536, _) => "wiki_ada_1M",
        _ => "unknown",
    };
    let docfg = crate::config::load_dataset_config(dim_name);
    let ocfg = docfg.orion;
    let run = experiment("ablation", k, overrides);
    let num_threads = run.threads;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    // Same L schedule as `run_cascade_ablation` for cross-panel comparability.
    let search_list_sizes = run.search_list_sizes.as_slice();
    let trials = run.trials;

    macro_rules! run_ablation_inner {
        ($N:literal) => {{
            use crate::runner::cascade::{search_compose, PrefilterChoice};

            let queries: Vec<[f32; $N]> = dataset
                .queries
                .iter()
                .map(|q| {
                    let mut arr = [0f32; $N];
                    arr.copy_from_slice(&q[..$N]);
                    arr
                })
                .collect();

            // ── Resolve cache path (PA-mode naming, shared with cascade-ablation). ──
            let local_pct: usize = std::env::var("ORION_LOCAL_PCT")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&v| (1..=100).contains(&v))
                .unwrap_or(60);
            let alpha_tag = format!("{:.2}", ocfg.alpha).replace('.', "_");
            let cache_dir = std::path::PathBuf::from("cache/orion_parlayann");
            std::fs::create_dir_all(&cache_dir).ok();
            // Build-or-load the production orion graph (default
            // `max_extra`). All 4 ablation variants share this one
            // graph — `no-extra` uses the same graph with the
            // `INCLUDE_EXTRAS` toggle flipped off at search time
            // (the post-convergence branch falls back to local+remote
            // instead of walking local+extra). Sharing one graph
            // eliminates the build-time topology confound of the
            // earlier `max_extra=0` rebuild approach.
            let cache_path = cache_dir.join(format!(
                "{}_n{}_r{}_l{}_a{}_ex{}_pct{}.bin",
                dim_name,
                num_points,
                ocfg.graph_degree,
                ocfg.build_search_list_size,
                alpha_tag,
                ocfg.max_extra,
                local_pct,
            ));
            let pgraph_path = cache_path.with_extension("pgraph");
            println!(
                "Loading orion graph ({}-dim, max_extra={})...",
                $N, ocfg.max_extra
            );
            let t_load = Instant::now();
            // Wrap in `pool.install` so the Vamana builder (which
            // queries `rayon::current_num_threads()`) sees the
            // 8-thread pool, not the global all-cores pool.
            let idx: Orion<$N> = pool.install(|| {
                if cache_path.exists() && pgraph_path.exists() {
                    println!("  → cache hit: loading {:?}", pgraph_path);
                    let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                    let mut idx = Orion::<$N>::load_from_cache(&cache_path, empty_ds)
                        .expect("load_from_cache failed");
                    idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                    idx
                } else if let Ok(staged_file_path) = std::env::var("ORION_STAGED_FILE") {
                    println!("  → import PA .staged from {}", staged_file_path);
                    let input = crate::runner::parlayann_bridge::load_from_staged_file(
                        &staged_file_path,
                        &flat_base[..num_points * $N],
                        $N,
                    )
                    .expect("parlayann_bridge::load_from_staged_file failed");
                    let max_extra_from_file = input.max_extra as usize;
                    let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                    let mut idx = Orion::<$N>::new(
                        empty_ds,
                        &input.partitions,
                        input.entry_point,
                        ocfg.graph_degree,
                        max_extra_from_file,
                        None,
                        None,
                        Some(cache_path.clone()),
                        true,
                    );
                    idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                    idx
                } else {
                    println!(
                        "  → build in-process, will cache to {:?}",
                        pgraph_path
                    );
                    let result = build_diskann_index(
                        &flat_base,
                        num_points,
                        dimension,
                        ocfg.alpha,
                        ocfg.graph_degree,
                        ocfg.build_search_list_size as u32,
                        false,
                        None,
                        None,
                        true,
                        ocfg.max_extra,
                    )
                    .expect("build failed");
                    let entry = result.entry_point;
                    drop(result.index);
                    let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                    let mut idx = Orion::<$N>::new(
                        empty_ds,
                        &result.partitions,
                        entry,
                        ocfg.graph_degree,
                        ocfg.max_extra,
                        None,
                        None,
                        Some(cache_path.clone()),
                        true,
                    );
                    idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                    idx
                }
            });
            println!("  → ready in {:.1}s", t_load.elapsed().as_secs_f64());

            // Single calibration shared across all 4 variants —
            // they all run on the same graph topology, so threshold
            // / early-exit derive identically.
            let calib_qs: Vec<[f32; $N]> =
                queries[..queries.len().min(calibration_samples())].to_vec();
            let calib = idx
                .calibrate(&calib_qs, calibration_l(k), ocfg.window_size, orion::CalibrationConfig {
                    k, metric: ocfg.admission.calibration_metric(),
                })
                .expect("calibrate failed");
            let (thr, ee) = (calib.threshold, calib.early_exit_limit);
            println!(
                "Calibrated (shared across all variants): threshold={:.2}, early_exit_limit={}\n",
                thr, ee
            );

            // Pre-materialise sidecars for the chosen cascade so per-L
            // QPS isn't biased by first-touch sidecar build. Sidecar
            // build is rayon-parallel (KMeans / k-means++ inner loops);
            // run it inside the 8-thread pool for consistency with the
            // measurement phase.
            pool.install(|| {
                crate::runner::cascade::pin_admission(&idx, ocfg.admission);
                match ocfg.prefilter {
                    PrefilterChoice::None => {}
                    PrefilterChoice::Jl => {
                        let _ = idx.ensure_quantized_dataset_jl();
                    }
                    PrefilterChoice::JlHadamard => {
                        let _ = idx.ensure_quantized_dataset_jl_hadamard();
                    }
                    PrefilterChoice::Rabitq => {
                        let _ = idx.ensure_quantized_dataset_rabitq();
                    }
                }
            });

            // Generic measure closure: time a search fn over `queries`
            // × `search_list_sizes` × `trials`, return median QPS per L.
            // Each call does ONE warmup pass at warmup_l before timing.
            let warmup_l = *search_list_sizes.first().unwrap_or(&16);
            let measure = |label: &str,
                           search_fn: &(dyn Fn(&[f32; $N], usize) -> Vec<u32> + Sync)|
             -> Vec<(f64, f64)> {
                println!("  [{label}]");
                pool.install(|| {
                    queries.par_iter().for_each(|q| {
                        let _ = search_fn(q, warmup_l);
                    });
                });
                let mut data = Vec::new();
                for &sls in search_list_sizes {
                    let mut qps_samples = Vec::with_capacity(trials);
                    let mut recall = 0.0f64;
                    for _ in 0..trials {
                        self::utils::flush_cache();
                        let t = Instant::now();
                        let results: Vec<Vec<u32>> = pool.install(|| {
                            queries.par_iter().map(|q| search_fn(q, sls)).collect()
                        });
                        let wall = t.elapsed();
                        qps_samples.push(queries.len() as f64 / wall.as_secs_f64());
                        recall =
                            metrics::recall::mean_recall(&results, &dataset.ground_truth, k);
                    }
                    qps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    data.push((recall, qps_samples[trials / 2]));
                }
                data
            };

            println!("\nRunning ablation sweep...\n");

            // ── 0. Origin: cascade-only (no extras, no early-stop) ──
            // Same cascade as the other three variants; both convergence-
            // side knobs disabled (`INCLUDE_EXTRAS=false`, `ee=MAX`).
            // This is the 2×2 factorial's bare reference point — deltas
            // to the other three attribute cleanly to (a) extras alone,
            // (b) early-stop alone, (c) the combined effect.
            orion::algorithm::search::set_include_extras(false);
            let origin_data = measure("origin (cascade-only)", &|q, sls| {
                search_compose::<$N>(
                    &idx,
                    q,
                    k,
                    sls,
                    ocfg.window_size,
                    thr,
                    usize::MAX,
                    ocfg.prefilter,
                    ocfg.admission,
                    ocfg.rerank,
                )
                .unwrap_or_default()
            });

            // ── 1. Full ──
            // Default INCLUDE_EXTRAS=true; convergence + early-exit on.
            orion::algorithm::search::set_include_extras(true);
            let full_data = measure("full", &|q, sls| {
                search_compose::<$N>(
                    &idx,
                    q,
                    k,
                    sls,
                    ocfg.window_size,
                    thr,
                    ee,
                    ocfg.prefilter,
                    ocfg.admission,
                    ocfg.rerank,
                )
                .unwrap_or_default()
            });

            // ── 2. No early-stop (convergence on, ee = MAX) ──
            // Convergence detection still fires; just doesn't terminate
            // the beam early. Extras still walked post-convergence.
            orion::algorithm::search::set_include_extras(true);
            let no_ee_data = measure("no-early-stop", &|q, sls| {
                search_compose::<$N>(
                    &idx,
                    q,
                    k,
                    sls,
                    ocfg.window_size,
                    thr,
                    usize::MAX,
                    ocfg.prefilter,
                    ocfg.admission,
                    ocfg.rerank,
                )
                .unwrap_or_default()
            });

            // ── 3. No extra — same graph, post-convergence branch
            //     falls back to local+remote (see `INCLUDE_EXTRAS`
            //     plumbing in `algorithm::search::mod`).
            orion::algorithm::search::set_include_extras(false);
            let no_extra_data = measure("no-extra", &|q, sls| {
                search_compose::<$N>(
                    &idx,
                    q,
                    k,
                    sls,
                    ocfg.window_size,
                    thr,
                    ee,
                    ocfg.prefilter,
                    ocfg.admission,
                    ocfg.rerank,
                )
                .unwrap_or_default()
            });
            // Restore default so any subsequent code (other dataset
            // dispatch arms, downstream tests in the same process)
            // gets production semantics.
            orion::algorithm::search::set_include_extras(true);

            // ── Summary table ──
            println!(
                "\n{:<6} {:<22} {:<22} {:<22} {:<22}",
                "L",
                "origin (R/QPS)",
                "full (R/QPS)",
                "no-ee (R/QPS)",
                "no-extra (R/QPS)"
            );
            println!("{}", "─".repeat(96));
            for (i, &sls) in search_list_sizes.iter().enumerate() {
                println!(
                    "L={:<4} {:.3}/{:<12.0}  {:.3}/{:<12.0}  {:.3}/{:<12.0}  {:.3}/{:<12.0}",
                    sls,
                    origin_data[i].0,
                    origin_data[i].1,
                    full_data[i].0,
                    full_data[i].1,
                    no_ee_data[i].0,
                    no_ee_data[i].1,
                    no_extra_data[i].0,
                    no_extra_data[i].1,
                );
            }

            // ── Save JSON ──
            let to_json = |data: &[(f64, f64)]| -> Vec<(f64, f64)> {
                data.iter().map(|&(r, q)| (r, q)).collect::<Vec<_>>()
            };
            let json = serde_json::json!({
                "dataset": dim_name,
                "dimension": $N,
                "num_points": num_points,
                "threads": num_threads,
                "search_list_sizes": search_list_sizes,
                "k": k,
                "sweep": run,
                "window_size": ocfg.window_size,
                "default_cascade": {
                    "prefilter":  format!("{:?}", ocfg.prefilter),
                    "admission":  format!("{:?}", ocfg.admission),
                    "rerank":     format!("{:?}", ocfg.rerank),
                },
                "origin":         to_json(&origin_data),
                "full":           to_json(&full_data),
                "no_early_stop":  to_json(&no_ee_data),
                "no_extra":       to_json(&no_extra_data),
            });
            let path = utils::result_path("ablation", dim_name, k);
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
                .expect("write json");
            println!("\nSaved {path}");
        }};
    }

    match dimension {
        DIM_32 => run_ablation_inner!(32),
        DIM_100 => run_ablation_inner!(100),
        DIM_128 => run_ablation_inner!(128),
        DIM_768 => run_ablation_inner!(768),
        DIM_960 => run_ablation_inner!(960),
        DIM_1536 => run_ablation_inner!(1536),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}

/// Cascade-stage ablation: measure per-stage QPS / recall contribution
/// of the prefilter, admission, and rerank stages.
#[allow(unused_imports)] // DIM_768 / DIM_1536 only used by the macro dispatch.
///
/// Holds the graph + admission tier fixed at the per-dataset defaults
/// from `sweep.yaml`. Sweeps four cascade variants:
///
///   1. **full**            `(default_prefilter, default_admission, default_rerank)`
///   2. **no-prefilter**    `(None, default_admission, default_rerank)`
///   3. **no-rerank**       `(default_prefilter, default_admission, None)`
///   4. **admission-only**  `(None, default_admission, None)`
///
/// Reading the panel:
///   * **(full − no-prefilter) at iso-recall**  → prefilter's QPS contribution
///   * **(no-rerank − full) at iso-L**          → rerank's QPS cost (and recall delta)
///   * **(admission-only − full)**              → combined effect of both auxiliary stages
///
/// All four variants share one built `Orion` and one
/// `calibrate()` call so thermal / cache / build-jitter drift can't
/// account for the QPS gaps.
fn run_cascade_ablation(dataset: &Dataset, k: usize, overrides: &config::SweepOverrides) {
    use rayon::prelude::*;
    use orion::{
        build_diskann_index, Orion, DIM_100, DIM_128, DIM_1536, DIM_32, DIM_768, DIM_960,
    };
    use std::time::Instant;

    let num_points = dataset.num_base();
    let dimension = dataset.dimension;
    let flat_base = dataset.base_flat();
    // Disambiguate dim=128 by num_points: sift (1M) vs deep10m (10M).
    let dim_name = match (dimension, num_points) {
        (32, _) => "glove25",
        (100, _) => "glove100",
        (128, n) if n >= 5_000_000 => "deep10m",
        (128, _) => "sift",
        (768, _) => "msmarco_bert_1M",
        (960, _) => "gist",
        (1536, _) => "wiki_ada_1M",
        _ => "unknown",
    };
    let ocfg = crate::config::load_dataset_config(dim_name).orion;
    let run = experiment("ablation", k, overrides);
    let num_threads = run.threads;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    // Same L schedule as `run_ablation` for cross-figure comparability.
    let search_list_sizes = run.search_list_sizes.as_slice();
    let trials = run.trials;

    macro_rules! run_cascade_ablation_inner {
        ($N:literal) => {{
            use crate::runner::cascade::{
                search_compose, PrefilterChoice, RerankChoice,
            };

            let queries: Vec<[f32; $N]> = dataset.queries.iter().map(|q| {
                let mut arr = [0f32; $N];
                arr.copy_from_slice(&q[..$N]);
                arr
            }).collect();

            // ── Build or load Orion once (shared across all 4 variants). ──
            //
            // Cache resolution mirrors `orion.rs` bin's PA-mode
            // naming so artifacts are interchangeable: once the bin has
            // populated `cache/orion_parlayann/<ds>_n…_r…_l…_a…_ex…_pct60.{bin,pgraph}`,
            // this benchmark loads the same graph in ~1–3s instead of
            // rebuilding for 30–120s. Three paths, in priority order:
            //   1. `.bin`+`.pgraph` on disk        → `load_from_cache` (fast)
            //   2. `ORION_STAGED_FILE` env set    → import PA `.staged`, save to cache
            //   3. neither                          → in-process Vamana, save to cache
            //
            // This makes the measured QPS reflect the *steady-state*
            // production setup (graph loaded once at startup, kept hot)
            // rather than first-launch + cold build.
            let local_pct: usize = std::env::var("ORION_LOCAL_PCT")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&v| (1..=100).contains(&v))
                .unwrap_or(60);
            let alpha_tag = format!("{:.2}", ocfg.alpha).replace('.', "_");
            let cache_dir = std::path::PathBuf::from("cache/orion_parlayann");
            let cache_path = cache_dir.join(format!(
                "{}_n{}_r{}_l{}_a{}_ex{}_pct{}.bin",
                dim_name, num_points, ocfg.graph_degree, ocfg.build_search_list_size,
                alpha_tag, ocfg.max_extra, local_pct,
            ));
            std::fs::create_dir_all(&cache_dir).ok();
            let pgraph_path = cache_path.with_extension("pgraph");

            println!(
                "Orion ({}-dim, alpha={}, default cascade: {:?}/{:?}/{:?})...",
                $N, ocfg.alpha, ocfg.prefilter, ocfg.admission, ocfg.rerank
            );
            let t_build = Instant::now();
            // All graph load/build wrapped in `pool.install` so the
            // Vamana builder (which reads `rayon::current_num_threads()`)
            // sees our 8-thread pool, not the global all-cores pool.
            // Keeps build-phase thermal/contention profile consistent
            // with the measurement phase.
            let idx: Orion<$N> = pool.install(|| {
                if cache_path.exists() && pgraph_path.exists() {
                    println!("  → cache hit: loading PhasedGraph from {:?}", pgraph_path);
                    let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                    let mut idx = Orion::<$N>::load_from_cache(&cache_path, empty_ds)
                        .expect("load_from_cache failed");
                    idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                    idx
                } else if let Ok(staged_file_path) = std::env::var("ORION_STAGED_FILE") {
                    println!(
                        "  → importing ParlayANN .staged from {} (cache will be \
                         written to {:?})",
                        staged_file_path, pgraph_path,
                    );
                    let input = crate::runner::parlayann_bridge::load_from_staged_file(
                        &staged_file_path, &flat_base[..num_points * $N], $N,
                    ).expect("parlayann_bridge::load_from_staged_file failed");
                    let entry = input.entry_point;
                    // Use header's max_extra (the variable per-node count cap
                    // from the C++ partition rule) rather than ocfg's static
                    // value — keeps the imported graph faithful.
                    let max_extra_from_file = input.max_extra as usize;
                    let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                    let mut idx = Orion::<$N>::new(
                        empty_ds, &input.partitions, entry,
                        ocfg.graph_degree, max_extra_from_file,
                        None, None, Some(cache_path.clone()), true,
                    );
                    idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                    idx
                } else {
                    println!(
                        "  → no PA cache / ORION_STAGED_FILE — building Vamana \
                         in-process. Will save to {:?} for subsequent runs.",
                        pgraph_path,
                    );
                    let result = build_diskann_index(
                        &flat_base, num_points, dimension, ocfg.alpha,
                        ocfg.graph_degree, ocfg.build_search_list_size as u32,
                        false, None, None, true, ocfg.max_extra,
                    ).expect("build failed");
                    let entry = result.entry_point;
                    drop(result.index);
                    let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
                    let mut idx = Orion::<$N>::new(
                        empty_ds, &result.partitions, entry,
                        ocfg.graph_degree, ocfg.max_extra,
                        None, None, Some(cache_path.clone()), true,
                    );
                    idx.dataset = rebuild_dataset::<$N>(&flat_base, num_points);
                    idx
                }
            });
            println!("  → graph ready in {:.1}s", t_build.elapsed().as_secs_f64());

            // Calibrate once with the default cascade settings.
            let calib_qs: Vec<[f32; $N]> = queries[..queries.len().min(calibration_samples())].to_vec();
            let calib = idx.calibrate(&calib_qs, calibration_l(k), ocfg.window_size, orion::CalibrationConfig {
                k, metric: ocfg.admission.calibration_metric(),
            }).expect("calibrate failed");
            let thr = calib.threshold;
            let ee = calib.early_exit_limit;
            println!("Calibrated: threshold={:.2}, early_exit_limit={}\n", thr, ee);

            // Materialise every sidecar the four variants might touch
            // BEFORE the timed sweep so per-variant build latency
            // doesn't contaminate the first L's QPS reading. The
            // admission sidecar is the only sidecar that's always
            // needed; the prefilter sidecar is needed only when the
            // default prefilter isn't None. Wrapped in `pool.install`
            // so the rayon-parallel KMeans/centroid kernels respect
            // our 8-thread pool rather than the global all-cores pool.
            pool.install(|| {
                match ocfg.admission {
                    crate::runner::cascade::AdmissionChoice::L2U8 => {
                        let _ = idx.ensure_quantized_dataset();
                    }
                    crate::runner::cascade::AdmissionChoice::L2U16 => {
                        let _ = idx.ensure_quantized_dataset_l2_u16();
                    }
                    crate::runner::cascade::AdmissionChoice::L2Kt => {
                        let _ = idx.ensure_quantized_dataset_l2_kt();
                    }
                    crate::runner::cascade::AdmissionChoice::MipsI8 => {
                        let _ = idx.ensure_quantized_dataset_mips();
                    }
                    crate::runner::cascade::AdmissionChoice::MipsI16 => {
                        let _ = idx.ensure_quantized_dataset_mips_i16();
                    }
                    crate::runner::cascade::AdmissionChoice::AdsF32 => {}
                }
                match ocfg.prefilter {
                    PrefilterChoice::None => {}
                    PrefilterChoice::Jl => {
                        let _ = idx.ensure_quantized_dataset_jl();
                    }
                    PrefilterChoice::JlHadamard => {
                        let _ = idx.ensure_quantized_dataset_jl_hadamard();
                    }
                    PrefilterChoice::Rabitq => {
                        let _ = idx.ensure_quantized_dataset_rabitq();
                    }
                }
            });


            // Closure: time `search_compose` over `queries` × `search_list_sizes`
            // × `trials`, return median QPS per L plus the matching recall.
            //
            // Each variant gets its own warmup pass at the smallest sweep L
            // before the timed loop. This drives DVFS to peak P-state, primes
            // prefetchers + the rayon worker pool, and pages in the
            // cascade-specific sidecars (different prefilter/rerank choices
            // touch different byte regions). Mirrors the warmup pattern in
            // `orion.rs::sweep_one_cascade` so cascade-ablation QPS
            // is comparable with the published orion numbers.
            let warmup_l = *search_list_sizes.first().unwrap_or(&16);
            let measure = |label: &str, pre: PrefilterChoice, rerank: RerankChoice|
                -> Vec<(f64, f64)>
            {
                println!("  [{label}] prefilter={:?}, admission={:?}, rerank={:?}",
                    pre, ocfg.admission, rerank);
                // Single warmup pass at the smallest L — cheap enough that
                // it doesn't materially extend the run, deep enough that
                // every page the timed sweep will touch gets brought into
                // RAM under the right cascade dispatch.
                pool.install(|| {
                    queries.par_iter().for_each(|q| {
                        let _ = search_compose::<$N>(
                            &idx, q, k, warmup_l, ocfg.window_size, thr, ee,
                            pre, ocfg.admission, rerank,
                        );
                    });
                });
                let mut out = Vec::with_capacity(search_list_sizes.len());
                for &sls in search_list_sizes {
                    let mut qps_samples = Vec::with_capacity(trials);
                    let mut recall = 0.0f64;
                    for _ in 0..trials {
                        self::utils::flush_cache();
                        let t = Instant::now();
                        let results: Vec<Vec<u32>> = pool.install(|| {
                            queries.par_iter().map(|q| {
                                search_compose::<$N>(
                                    &idx, q, k, sls, ocfg.window_size, thr, ee,
                                    pre, ocfg.admission, rerank,
                                )
                                .unwrap_or_default()
                            }).collect()
                        });
                        let wall = t.elapsed();
                        qps_samples.push(queries.len() as f64 / wall.as_secs_f64());
                        recall = metrics::recall::mean_recall(
                            &results, &dataset.ground_truth, k);
                    }
                    qps_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    out.push((recall, qps_samples[trials / 2]));
                }
                out
            };

            // ── Four variants. ──
            println!("\nRunning cascade-stage ablation sweep...\n");
            let full_data           = measure("full",            ocfg.prefilter,  ocfg.rerank);
            let no_pf_data          = measure("no-prefilter",    PrefilterChoice::None, ocfg.rerank);
            let no_rerank_data      = measure("no-rerank",       ocfg.prefilter,  RerankChoice::None);
            let admission_only_data = measure("admission-only",  PrefilterChoice::None, RerankChoice::None);

            // ── Print summary table. ──
            println!(
                "\n{:<6} {:<22} {:<22} {:<22} {:<22}",
                "L", "full (R/QPS)", "no-pre (R/QPS)", "no-rerank (R/QPS)", "adm-only (R/QPS)"
            );
            println!("{}", "─".repeat(96));
            for (i, &sls) in search_list_sizes.iter().enumerate() {
                println!(
                    "L={:<4} {:.3}/{:<12.0}  {:.3}/{:<12.0}  {:.3}/{:<12.0}  {:.3}/{:<12.0}",
                    sls,
                    full_data[i].0,           full_data[i].1,
                    no_pf_data[i].0,          no_pf_data[i].1,
                    no_rerank_data[i].0,      no_rerank_data[i].1,
                    admission_only_data[i].0, admission_only_data[i].1,
                );
            }

            // ── Save JSON for the matplotlib renderer. ──
            let to_json = |data: &[(f64, f64)]| -> Vec<(f64, f64)> {
                data.iter().map(|&(r, q)| (r, q)).collect::<Vec<_>>()
            };
            let json = serde_json::json!({
                "dataset": dim_name,
                "dimension": $N,
                "num_points": num_points,
                "threads": num_threads,
                "search_list_sizes": search_list_sizes,
                "k": k,
                "sweep": run,
                "window_size": ocfg.window_size,
                "default_cascade": {
                    "prefilter":  format!("{:?}", ocfg.prefilter),
                    "admission":  format!("{:?}", ocfg.admission),
                    "rerank":     format!("{:?}", ocfg.rerank),
                },
                "full":            to_json(&full_data),
                "no_prefilter":    to_json(&no_pf_data),
                "no_rerank":       to_json(&no_rerank_data),
                "admission_only":  to_json(&admission_only_data),
            });
            let path = utils::result_path("cascade_ablation", dim_name, k);
            std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
                .expect("write json");
            println!("\nSaved {path}");
        }};
    }

    match dimension {
        DIM_32 => run_cascade_ablation_inner!(32),
        DIM_100 => run_cascade_ablation_inner!(100),
        DIM_128 => run_cascade_ablation_inner!(128),
        DIM_768 => run_cascade_ablation_inner!(768),
        DIM_960 => run_cascade_ablation_inner!(960),
        DIM_1536 => run_cascade_ablation_inner!(1536),
        _ => panic!("Unsupported dimension: {dimension}"),
    }
}
