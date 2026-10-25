/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! DiskANN-only L-sweep — produces a QPS-vs-recall curve in the same
//! shape as `orion`'s output so `collect_sweep_medians.py`
//! can fold it into the three-way comparison JSON alongside
//! Orion and ParlayANN.
//!
//! ## CLI
//!
//! ```text
//! diskann_sweep <dataset> [--metric l2|cosine] [--max-points N]
//!                         [--search-list-sizes L1,L2,...]
//!                         [--k K] [--trials N] [--threads N]
//!                         [--base FILE] [--query FILE] [--ground-truth FILE]
//!                         [--cache-dir DIR] [--prepare-only]
//!
//! Large SIFT presets accept bvecs/u8bin inputs and stream-convert to f32.
//! Temporary fbin files live in --cache-dir (allow 512 GB for SIFT1B),
//! alongside the saved index; they are removed after the build or on unwind.
//! ```
//!
//! This baseline builds an independent Vamana graph using the recorded
//! dataset recipe; matching build parameters do not imply identical topology.
//! The implemented distance backend uses L2, with normalized inputs for
//! angular workloads, including the normalized MS-MARCO dataset.
//!
//! ## Output format
//!
//! Mirrors `orion`'s headline-line shape so
//! `collect_sweep_medians.py` consumes it with the same regex.

#[path = "../utils.rs"]
mod utils;

use diskann::index::{create_inmem_index, ANNInmemIndex};
use diskann::model::{IndexConfiguration, IndexWriteParametersBuilder};
#[path = "diskann_sweep_support/input.rs"]
mod input;
use orion_cli::cli::data::{inspect_vector_file, VectorFormat};
use std::path::Path;
use std::time::Instant;
use vector::Metric;

const DEFAULT_K: usize = 10;
const NUM_THREADS: usize = 8;
const DEFAULT_L_SCHEDULE: &[usize] = &[
    16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 90, 100, 114, 128, 144,
    160, 180, 200, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024,
];

fn mean_recall(results: &[Vec<u32>], gt: &[Vec<u32>], k: usize) -> f64 {
    use std::collections::HashSet;
    let mut sum = 0.0f64;
    let n = results.len();
    for i in 0..n {
        let truth: HashSet<u32> = gt[i].iter().take(k).copied().collect();
        let hits = results[i].iter().filter(|x| truth.contains(x)).count();
        sum += hits as f64 / k as f64;
    }
    sum / n as f64
}

fn metric_default_for_dataset(_dataset: &str) -> Metric {
    // The `diskann` core crate's f32 distance kernel only implements
    // L2 (Cosine panics at runtime — see `vector/src/distance.rs:57`).
    // For datasets whose intrinsic ranking is MIPS / cosine
    // (glove*, msmarco_bert_1M, wiki_ada_1M) we instead L2-normalise
    // the input data before handing it to DiskANN — L2 on unit-norm
    // vectors gives cosine ranking. See `should_normalize_for_diskann`.
    Metric::L2
}

/// Whether to L2-normalise `data` + `queries` before handing them to
/// DiskANN. The intended ranking on these datasets is cosine / MIPS;
/// DiskANN only ships an L2 kernel, but L2 on unit-norm vectors gives
/// cosine ranking — so we normalise the input and let DiskANN use L2.
/// The MS-MARCO experiment uses normalized inputs; keep that ranking unchanged.
fn should_normalize_for_diskann(dataset: &str) -> bool {
    matches!(
        dataset,
        "glove25" | "glove100" | "msmarco_bert_1M" | "wiki_ada_1M"
    )
}

fn l2_normalize_in_place(data: &mut [f32], dim: usize) {
    use rayon::prelude::*;
    data.par_chunks_mut(dim).for_each(|chunk| {
        let mut s = 0.0f64;
        for &v in chunk.iter() {
            s += (v as f64) * (v as f64);
        }
        if s > 0.0 {
            let inv = 1.0f32 / (s.sqrt() as f32);
            for v in chunk.iter_mut() {
                *v *= inv;
            }
        }
    });
}

fn validate_search(k: usize, trials: usize, sizes: &[usize]) {
    assert!(k > 0 && trials > 0, "k and trials must be positive");
    assert!(!sizes.is_empty(), "empty search schedule");
    assert!(
        sizes.iter().all(|&l| l >= k && l <= u32::MAX as usize),
        "require k <= L <= u32::MAX"
    );
    assert!(
        sizes.windows(2).all(|w| w[0] < w[1]),
        "L must be strictly increasing"
    );
}

#[derive(Default)]
struct InputOptions {
    base: Option<String>,
    query: Option<String>,
    ground_truth: Option<String>,
    cache_dir: Option<String>,
    threads: Option<usize>,
    prepare_only: bool,
}

fn parse_args() -> (
    String,
    usize,
    Option<Metric>,
    Vec<usize>,
    usize,
    usize,
    InputOptions,
) {
    let args: Vec<String> = std::env::args().collect();
    let mut dataset = "sift".to_string();
    let mut max_points: usize = usize::MAX;
    let mut metric_override: Option<Metric> = None;
    let mut search_list_sizes: Option<Vec<usize>> = None;
    let mut k = DEFAULT_K;
    let mut trials = 1;
    let mut input = InputOptions::default();
    let mut i = 1;
    let mut seen_positional = false;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--prepare-only" => input.prepare_only = true,
            "--base" | "--query" | "--ground-truth" | "--cache-dir" | "--threads" => {
                i += 1;
                let value = args
                    .get(i)
                    .unwrap_or_else(|| panic!("{a} needs a value"))
                    .clone();
                match a.as_str() {
                    "--base" => input.base = Some(value),
                    "--query" => input.query = Some(value),
                    "--ground-truth" => input.ground_truth = Some(value),
                    "--cache-dir" => input.cache_dir = Some(value),
                    _ => {
                        let threads: usize = value.parse().expect("invalid thread count");
                        assert!(threads > 0 && threads <= u32::MAX as usize);
                        input.threads = Some(threads);
                    }
                }
            }
            "--k" => {
                i += 1;
                k = args
                    .get(i)
                    .expect("--k needs a value")
                    .parse()
                    .expect("invalid k");
            }
            "--trials" => {
                i += 1;
                trials = args
                    .get(i)
                    .expect("--trials needs a value")
                    .parse()
                    .expect("invalid trials");
            }
            "--metric" => {
                i += 1;
                let v = args.get(i).expect("--metric needs a value").to_lowercase();
                metric_override = Some(match v.as_str() {
                    "l2" => Metric::L2,
                    "cosine" | "cos" => Metric::Cosine,
                    other => panic!("unknown metric '{other}' (expected l2|cosine)"),
                });
            }
            "--max-points" => {
                i += 1;
                max_points = args[i].parse().expect("--max-points needs a number");
            }
            "--search-list-sizes" => {
                i += 1;
                let raw = args.get(i).expect("--search-list-sizes needs a value");
                let parsed: Vec<usize> = raw
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse()
                            .expect("L value must be a positive integer")
                    })
                    .collect();
                search_list_sizes = Some(parsed);
            }
            _ if !seen_positional => {
                dataset = a.clone();
                seen_positional = true;
            }
            _ => {
                eprintln!("Unknown arg: {a}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let sizes = search_list_sizes.unwrap_or_else(|| {
        let mut sizes: Vec<usize> = DEFAULT_L_SCHEDULE
            .iter()
            .copied()
            .filter(|&l| l >= k)
            .collect();
        if !sizes.contains(&k) && k > DEFAULT_K {
            sizes.insert(0, k);
        }
        sizes
    });
    validate_search(k, trials, &sizes);
    (
        dataset,
        max_points,
        metric_override,
        sizes,
        k,
        trials,
        input,
    )
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    let (dataset, max_points, metric_override, search_list_sizes, k, trials, options) =
        parse_args();

    // (cache_key, base_path, query_path, gt_path, dim, alpha, R, L_build)
    let (cache_key, base_path, query_path, gt_path, _dim, alpha, r, build_l) =
        match dataset.as_str() {
            "sift10m" | "sift100m" | "sift1b" => (
                dataset.as_str(),
                "data/sift1b/bigann_base.bvecs",
                "data/sift1b/bigann_query.bvecs",
                match dataset.as_str() {
                    "sift10m" => "data/sift1b/idx_10M.ivecs",
                    "sift100m" => "data/sift1b/idx_100M.ivecs",
                    _ => "data/sift1b/idx_1B.ivecs",
                },
                128usize,
                1.15f32,
                64u32,
                128usize,
            ),
            "sift" => (
                "sift",
                "data/sift/sift_base.fvecs",
                "data/sift/sift_query.fvecs",
                "data/sift/sift_groundtruth.ivecs",
                128usize,
                1.15f32,
                64u32,
                128usize,
            ),
            "glove25" => (
                "glove25",
                "data/glove25_norm/glove-25-angular_base.fvecs",
                "data/glove25_norm/glove-25-angular_query.fvecs",
                "data/glove25_norm/glove-25-angular_groundtruth.ivecs",
                32usize,
                1.0f32,
                100u32,
                200usize,
            ),
            "glove100" => (
                "glove100",
                "data/glove100_norm/glove-100-angular_base.fvecs",
                "data/glove100_norm/glove-100-angular_query.fvecs",
                "data/glove100_norm/glove-100-angular_groundtruth.ivecs",
                100usize,
                1.0f32,
                100u32,
                200usize,
            ),
            "gist" => (
                "gist",
                "data/gist/gist_base.fvecs",
                "data/gist/gist_query.fvecs",
                "data/gist/gist_groundtruth.ivecs",
                960usize,
                1.1f32,
                100u32,
                200usize,
            ),
            "deep10m" => (
                "deep10m",
                "data/deep10m/deep10m_base.fvecs",
                "data/deep10m/deep10m_query.fvecs",
                "data/deep10m/deep10m_groundtruth.ivecs",
                128usize,
                1.05f32,
                64u32,
                128usize,
            ),
            "fashion-mnist" => (
                "fashion-mnist",
                "data/fashion-mnist/fashion-mnist-784-euclidean_base.fvecs",
                "data/fashion-mnist/fashion-mnist-784-euclidean_query.fvecs",
                "data/fashion-mnist/fashion-mnist-784-euclidean_groundtruth.ivecs",
                784usize,
                1.1f32,
                40u32,
                80usize,
            ),
            "msmarco_bert_1M" => (
                "msmarco_bert_1M",
                "data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs",
                "data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs",
                "data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs",
                768usize,
                1.0f32,
                64u32,
                128usize,
            ),
            "wiki_ada_1M" => (
                "wiki_ada_1M",
                "data/wiki_ada_1M/wiki_ada_1M_base.fvecs",
                "data/wiki_ada_1M/wiki_ada_1M_query.fvecs",
                "data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs",
                1536usize,
                1.05f32,
                100u32,
                200usize,
            ),
            _ => panic!("Unknown dataset: {dataset}"),
        };

    let metric = metric_override.unwrap_or_else(|| metric_default_for_dataset(&dataset));
    assert!(
        matches!(metric, Metric::L2),
        "DiskANN f32 supports L2 only; normalized presets use L2"
    );
    let metric_tag = match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cos",
    };
    println!(
        "Loading {dataset} (metric={metric_tag}, max_points={})...",
        max_points
    );

    let base_path = Path::new(options.base.as_deref().unwrap_or(base_path));
    let query_path = Path::new(options.query.as_deref().unwrap_or(query_path));
    let gt_path = Path::new(options.ground_truth.as_deref().unwrap_or(gt_path));
    let format = VectorFormat::infer(base_path).expect("base format");
    let header = inspect_vector_file(base_path, format).expect("base metadata");
    let preset_limit = match dataset.as_str() {
        "sift10m" => 10_000_000,
        "sift100m" => 100_000_000,
        "sift1b" => 1_000_000_000,
        _ => usize::MAX,
    };
    let n = header.count.min(max_points).min(preset_limit);
    let dim = header.dimension;
    assert!(
        n > 0 && n <= i32::MAX as usize,
        "DiskANN requires 0 < N <= i32::MAX"
    );
    let normalize = should_normalize_for_diskann(&dataset);
    let threads = options.threads.unwrap_or(NUM_THREADS);

    // Only the small query set is resident in the driver. Base coordinates are
    // streamed into a build file, then loaded once into the index's allocation.
    let (queries, gt) = if options.prepare_only {
        (Vec::new(), Vec::new())
    } else {
        input::load_queries(query_path, gt_path, dim, n, k, normalize)
            .expect("query/ground-truth input")
    };
    let nq = queries.len();
    println!("Base: {n} × {dim} f32; {nq} queries; {threads} threads");

    // Cache stub mirrors the orion paths so disk usage is comparable.
    let alpha_tag = format!("{:.2}", alpha).replace('.', "_");
    let cache_dir =
        std::path::PathBuf::from(options.cache_dir.as_deref().unwrap_or("cache/diskann"));
    std::fs::create_dir_all(&cache_dir).expect("create cache directory");
    let cache_path = cache_dir.join(format!(
        "{cache_key}_n{n}_r{r}_l{build_l}_a{alpha_tag}_{metric_tag}.bin"
    ));
    let cache_data_path = cache_path.with_extension("bin.data");

    // ── Build or load DiskANN ────────────────────────────────────────
    let num_threads = threads as u32;
    let write_params = IndexWriteParametersBuilder::new(build_l as u32, r)
        .with_alpha(alpha)
        .with_num_threads(num_threads)
        .build();
    let config =
        IndexConfiguration::new(metric, dim, dim, n, false, 0, false, 0, 1.0, write_params);

    // A manifest commits the cache only after both index files are saved.
    // Old/unidentified caches are rebuilt rather than silently reused.
    let manifest_path = cache_path.with_extension("manifest.json");
    let identity =
        input::cache_identity(base_path, n, dim, normalize, r, build_l, alpha, metric_tag)
            .expect("base identity");
    let cache_valid = cache_path.exists()
        && cache_data_path.exists()
        && std::fs::read(&manifest_path).ok().as_deref() == Some(identity.as_slice());
    let build_input = if cache_valid {
        None
    } else {
        // Convert before allocating the graph: conversion buffers never overlap
        // with construction scratch, and a failed conversion cannot consume TBs.
        Some(
            input::BuildInput::prepare(base_path, format, n, dim, normalize, &cache_dir)
                .expect("stream base conversion"),
        )
    };

    let mut index: Box<dyn ANNInmemIndex<f32>> =
        create_inmem_index(config).expect("create_inmem_index failed");

    println!("DiskANN (R={r}, L_build={build_l}, α={alpha:.2}, metric={metric_tag})...",);
    let t_build = Instant::now();
    if cache_valid {
        index
            .load(cache_path.to_str().unwrap(), n)
            .expect("DiskANN cache load failed");
        println!(
            "DiskANN loaded from cache in {:.2}s",
            t_build.elapsed().as_secs_f32()
        );
    } else {
        if manifest_path.exists() {
            std::fs::remove_file(&manifest_path).expect("invalidate old cache manifest");
        }
        let build_input = build_input.expect("build input");
        index
            .build(build_input.path().to_str().unwrap(), n)
            .expect("DiskANN build failed");
        // Build has loaded the vectors into the index. Release the owned
        // conversion before save writes another full vector copy to disk.
        // Borrowed source fbin files are preserved by BuildInput::drop.
        drop(build_input);
        index
            .save(cache_path.to_str().unwrap())
            .expect("DiskANN cache save failed");
        std::fs::write(&manifest_path, &identity).expect("commit cache manifest");
        println!("DiskANN built in {:.2}s", t_build.elapsed().as_secs_f32());
    }

    if options.prepare_only {
        return;
    }
    utils::set_thread_qos_user_interactive();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(|_| utils::set_thread_qos_user_interactive())
        .build()
        .expect("rayon pool build failed");

    // Untimed warmup at the largest L to prime caches.
    let max_l = *search_list_sizes.iter().max().unwrap();
    {
        use rayon::prelude::*;
        let _ = pool.install(|| {
            queries
                .par_iter()
                .map(|q| {
                    let mut ids = vec![0u32; k];
                    index
                        .search(q, k, max_l as u32, &mut ids)
                        .expect("warmup search failed");
                    ids
                })
                .collect::<Vec<_>>()
        });
    }

    println!("═══ DiskANN sweep (R={r}, α={alpha:.2}, metric={metric_tag}) ═══",);

    for &l in &search_list_sizes {
        let mut samples = Vec::with_capacity(trials);
        let mut recall = 0.0f64;
        for _ in 0..trials {
            utils::flush_cache();
            let t = Instant::now();
            let results: Vec<Vec<u32>> = {
                use rayon::prelude::*;
                pool.install(|| {
                    queries
                        .par_iter()
                        .map(|q| {
                            let mut ids = vec![0u32; k];
                            index
                                .search(q, k, l as u32, &mut ids)
                                .expect("search failed");
                            ids
                        })
                        .collect()
                })
            };
            let wall = t.elapsed();
            let qps = nq as f64 / wall.as_secs_f64();
            recall = mean_recall(&results, &gt, k);
            samples.push(qps);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let qps_med = samples[samples.len() / 2];
        println!("  L={:>4}  R@{}={:.6}  QPS={:.0}", l, k, recall, qps_med);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recall_uses_requested_k() {
        assert_eq!(mean_recall(&[vec![1, 9]], &[vec![1, 2, 3]], 2), 0.5);
        let ids: Vec<u32> = (0..100).collect();
        assert_eq!(mean_recall(&[ids.clone()], &[ids], 100), 1.0);
    }
    #[test]
    fn accepts_larger_k() {
        validate_search(100, 5, &[100, 128, 2048]);
    }
    #[test]
    #[should_panic]
    fn rejects_short_beam() {
        validate_search(100, 5, &[64, 128]);
    }
}
