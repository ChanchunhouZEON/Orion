/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! DiskANN-only L-sweep — produces a QPS-vs-recall curve in the same
//! shape as `staged_diskann`'s output so `collect_sweep_medians.py`
//! can fold it into the three-way comparison JSON alongside
//! StagedDiskANN and ParlayANN.
//!
//! ## CLI
//!
//! ```text
//! diskann_sweep <dataset> [--metric l2|cosine] [--max-points N]
//!                         [--search-list-sizes L1,L2,...]
//! ```
//!
//! Per-dataset build params (`R`, `L_build`, `α`) match
//! `staged_diskann`'s defaults verbatim so the three engines
//! (DiskANN / StagedDiskANN / ParlayANN) all build the same graph
//! topology. Metric defaults: L2 for SIFT/Deep10M/GIST/Fashion-MNIST,
//! Cosine for GloVe/MSMarco-BERT/Wiki-ada (DiskANN's `vector::Metric`
//! ships only L2 + Cosine, so raw-MIPS workloads get the
//! Cosine-on-raw approximation).
//!
//! ## Output format
//!
//! Mirrors `staged_diskann`'s headline-line shape so
//! `collect_sweep_medians.py` consumes it with the same regex.

#[path = "../utils.rs"]
mod utils;

use diskann::index::{ANNInmemIndex, create_inmem_index};
use diskann::model::{IndexConfiguration, IndexWriteParametersBuilder};
use std::io::Write;
use std::time::Instant;
use vector::Metric;

const K: usize = 10;
const TRIALS: usize = 1;
const NUM_THREADS: usize = 8;
const DEFAULT_L_SCHEDULE: &[usize] = &[
    16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 90, 100, 114, 128, 144,
    160, 180, 200, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024,
];

fn load_fvecs(path: &str, max_points: usize) -> (Vec<f32>, usize, usize) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("Cannot open {path}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let dim = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let record_bytes = 4 + dim * 4;
    let total = buf.len() / record_bytes;
    let n = total.min(max_points);
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let base = i * record_bytes + 4;
        for d in 0..dim {
            let off = base + d * 4;
            data.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
    }
    (data, n, dim)
}

fn load_ivecs(path: &str) -> (Vec<Vec<u32>>, usize) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("Cannot open {path}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let k = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let record_bytes = 4 + k * 4;
    let n = buf.len() / record_bytes;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * record_bytes + 4;
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            let off = base + j * 4;
            row.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
        out.push(row);
    }
    (out, n)
}

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
/// (Note: this gives cosine, not raw MIPS — on the raw-MIPS-GT
/// datasets like msmarco_bert_1M the resulting recall will be capped
/// by the cosine-vs-MIPS ranking gap, since GT was computed as raw
/// dot product.)
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

fn parse_args() -> (String, usize, Option<Metric>, Vec<usize>) {
    let args: Vec<String> = std::env::args().collect();
    let mut dataset = "sift".to_string();
    let mut max_points: usize = usize::MAX;
    let mut metric_override: Option<Metric> = None;
    let mut search_list_sizes: Option<Vec<usize>> = None;
    let mut i = 1;
    let mut seen_positional = false;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
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
                    .map(|s| s.trim().parse().expect("L value must be a positive integer"))
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
    (
        dataset,
        max_points,
        metric_override,
        search_list_sizes.unwrap_or_else(|| DEFAULT_L_SCHEDULE.to_vec()),
    )
}

/// Write a flat `[f32]` slab as DiskANN's expected on-disk format:
/// `[i32 num_points][i32 dimension][f32 × num_points × dimension]`.
fn write_diskann_bin(
    data: &[f32],
    num_points: usize,
    dimension: usize,
) -> std::io::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "diskann_sweep_{}_{}.bin",
        std::process::id(),
        num_points
    ));
    let mut file = std::fs::File::create(&path)?;
    file.write_all(&(num_points as i32).to_le_bytes())?;
    file.write_all(&(dimension as i32).to_le_bytes())?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    file.write_all(bytes)?;
    file.flush()?;
    Ok(path)
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    let (dataset, max_points, metric_override, search_list_sizes) = parse_args();

    // (cache_key, base_path, query_path, gt_path, dim, alpha, R, L_build)
    let (cache_key, base_path, query_path, gt_path, _dim, alpha, r, build_l) = match dataset
        .as_str()
    {
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
    let metric_tag = match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cos",
    };
    println!(
        "Loading {dataset} (metric={metric_tag}, max_points={})...",
        max_points
    );

    let (mut data, n, dim) = load_fvecs(base_path, max_points);
    let (mut qdata, nq, _) = load_fvecs(query_path, usize::MAX);
    let (gt, ngt) = load_ivecs(gt_path);
    assert_eq!(nq, ngt, "query count ≠ groundtruth count");

    if should_normalize_for_diskann(&dataset) {
        println!(
            "L2-normalising {n} base + {nq} query vectors (DiskANN approximates cosine ranking via L2 on unit-norm input)"
        );
        l2_normalize_in_place(&mut data, dim);
        l2_normalize_in_place(&mut qdata, dim);
    }

    let queries: Vec<Vec<f32>> = (0..nq).map(|i| qdata[i * dim..(i + 1) * dim].to_vec()).collect();
    println!("Loaded {n} base, {nq} queries.");

    // Cache stub mirrors the staged paths so disk usage is comparable.
    let alpha_tag = format!("{:.2}", alpha).replace('.', "_");
    let cache_dir = std::path::PathBuf::from("cache/diskann");
    std::fs::create_dir_all(&cache_dir).ok();
    let cache_path = cache_dir.join(format!(
        "{cache_key}_n{n}_r{r}_l{build_l}_a{alpha_tag}_{metric_tag}.bin"
    ));
    let cache_data_path = cache_path.with_extension("bin.data");

    // ── Build or load DiskANN ────────────────────────────────────────
    let num_threads = NUM_THREADS as u32;
    let write_params = IndexWriteParametersBuilder::new(build_l as u32, r)
        .with_alpha(alpha)
        .with_num_threads(num_threads)
        .build();
    let config = IndexConfiguration::new(
        metric, dim, dim, n, false, 0, false, 0, 1.0, write_params,
    );

    let mut index: Box<dyn ANNInmemIndex<f32>> =
        create_inmem_index(config).expect("create_inmem_index failed");

    println!(
        "DiskANN (R={r}, L_build={build_l}, α={alpha:.2}, metric={metric_tag})...",
    );
    let t_build = Instant::now();
    if cache_path.exists() && cache_data_path.exists() {
        index
            .load(cache_path.to_str().unwrap(), n)
            .expect("DiskANN cache load failed");
        println!(
            "DiskANN loaded from cache in {:.2}s",
            t_build.elapsed().as_secs_f32()
        );
    } else {
        let temp_data = write_diskann_bin(&data, n, dim).expect("temp data write failed");
        index
            .build(temp_data.to_str().unwrap(), n)
            .expect("DiskANN build failed");
        index
            .save(cache_path.to_str().unwrap())
            .expect("DiskANN cache save failed");
        std::fs::remove_file(&temp_data).ok();
        println!(
            "DiskANN built in {:.2}s",
            t_build.elapsed().as_secs_f32()
        );
    }

    // Pin queries (consistency with staged_sweep's mlock setup).
    utils::set_thread_qos_user_interactive();
    if let Some(first) = queries.first() {
        let qbytes = queries
            .iter()
            .map(|q| q.len() * std::mem::size_of::<f32>())
            .sum::<usize>();
        utils::mlock_bytes("queries", first.as_ptr() as *const u8, qbytes);
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(NUM_THREADS)
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
                    let mut ids = vec![0u32; K];
                    let _ = index.search(q, K, max_l as u32, &mut ids);
                    ids
                })
                .collect::<Vec<_>>()
        });
    }

    println!(
        "═══ DiskANN sweep (R={r}, α={alpha:.2}, metric={metric_tag}) ═══",
    );

    for &l in &search_list_sizes {
        let mut samples = Vec::with_capacity(TRIALS);
        let mut recall = 0.0f64;
        for _ in 0..TRIALS {
            utils::flush_cache();
            let t = Instant::now();
            let results: Vec<Vec<u32>> = {
                use rayon::prelude::*;
                pool.install(|| {
                    queries
                        .par_iter()
                        .map(|q| {
                            let mut ids = vec![0u32; K];
                            let _ = index.search(q, K, l as u32, &mut ids);
                            ids
                        })
                        .collect()
                })
            };
            let wall = t.elapsed();
            let qps = nq as f64 / wall.as_secs_f64();
            recall = mean_recall(&results, &gt, K);
            samples.push(qps);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let qps_med = samples[samples.len() / 2];
        println!("  L={:>4}  R@10={:.4}  QPS={:.0}", l, recall, qps_med);
    }
}
