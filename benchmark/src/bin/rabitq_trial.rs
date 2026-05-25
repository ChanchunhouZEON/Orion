/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Side-by-side trial of RaBitQ 1-bit quantization vs the production
//! L2-Q (u8) path on the same StagedDiskANN graph. Reports per-L QPS +
//! recall for both methods, plus the one-time sidecar build/load cost.
//!
//! Reuses the existing `cache/staged/<dataset>_…<R>_l<L>_a<alpha>_ex<ex>.bin`
//! cache produced by `staged_sweep` — graph must already exist (run
//! `staged_sweep <dataset>` first if missing).
//!
//! Usage:
//!   ./target/release/rabitq_trial sift
//!   ./target/release/rabitq_trial sift --ls 32,48,64,128
//!   ./target/release/rabitq_trial gist --ls 48,96,192
//!
//! What it prints per `L`:
//!   - Baseline L2-Q QPS + Recall@10
//!   - RaBitQ        QPS + Recall@10
//!   - Speedup ratio (RaBitQ / baseline; <1 means RaBitQ is slower —
//!     expected until the NEON kernel lands)
//!
//! Setup mirrors `staged_sweep`: mlock hot regions, USER_INTERACTIVE
//! QoS on workers, `flush_cache()` between trials.

use std::collections::HashSet;
use std::time::Instant;

use rayon::prelude::*;
use staged_diskann::StagedDiskANN;

#[path = "../utils.rs"]
mod utils;

const K: usize = 10;
const CALIB_SAMPLE: usize = 200;
const CALIB_L: usize = 48;
const NUM_THREADS: usize = 8;
const DEFAULT_LS: &[usize] = &[32, 48, 64, 96, 128, 192, 256];
const DEFAULT_MAX_EXTRA: usize = 16;
const DEFAULT_WS: usize = 5;
const TRIALS: usize = 3;

fn parse_args() -> (String, Vec<usize>, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut dataset = "sift".to_string();
    let mut ls: Option<Vec<usize>> = None;
    let mut n_override: Option<usize> = None;
    let mut i = 1;
    let mut saw_positional = false;
    while i < args.len() {
        match args[i].as_str() {
            "--ls" | "--search-list-sizes" => {
                i += 1;
                let raw = args.get(i).expect("--ls needs a comma list");
                let parsed: Vec<usize> = raw
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().expect("bad L value"))
                    .collect();
                assert!(!parsed.is_empty(), "--ls is empty");
                ls = Some(parsed);
            }
            "--n" | "--max-points" => {
                i += 1;
                n_override = Some(
                    args.get(i)
                        .and_then(|s| s.parse::<usize>().ok())
                        .expect("--n needs a positive integer"),
                );
            }
            other if !saw_positional => {
                dataset = other.to_string();
                saw_positional = true;
            }
            other => panic!("unknown arg: {other}"),
        }
        i += 1;
    }
    // Per-dataset defaults for the truncation: we have cached graphs
    // for SIFT 1M and GIST 100k. Override with --n.
    let n_default = match dataset.as_str() {
        "gist" => 100_000,
        _ => usize::MAX,
    };
    (
        dataset,
        ls.unwrap_or_else(|| DEFAULT_LS.to_vec()),
        n_override.unwrap_or(n_default),
    )
}

fn load_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("cannot open {path}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let dim = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let rec = 4 + dim * 4;
    let n = buf.len() / rec;
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let base = i * rec + 4;
        for d in 0..dim {
            let off = base + d * 4;
            out.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
    }
    (out, n, dim)
}

fn load_ivecs(path: &str) -> Vec<Vec<u32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("cannot open {path}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let k = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let rec = 4 + k * 4;
    let n = buf.len() / rec;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * rec + 4;
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            let off = base + j * 4;
            row.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
        out.push(row);
    }
    out
}

/// Brute-force ground truth on the **loaded subset**. Necessary when
/// `n < n_full` because the on-disk GT file references vertex IDs from
/// the full base set — those IDs are largely outside our truncated
/// slice, so disk-GT-based recall would be meaningless.
///
/// Parallelized over queries via rayon. For GIST 100k × 1000 queries
/// (D=960) this runs in ~3-5 seconds on M2 8 cores.
fn brute_force_gt(
    base: &[f32],
    queries: &[Vec<f32>],
    n: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    queries
        .par_iter()
        .map(|q| {
            // Compute L2² to every base vector; keep a running top-K by
            // distance using a max-heap.
            let mut top: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
            for vid in 0..n {
                let x = &base[vid * dim..(vid + 1) * dim];
                let mut d = 0.0f32;
                for j in 0..dim {
                    let diff = q[j] - x[j];
                    d += diff * diff;
                }
                // Maintain a sorted-descending top-K (small k, linear
                // insertion is fine and avoids BinaryHeap allocation).
                if top.len() < k {
                    top.push((d, vid as u32));
                    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                } else if d < top[0].0 {
                    top[0] = (d, vid as u32);
                    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }
            }
            // Final ascending order so callers can compare against
            // increasing-distance search outputs.
            top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            top.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

fn mean_recall(results: &[Vec<u32>], gt: &[Vec<u32>], k: usize) -> f64 {
    let n = results.len();
    let mut s = 0.0;
    for i in 0..n {
        let truth: HashSet<u32> = gt[i].iter().take(k).copied().collect();
        let hits = results[i].iter().filter(|x| truth.contains(x)).count();
        s += hits as f64 / k as f64;
    }
    s / n as f64
}

/// Per-dataset config. Adding a new entry = one line.
struct DsConfig {
    cache_key: &'static str,
    base: &'static str,
    query: &'static str,
    gt: &'static str,
    dim: usize,
    alpha: f32,
    r: usize,
    build_l: usize,
}

fn ds_config(name: &str) -> DsConfig {
    match name {
        "sift" => DsConfig {
            cache_key: "sift",
            base: "data/sift/sift_base.fvecs",
            query: "data/sift/sift_query.fvecs",
            gt: "data/sift/sift_groundtruth.ivecs",
            dim: 128,
            alpha: 1.15,
            r: 64,
            build_l: 128,
        },
        "gist" => DsConfig {
            cache_key: "gist",
            base: "data/gist/gist_base.fvecs",
            query: "data/gist/gist_query.fvecs",
            gt: "data/gist/gist_groundtruth.ivecs",
            dim: 960,
            alpha: 1.5,
            r: 32,
            build_l: 48,
        },
        _ => panic!("unsupported dataset {name} — use sift or gist"),
    }
}

macro_rules! run_trial {
    ($cfg:ident, $data:ident, $queries:ident, $n:ident, $N:literal, $gt:ident, $ls:ident) => {{
        // Cache path uses the same naming convention as `staged_sweep`
        // so we hit the existing on-disk graph.
        let alpha_tag = format!("{:.2}", $cfg.alpha).replace('.', "_");
        let cache_dir = std::path::PathBuf::from("cache/staged");
        let cache_path = cache_dir.join(format!(
            "{}_n{}_r{}_l{}_a{}_ex{}.bin",
            $cfg.cache_key, $n, $cfg.r, $cfg.build_l, alpha_tag, DEFAULT_MAX_EXTRA
        ));
        let pgraph_path = cache_path.with_extension("pgraph");

        if !cache_path.exists() || !pgraph_path.exists() {
            panic!(
                "graph cache missing: {:?} (.bin) / {:?} (.pgraph) — run `staged_sweep {}` first to populate",
                cache_path, pgraph_path, $cfg.cache_key
            );
        }

        println!("Loading cached PhasedGraph from {:?}", pgraph_path);
        let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
        let mut staged = StagedDiskANN::<$N>::load_from_cache(&cache_path, empty_ds)
            .expect("load_from_cache failed");
        let mut ds = diskann::model::InmemDataset::<f32, $N>::new($n, 1.0).unwrap();
        ds.data.memcpy(&$data[..$n * $N]).unwrap();
        staged.dataset = ds;

        let queries_arr: Vec<[f32; $N]> = $queries
            .iter()
            .map(|q| {
                let mut a = [0f32; $N];
                a.copy_from_slice(&q[..$N]);
                a
            })
            .collect();

        // Ensure both sidecars (u8 baseline + RaBitQ) before the timed
        // region so the per-L numbers don't include first-call build
        // cost. Time both individually for the build-cost report.
        println!("\n── Sidecar setup ──");
        let t_u8 = Instant::now();
        let _ = staged.ensure_quantized_dataset();
        println!(
            "  u8 (L2-Q baseline):   {:>6.2}s",
            t_u8.elapsed().as_secs_f32(),
        );
        let t_rbq = Instant::now();
        let _ = staged.ensure_quantized_dataset_rabitq();
        println!(
            "  RaBitQ 1-bit:         {:>6.2}s   (D={}, codes={} B/vertex)",
            t_rbq.elapsed().as_secs_f32(),
            $N,
            staged_diskann::model::dataset::rabitq_dataset::rabitq_code_stride($N),
        );
        let t_b4 = Instant::now();
        let _ = staged.ensure_quantized_dataset_rabitq_b4();
        println!(
            "  RaBitQ 4-bit:         {:>6.2}s   (D={}, codes={} B/vertex)",
            t_b4.elapsed().as_secs_f32(),
            $N,
            staged_diskann::model::dataset::rabitq_b4_dataset::rabitq_b4_code_stride($N),
        );

        // ── Diagnostic: estimator-vs-truth on real (query, vertex)
        // pairs from the GT file. If the estimator is correctly
        // unbiased, est should track truth closely for near pairs and
        // for far pairs; large systematic offset means there's a bug.
        // High variance with correct mean → noise issue (paper's
        // O(||x||/sqrt(D))). If estimates are clearly biased or
        // randomly distributed regardless of true distance: it's a bug.
        {
            let rbq = staged.ensure_quantized_dataset_rabitq();
            let base = staged.dataset.data.as_slice();

            // Sanity 1: rotation orthogonality at the actual N. Sample
            // row-pairs and check P @ P^T is close to I.
            {
                let rot = rbq.rotation.as_slice();
                let mut max_offdiag = 0.0f32;
                let mut max_diag_dev = 0.0f32;
                // Spot-check 32 row pairs (not all, that's N² = 16k-921k).
                let probes = [
                    (0, 0),
                    (1, 1),
                    ($N / 4, $N / 4),
                    ($N - 1, $N - 1),
                    (0, 1),
                    (0, $N / 2),
                    (1, $N - 1),
                    ($N / 3, $N / 3 + 1),
                ];
                for &(i, j) in probes.iter() {
                    let row_i = &rot[i * $N..(i + 1) * $N];
                    let row_j = &rot[j * $N..(j + 1) * $N];
                    let dot: f32 = row_i.iter().zip(row_j).map(|(a, b)| a * b).sum();
                    if i == j {
                        let dev = (dot - 1.0).abs();
                        if dev > max_diag_dev {
                            max_diag_dev = dev;
                        }
                    } else {
                        let dev = dot.abs();
                        if dev > max_offdiag {
                            max_offdiag = dev;
                        }
                    }
                }
                println!(
                    "\n── Diagnostic ──\n  Rotation orthogonality (sampled): max |diag - 1| = {:.2e}, max |offdiag| = {:.2e}",
                    max_diag_dev, max_offdiag,
                );
            }

            // s_x distribution: should concentrate near asymptotic
            // sqrt(2N/π). High variance → per-vertex correction matters.
            // Near-constant → using s_x vs the constant should be ~no-op.
            {
                let n_samp = rbq.s_values.len().min(10000);
                let mut s_vals: Vec<f32> = rbq.s_values[..n_samp].to_vec();
                s_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p1 = s_vals[s_vals.len() / 100];
                let p50 = s_vals[s_vals.len() / 2];
                let p99 = s_vals[s_vals.len() * 99 / 100];
                let mean: f32 = s_vals.iter().sum::<f32>() / s_vals.len() as f32;
                let var: f32 =
                    s_vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / s_vals.len() as f32;
                let std = var.sqrt();
                println!(
                    "  s_x dist (n={n_samp}): asymptotic = {:.3}, p1={:.3} p50={:.3} p99={:.3} mean={:.3} std={:.3} (cv={:.3})",
                    rbq.scale, p1, p50, p99, mean, std, std / mean.max(1e-6),
                );
            }

            // Sanity 2: estimator vs truth on real queries. For each of
            // 3 sample queries, pick the true top-1 NN, true rank-50,
            // and a random "far" vertex; compute true L2² and the
            // RaBitQ estimate; print side by side.
            println!(
                "  q_idx | vert (rank) |   true L2²  |   est L2²  |  err   | err/||x||"
            );
            let q_indices = [0usize, 100, 999];
            for &qi in q_indices.iter() {
                let q_orig: [f32; $N] = {
                    let mut a = [0f32; $N];
                    a.copy_from_slice(&$queries[qi][..$N]);
                    a
                };
                let q_norm_sq: f32 = q_orig.iter().map(|v| v * v).sum();
                let mut rotated_q = [0f32; $N];
                rbq.rotate_query(&q_orig, &mut rotated_q);

                let probes = [
                    ("top-1   ", $gt[qi][0] as usize),
                    ("rank-50 ", $gt[qi][49.min($gt[qi].len() - 1)] as usize),
                    ("random  ", (qi.wrapping_mul(2654435761) % $n)),
                ];
                for &(label, vid) in probes.iter() {
                    // GT was computed on the full base; if we truncated
                    // (e.g. GIST 100k subset), some GT IDs may be out of
                    // range — skip them rather than crash.
                    if vid >= $n {
                        continue;
                    }
                    let x = &base[vid * $N..(vid + 1) * $N];
                    let true_l2_sq: f32 =
                        q_orig.iter().zip(x.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                    let est = rbq.estimate_l2_sq(&rotated_q, q_norm_sq, vid as u32);
                    let err = est - true_l2_sq;
                    let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let err_rel = err.abs() / x_norm.max(1.0);
                    println!(
                        "  {qi:>5} | {vid:>5} ({label}) | {:>10.1} | {:>10.1} | {:>+7.1} | {:>+7.2}",
                        true_l2_sq, est, err, err_rel
                    );
                }
            }
            println!();
        }

        // Calibrate at CALIB_L=48 with real queries — same recipe as
        // staged_sweep. RaBitQ uses the same calibration since it
        // shares the search loop's convergence + early-exit machinery.
        let calib_qs: Vec<[f32; $N]> = queries_arr[..CALIB_SAMPLE.min(queries_arr.len())].to_vec();
        let calib = staged
            .calibrate(&calib_qs, CALIB_L, DEFAULT_WS)
            .expect("calibrate failed");
        let thr = calib.threshold;
        let ee = calib.early_exit_limit;
        println!("Calibrated: threshold={thr:.2}, early_exit_limit={ee}\n");

        // mlock hot regions (mirrors staged_sweep's setup).
        {
            let ds_ptr = staged.dataset.data.as_slice().as_ptr() as *const u8;
            let ds_len = staged.dataset.data.len() * std::mem::size_of::<f32>();
            utils::mlock_bytes("dataset f32", ds_ptr, ds_len);

            let q_u8 = staged.ensure_quantized_dataset();
            utils::mlock_bytes(
                "qdataset u8",
                q_u8.data.as_slice().as_ptr() as *const u8,
                q_u8.data.len(),
            );

            let q_rbq = staged.ensure_quantized_dataset_rabitq();
            utils::mlock_bytes(
                "qdataset RaBitQ codes",
                q_rbq.codes.as_slice().as_ptr(),
                q_rbq.codes.len(),
            );
            utils::mlock_bytes(
                "qdataset RaBitQ norms",
                q_rbq.norms.as_ptr() as *const u8,
                q_rbq.norms.len() * std::mem::size_of::<f32>(),
            );
            utils::mlock_bytes(
                "qdataset RaBitQ rotation",
                q_rbq.rotation.as_slice().as_ptr() as *const u8,
                q_rbq.rotation.len() * std::mem::size_of::<f32>(),
            );

            let q_b4 = staged.ensure_quantized_dataset_rabitq_b4();
            utils::mlock_bytes(
                "qdataset RaBitQ-B4 codes",
                q_b4.codes.as_slice().as_ptr(),
                q_b4.codes.len(),
            );
            utils::mlock_bytes(
                "qdataset RaBitQ-B4 taus",
                q_b4.taus.as_ptr() as *const u8,
                q_b4.taus.len() * std::mem::size_of::<f32>(),
            );

            let pg = staged.graph.buffer_bytes();
            utils::mlock_bytes("pgraph slab", pg.as_ptr(), pg.len());

            let qbytes_ptr = queries_arr.as_ptr() as *const u8;
            let qbytes_len = queries_arr.len() * std::mem::size_of::<[f32; $N]>();
            utils::mlock_bytes("queries", qbytes_ptr, qbytes_len);
        }

        // Build the rayon pool once; both searches reuse it.
        utils::set_thread_qos_user_interactive();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .start_handler(|_| utils::set_thread_qos_user_interactive())
            .build()
            .unwrap();

        // Warmup at L=1024 (broad beam touches more pages than the L
        // schedule will). Same trick as staged_sweep.
        let warmup_l = 1024.min($ls.iter().copied().max().unwrap_or(64) * 4);
        let _ = pool.install(|| {
            staged
                .search_batch_l2_u8_q(&queries_arr, K, warmup_l, DEFAULT_WS, thr, ee)
                .unwrap()
        });
        let _ = pool.install(|| {
            staged
                .search_batch_rabitq(&queries_arr, K, warmup_l, DEFAULT_WS, thr, ee)
                .unwrap()
        });
        let _ = pool.install(|| {
            staged
                .search_batch_rabitq_b4(&queries_arr, K, warmup_l, DEFAULT_WS, thr, ee)
                .unwrap()
        });

        println!(
            "{:>5}  {:>9} {:>7}  {:>9} {:>7}  {:>9} {:>7}",
            "L", "u8 QPS", "u8 R@10", "rbq QPS", "rbq R@10", "b4 QPS", "b4 R@10",
        );
        println!("  {}", "─".repeat(80));

        for &l in $ls.iter() {
            // Helper: timed sweep over `TRIALS` for a given search fn.
            macro_rules! run_n {
                ($call:expr) => {{
                    let mut samples = Vec::with_capacity(TRIALS);
                    let mut recall = 0.0;
                    for _ in 0..TRIALS {
                        utils::flush_cache();
                        let t = Instant::now();
                        let res = pool.install(|| $call);
                        let wall = t.elapsed();
                        samples.push(queries_arr.len() as f64 / wall.as_secs_f64());
                        recall = mean_recall(&res, &$gt, K);
                    }
                    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    (samples[TRIALS / 2], recall)
                }};
            }

            let (u8_qps, u8_recall) = run_n!(staged
                .search_batch_l2_u8_q(&queries_arr, K, l, DEFAULT_WS, thr, ee)
                .unwrap());
            let (rbq_qps, rbq_recall) = run_n!(staged
                .search_batch_rabitq(&queries_arr, K, l, DEFAULT_WS, thr, ee)
                .unwrap());
            let (b4_qps, b4_recall) = run_n!(staged
                .search_batch_rabitq_b4(&queries_arr, K, l, DEFAULT_WS, thr, ee)
                .unwrap());

            println!(
                "{:>5}  {:>9.0} {:>7.4}  {:>9.0} {:>7.4}  {:>9.0} {:>7.4}",
                l, u8_qps, u8_recall, rbq_qps, rbq_recall, b4_qps, b4_recall,
            );
        }
    }};
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    let (dataset, ls, max_n) = parse_args();
    let cfg = ds_config(&dataset);
    println!("Loading {} (dim={}, n cap={})...", dataset, cfg.dim, max_n);
    let (data, n_full, _) = load_fvecs(cfg.base);
    let n = n_full.min(max_n);
    let (qdata, nq, _) = load_fvecs(cfg.query);
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|i| qdata[i * cfg.dim..(i + 1) * cfg.dim].to_vec())
        .collect();

    // When the base is truncated, the on-disk GT (which references the
    // full base by vertex ID) becomes meaningless — most "true top-K"
    // IDs are outside our slice and unrecallable. Recompute brute-force
    // GT on the actual loaded subset so QPS-vs-recall is comparable
    // across methods.
    let gt: Vec<Vec<u32>> = if n < n_full {
        println!(
            "Computing brute-force GT for the {n}-vertex subset ({} queries × {n} × {} dim)...",
            nq, cfg.dim,
        );
        let t = Instant::now();
        let gt = brute_force_gt(&data, &queries, n, cfg.dim, 100);
        println!(
            "  brute-force GT done in {:.2}s",
            t.elapsed().as_secs_f32()
        );
        gt
    } else {
        load_ivecs(cfg.gt)
    };
    println!(
        "Loaded {} base (capped from {}) + {} queries, gt rows = {}",
        n, n_full, nq, gt.len()
    );

    match cfg.dim {
        128 => run_trial!(cfg, data, queries, n, 128, gt, ls),
        960 => run_trial!(cfg, data, queries, n, 960, gt, ls),
        d => panic!("unsupported dim {d}"),
    }
}
