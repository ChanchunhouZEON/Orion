#!/usr/bin/env python3
"""
Baseline comparison & ablation study.

Compares Orion against:
  - DiskANN (Vamana) — our Rust implementation
  - HNSW (hnswlib) — official C++ implementation via Python bindings

Ablation variants of Orion:
  - full: all optimizations enabled (auto-calibrated)
  - no-convergence: epsilon=0 (never switch to reranking)
  - no-early-exit: ee=0 (no early termination)
  - no-extra: max_extra=0 (no pruned candidate reuse)
  - no-abandon: use full distance (no early abandon)

Usage:
  python benchmark/scripts/baseline_comparison.py [--dataset sift] [--max-points 10000]
"""

import argparse
import json
import os
import subprocess
import time
import numpy as np

from benchmark_support import read_fvecs, read_ivecs, recall_at_k, ROOT
from benchmark_support import BASELINE_DATASETS, dataset_paths, evaluation_ground_truth, binary_path


def run_hnswlib(base, queries, k, search_list_sizes, num_threads=8, trials=5, space="l2"):
    """Run hnswlib benchmark, return list of (recall, qps) at each ef."""
    import hnswlib
    dim = base.shape[1]
    n = base.shape[0]

    # Build index — `space` is the hnswlib distance: 'l2' / 'ip' /
    # 'cosine'. Per-dataset choice from `dataset_paths(ds)["space"]`.
    print(f"  Building HNSW (M=16, ef_construction=200, {n} pts, space={space})...")
    t0 = time.time()
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.set_num_threads(num_threads)
    index.add_items(base, np.arange(n))
    build_time = time.time() - t0
    print(f"  HNSW built in {build_time:.2f}s")

    results = []
    for ef in search_list_sizes:
        index.set_ef(ef)
        qps_samples = []
        labels = None
        for _ in range(trials):
            t = time.time()
            labels, distances = index.knn_query(queries, k=k, num_threads=num_threads)
            wall = time.time() - t
            qps_samples.append(len(queries) / wall)
        qps = sorted(qps_samples)[trials // 2]  # median
        results.append((labels, qps))

    return results, build_time

def run_rust_sweep(dataset, trials):
    """Refresh the same three-engine artifact consumed by --no-rust.

    The former qps-recall-sweep algorithm no longer exists in benchmark.
    Route through the maintained wrapper instead of reading stale legacy JSON.
    """
    env = dict(os.environ, DATASET=dataset, NUM_RUNS=str(trials), MAX_POINTS="0")
    subprocess.run(["bash", str(ROOT / "benchmark/scripts/sweep_orion_vs_diskann_vs_parlayann.sh")],
                   cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sift", choices=BASELINE_DATASETS)
    parser.add_argument("--max-points", type=int, default=0,
                        help="0 = use the full dataset (no GT recomputation).")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--no-rust", action="store_true",
        help=("Skip refreshing the three-engine comparison and read the "
              "DiskANN / Orion series from "
              "`visualizations/sweep_orion_vs_parlayann_<ds>.json` "
              "(produced by `sweep_orion_vs_diskann_vs_parlayann.sh`). "
              "Use this when the head-to-head sweep is already current."),
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    if min(args.k, args.threads, args.trials) <= 0 or args.max_points < 0:
        parser.error("k, threads and trials must be positive; max-points must be nonnegative")

    if args.k != 10 or args.threads != 8 or args.max_points != 0:
        parser.error("The three-engine panel requires full presets, k=10 and threads=8; "
                     "use dbms_baselines.py for independent subset/k/thread experiments")

    # Cap every BLAS/OpenMP runtime to args.threads before lazy imports.
    # hnswlib already obeys set_num_threads, but the loaded numpy /
    # scipy stack may otherwise grab all cores during recall calc.
    t = str(args.threads)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = t

    paths = dataset_paths(args.dataset)
    search_list_sizes = [16, 20, 24, 32, 40, 48, 56, 64, 80, 100, 128, 160, 200, 256]

    print(f"\n{'='*60}")
    print(f"  Baseline Comparison: {args.dataset} ({args.max_points} pts)")
    print(f"{'='*60}\n")

    # ── Load data for HNSW ──
    print("Loading dataset...")
    base, n, dim = read_fvecs(paths["base"], args.max_points)
    queries, nq, _ = read_fvecs(paths["query"])
    gt = read_ivecs(paths["gt"])
    # Recompute GT if we subsampled
    if not nq or args.k > n or queries.shape[1] != dim:
        parser.error("Empty queries, k exceeds base count, or mismatched dimensions")
    gt = evaluation_ground_truth(base, queries, gt, args.k, paths["metric"])
    print(f"  {n} base, {nq} queries, dim={dim}\n")

    search_list_sizes = sorted(set([args.k] + [ef for ef in search_list_sizes if ef >= args.k]))
    all_results = {}

    # ── 1. HNSW (hnswlib official) ──
    print("── HNSW (hnswlib) ──")
    hnsw_results, hnsw_build = run_hnswlib(
        base, queries, args.k, search_list_sizes, args.threads, args.trials,
        space=paths.get("space", "l2"),
    )
    hnsw_data = []
    for (labels, qps), ef in zip(hnsw_results, search_list_sizes):
        r = recall_at_k(labels.tolist(), gt.tolist(), args.k)
        hnsw_data.append([round(r, 4), round(qps)])
        print(f"  ef={ef:>4}  R@{args.k}={r:.4f}  QPS={qps:.0f}")
    all_results["hnsw"] = hnsw_data

    # ── 2. Read the maintained three-engine comparison artifact ──
    if not args.no_rust:
        run_rust_sweep(args.dataset, args.trials)
    sv_path = f"visualizations/sweep_orion_vs_parlayann_{args.dataset}.json"
    with open(sv_path) as stream:
        comparison = json.load(stream)
    for field, expected in (("dataset", args.dataset), ("num_points", n),
                            ("k", args.k), ("threads", args.threads)):
        # Older comparison artifacts predate explicit k/threads metadata.
        actual = comparison.get(field, {"k": 10, "threads": 8}.get(field))
        if actual != expected:
            parser.error(f"{sv_path}: incompatible {field}={actual}, expected {expected}")
    for engine in ("diskann", "orion", "parlayann"):
        if engine in comparison:
            all_results[engine] = comparison[engine]

    for name in ["diskann", "orion"]:
        print(f"\n  {name}:")
        for point in all_results[name]:
            r, qps = point
            print(f"    R@{args.k}={r:.4f}  QPS={qps:.0f}")

    # ── Save combined results ──
    # Merge into the existing baseline JSON if present so that prior
    # phases (e.g. dbms_baselines writing usearch_hnsw / lancedb_hnsw,
    # additional_baselines writing faiss/annoy) are preserved.
    out_path = f"visualizations/baseline_{args.dataset}.json"
    os.makedirs("visualizations", exist_ok=True)
    if os.path.exists(out_path):
        with open(out_path) as f:
            output = json.load(f)
    else:
        output = {}
    output.update({
        "dataset": args.dataset,
        "dimension": dim,
        "num_points": n,
        "threads": args.threads,
        "k": args.k,
        "hnsw": all_results["hnsw"],
        "diskann": all_results["diskann"],
        "orion": all_results["orion"],
    })
    bt = output.get("build_time", {})
    bt["hnsw"] = round(hnsw_build, 3)
    output["build_time"] = bt
    if "parlayann" in all_results:
        output["parlayann"] = all_results["parlayann"]
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
