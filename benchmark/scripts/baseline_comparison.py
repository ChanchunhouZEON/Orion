#!/usr/bin/env python3
"""
Baseline comparison & ablation study.

Compares StagedDiskANN against:
  - DiskANN (Vamana) — our Rust implementation
  - HNSW (hnswlib) — official C++ implementation via Python bindings

Ablation variants of StagedDiskANN:
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
import struct
import subprocess
import time
import numpy as np

def read_fvecs(path, max_n=0):
    """Read .fvecs file, return (data, num_points, dim)."""
    with open(path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0, 2)
        file_size = f.tell()
        record_bytes = 4 + dim * 4
        total = file_size // record_bytes
        n = min(total, max_n) if max_n > 0 else total
        f.seek(0)
        data = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            d = struct.unpack('i', f.read(4))[0]
            assert d == dim
            data[i] = np.array(struct.unpack(f'{dim}f', f.read(dim * 4)))
    return data, n, dim

def read_ivecs(path, max_n=0):
    """Read .ivecs file."""
    with open(path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0, 2)
        file_size = f.tell()
        record_bytes = 4 + dim * 4
        total = file_size // record_bytes
        n = min(total, max_n) if max_n > 0 else total
        f.seek(0)
        data = np.zeros((n, dim), dtype=np.int32)
        for i in range(n):
            d = struct.unpack('i', f.read(4))[0]
            data[i] = np.array(struct.unpack(f'{dim}i', f.read(dim * 4)))
    return data

def recall_at_k(results, ground_truth, k):
    """Compute mean recall@k."""
    n = min(len(results), len(ground_truth))
    total = 0.0
    for i in range(n):
        gt_set = set(ground_truth[i][:k])
        hits = sum(1 for r in results[i][:k] if r in gt_set)
        total += hits / k
    return total / n

def run_hnswlib(base, queries, k, search_list_sizes, num_threads=8, trials=5):
    """Run hnswlib benchmark, return list of (recall, qps) at each ef."""
    import hnswlib
    dim = base.shape[1]
    n = base.shape[0]

    # Build index
    print(f"  Building HNSW (M=16, ef_construction=200, {n} pts)...")
    t0 = time.time()
    index = hnswlib.Index(space='l2', dim=dim)
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

def run_rust_sweep(base_path, query_path, gt_path, max_points, algorithm="qps-recall-sweep"):
    """Run Rust benchmark sweep, parse JSON output."""
    cmd = [
        "./target/release/benchmark",
        "--base", base_path,
        "--query", query_path,
        "--groundtruth", gt_path,
        "--algorithms", algorithm,
        "--max-points", str(max_points),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Rust benchmark failed: {result.returncode}")
    return result.stdout

DATASET_PATHS = {
    "sift": {
        "base": "data/sift/sift_base.fvecs",
        "query": "data/sift/sift_query.fvecs",
        "gt": "data/sift/sift_groundtruth.ivecs",
    },
    "glove25": {
        "base": "data/glove25/glove-25-angular_base.fvecs",
        "query": "data/glove25/glove-25-angular_query.fvecs",
        "gt": "data/glove25/glove-25-angular_groundtruth.ivecs",
    },
    "glove100": {
        "base": "data/glove100/glove-100-angular_base.fvecs",
        "query": "data/glove100/glove-100-angular_query.fvecs",
        "gt": "data/glove100/glove-100-angular_groundtruth.ivecs",
    },
    "gist": {
        "base": "data/gist/gist_base.fvecs",
        "query": "data/gist/gist_query.fvecs",
        "gt": "data/gist/gist_groundtruth.ivecs",
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sift", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    paths = DATASET_PATHS[args.dataset]
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
    if n < gt.shape[0] or np.max(gt) >= n:
        print(f"Recomputing ground truth for {n} points...")
        from scipy.spatial.distance import cdist
        dists = cdist(queries, base, metric='sqeuclidean')
        gt = np.argsort(dists, axis=1)[:, :100].astype(np.int32)
    print(f"  {n} base, {nq} queries, dim={dim}\n")

    all_results = {}

    # ── 1. HNSW (hnswlib official) ──
    print("── HNSW (hnswlib) ──")
    hnsw_results, hnsw_build = run_hnswlib(
        base, queries, args.k, search_list_sizes, args.threads, args.trials
    )
    hnsw_data = []
    for (labels, qps), ef in zip(hnsw_results, search_list_sizes):
        r = recall_at_k(labels.tolist(), gt.tolist(), args.k)
        hnsw_data.append([round(r, 4), round(qps)])
        print(f"  ef={ef:>4}  R@{args.k}={r:.4f}  QPS={qps:.0f}")
    all_results["hnsw"] = hnsw_data

    # ── 2. DiskANN + StagedDiskANN (Rust sweep) ──
    print("\n── DiskANN + StagedDiskANN (Rust) ──")
    run_rust_sweep(paths["base"], paths["query"], paths["gt"], args.max_points)
    # Read the generated JSON
    json_path = f"visualizations/qps_recall_{args.dataset}.json"
    with open(json_path) as f:
        rust_data = json.load(f)
    all_results["diskann"] = rust_data["diskann"]
    all_results["staged"] = rust_data["staged"]

    for name in ["diskann", "staged"]:
        print(f"\n  {name}:")
        for point in all_results[name]:
            r, qps = point
            print(f"    R@{args.k}={r:.4f}  QPS={qps:.0f}")

    # ── Save combined results ──
    output = {
        "dataset": args.dataset,
        "dimension": dim,
        "num_points": n,
        "threads": args.threads,
        "k": args.k,
        "hnsw": all_results["hnsw"],
        "diskann": all_results["diskann"],
        "staged": all_results["staged"],
        "build_time": {
            "hnsw": round(hnsw_build, 3),
        },
    }
    out_path = f"visualizations/baseline_{args.dataset}.json"
    os.makedirs("visualizations", exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
