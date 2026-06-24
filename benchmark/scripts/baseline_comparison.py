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
import struct
import subprocess
import time
import numpy as np

def read_fvecs(path, max_n=0):
    """Read .fvecs as (data, num_points, dim) — vectorised via np.fromfile.
    Format: each record is `[dim:i32][data:f32 × dim]`; read the whole
    file as f32, reshape to (n, 1 + dim), drop the leading dim-as-f32
    placeholder column. Two orders of magnitude faster than a per-row
    `struct.unpack` loop on million-point datasets."""
    with open(path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
    record_floats = 1 + dim
    raw = np.fromfile(path, dtype=np.float32)
    total = raw.size // record_floats
    n = min(total, max_n) if max_n > 0 else total
    data = raw[: n * record_floats].reshape(n, record_floats)[:, 1:].copy()
    return data, n, dim

def read_ivecs(path, max_n=0):
    """Read .ivecs file — same vectorised pattern as `read_fvecs`."""
    with open(path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
    record_ints = 1 + dim
    raw = np.fromfile(path, dtype=np.int32)
    total = raw.size // record_ints
    n = min(total, max_n) if max_n > 0 else total
    return raw[: n * record_ints].reshape(n, record_ints)[:, 1:].copy()

def recall_at_k(results, ground_truth, k):
    """Compute mean recall@k."""
    n = min(len(results), len(ground_truth))
    total = 0.0
    for i in range(n):
        gt_set = set(ground_truth[i][:k])
        hits = sum(1 for r in results[i][:k] if r in gt_set)
        total += hits / k
    return total / n

def run_hnswlib(base, queries, k, search_list_sizes, num_threads=8, trials=5, space="l2"):
    """Run hnswlib benchmark, return list of (recall, qps) at each ef."""
    import hnswlib
    dim = base.shape[1]
    n = base.shape[0]

    # Build index — `space` is the hnswlib distance: 'l2' / 'ip' /
    # 'cosine'. Per-dataset choice from `DATASET_PATHS[<ds>]["space"]`.
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
        "space": "l2",
    },
    "glove25": {
        "base": "data/glove25_norm/glove-25-angular_base.fvecs",
        "query": "data/glove25_norm/glove-25-angular_query.fvecs",
        "gt": "data/glove25_norm/glove-25-angular_groundtruth.ivecs",
        "space": "l2",  # pre-normalised → L2 on unit-norm == cosine ranking
    },
    "glove100": {
        "base": "data/glove100_norm/glove-100-angular_base.fvecs",
        "query": "data/glove100_norm/glove-100-angular_query.fvecs",
        "gt": "data/glove100_norm/glove-100-angular_groundtruth.ivecs",
        "space": "l2",
    },
    "gist": {
        "base": "data/gist/gist_base.fvecs",
        "query": "data/gist/gist_query.fvecs",
        "gt": "data/gist/gist_groundtruth.ivecs",
        "space": "l2",
    },
    "deep10m": {
        "base": "data/deep10m/deep10m_base.fvecs",
        "query": "data/deep10m/deep10m_query.fvecs",
        "gt": "data/deep10m/deep10m_groundtruth.ivecs",
        "space": "l2",
    },
    "msmarco_bert_1M": {
        "base": "data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs",
        "query": "data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs",
        "gt": "data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs",
        "space": "ip",   # raw dot product on non-unit-norm BERT vectors
    },
    "wiki_ada_1M": {
        "base": "data/wiki_ada_1M/wiki_ada_1M_base.fvecs",
        "query": "data/wiki_ada_1M/wiki_ada_1M_query.fvecs",
        "gt": "data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs",
        "space": "l2",   # ada-002 vectors are unit-norm → L2 == cosine
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sift", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--max-points", type=int, default=10000,
                        help="0 = use the full dataset (no GT recomputation).")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--no-rust", action="store_true",
        help=("Skip the embedded `qps-recall-sweep` call and read the "
              "DiskANN / Orion series from "
              "`visualizations/sweep_orion_vs_parlayann_<ds>.json` "
              "(produced by `sweep_orion_vs_diskann_vs_parlayann.sh`). "
              "Use this when the head-to-head sweep is already current."),
    )
    args = parser.parse_args()

    # Cap every BLAS/OpenMP runtime to args.threads before lazy imports.
    # hnswlib already obeys set_num_threads, but the loaded numpy /
    # scipy stack may otherwise grab all cores during recall calc.
    t = str(args.threads)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = t

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
        base, queries, args.k, search_list_sizes, args.threads, args.trials,
        space=paths.get("space", "l2"),
    )
    hnsw_data = []
    for (labels, qps), ef in zip(hnsw_results, search_list_sizes):
        r = recall_at_k(labels.tolist(), gt.tolist(), args.k)
        hnsw_data.append([round(r, 4), round(qps)])
        print(f"  ef={ef:>4}  R@{args.k}={r:.4f}  QPS={qps:.0f}")
    all_results["hnsw"] = hnsw_data

    # ── 2. DiskANN + Orion (Rust sweep) ──
    if args.no_rust:
        # Pull from the head-to-head sweep produced by
        # `sweep_orion_vs_diskann_vs_parlayann.sh`. That file is the
        # source of truth for the 3-engine comparison and the Rust
        # numbers there are α/R/L-aligned with PA across the full
        # sweep.yaml `search_list_sizes`.
        sv_path = f"visualizations/sweep_orion_vs_parlayann_{args.dataset}.json"
        print(f"\n── DiskANN + Orion — reading {sv_path} (--no-rust) ──")
        with open(sv_path) as f:
            sv = json.load(f)
        all_results["diskann"] = sv["diskann"]
        all_results["orion"] = sv["orion"]
        if "parlayann" in sv:
            all_results["parlayann"] = sv["parlayann"]
    else:
        print("\n── DiskANN + Orion (Rust) ──")
        run_rust_sweep(paths["base"], paths["query"], paths["gt"], args.max_points)
        json_path = f"visualizations/qps_recall_{args.dataset}.json"
        with open(json_path) as f:
            rust_data = json.load(f)
        all_results["diskann"] = rust_data["diskann"]
        all_results["orion"] = rust_data["orion"]

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
