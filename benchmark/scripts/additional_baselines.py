#!/usr/bin/env python3
"""
Additional baseline algorithms: FAISS (IVF-Flat, IVF-PQ), Annoy, PyNNDescent.
Appends results to existing baseline_{dataset}.json.

Usage:
  python benchmark/scripts/additional_baselines.py --dataset sift --max-points 100000
"""

import argparse
import json
import os
import time
import numpy as np

from benchmark_support import read_fvecs, read_ivecs, recall_at_k, ROOT
from benchmark_support import BASELINE_DATASETS, dataset_paths, evaluation_ground_truth, binary_path


def run_faiss_ivf_flat(base, queries, gt, k, nprobe_list, nlist=100, num_threads=8, trials=5, metric="l2"):
    import faiss
    faiss.omp_set_num_threads(num_threads)
    dim = base.shape[1]
    n = base.shape[0]

    print(f"  Building FAISS IVF-Flat (nlist={nlist}, {n} pts)...")
    t0 = time.time()
    metric_id = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    quantizer = faiss.IndexFlat(dim, metric_id)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric_id)
    index.train(base)
    index.add(base)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    results = []
    for nprobe in nprobe_list:
        index.nprobe = nprobe
        qps_samples = []
        D = None
        I = None
        for _ in range(trials):
            t = time.time()
            D, I = index.search(queries, k)
            wall = time.time() - t
            qps_samples.append(len(queries) / wall)
        qps = sorted(qps_samples)[trials // 2]
        r = recall_at_k(I.tolist(), gt.tolist(), k)
        results.append([round(r, 4), round(qps)])
        print(f"    nprobe={nprobe:>4}  R@{k}={r:.4f}  QPS={qps:.0f}")

    return results, build_time

def run_faiss_ivf_pq(base, queries, gt, k, nprobe_list, nlist=100, m_pq=16, num_threads=8, trials=5, metric="l2"):
    import faiss
    faiss.omp_set_num_threads(num_threads)
    dim = base.shape[1]
    n = base.shape[0]

    print(f"  Building FAISS IVF-PQ (nlist={nlist}, m={m_pq}, {n} pts)...")
    t0 = time.time()
    metric_id = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    quantizer = faiss.IndexFlat(dim, metric_id)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8, metric_id)
    index.train(base)
    index.add(base)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    results = []
    for nprobe in nprobe_list:
        index.nprobe = nprobe
        qps_samples = []
        for _ in range(trials):
            t = time.time()
            D, I = index.search(queries, k)
            wall = time.time() - t
            qps_samples.append(len(queries) / wall)
        qps = sorted(qps_samples)[trials // 2]
        r = recall_at_k(I.tolist(), gt.tolist(), k)
        results.append([round(r, 4), round(qps)])
        print(f"    nprobe={nprobe:>4}  R@{k}={r:.4f}  QPS={qps:.0f}")

    return results, build_time

def run_annoy(base, queries, gt, k, search_k_list, n_trees=50, num_threads=8, trials=5, metric="l2"):
    from annoy import AnnoyIndex
    dim = base.shape[1]
    n = base.shape[0]

    print(f"  Building Annoy (n_trees={n_trees}, {n} pts, threads={num_threads})...")
    t0 = time.time()
    index = AnnoyIndex(dim, {"l2": "euclidean", "ip": "dot", "cos": "angular"}[metric])
    for i in range(n):
        index.add_item(i, base[i])
    # n_jobs=N caps the parallel tree-build to N threads. `-1` (the
    # AnnoyIndex.build default) means "all cores", which was unfair
    # vs the explicit 8-thread setting on every other baseline.
    index.build(n_trees, n_jobs=num_threads)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    results = []
    for search_k in search_k_list:
        qps_samples = []
        all_ids = None
        for _ in range(trials):
            ids_list = []
            t = time.time()
            for q in queries:
                ids = index.get_nns_by_vector(q.tolist(), k, search_k=search_k)
                ids_list.append(ids)
            wall = time.time() - t
            qps_samples.append(len(queries) / wall)
            all_ids = ids_list
        qps = sorted(qps_samples)[trials // 2]
        r = recall_at_k(all_ids, gt.tolist(), k)
        results.append([round(r, 4), round(qps)])
        print(f"    search_k={search_k:>5}  R@{k}={r:.4f}  QPS={qps:.0f}")

    return results, build_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sift", choices=BASELINE_DATASETS)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    os.chdir(ROOT)
    if min(args.k, args.threads, args.trials) <= 0 or args.max_points < 0:
        parser.error("k, threads and trials must be positive; max-points must be nonnegative")

    # Cap every BLAS/OpenMP runtime to args.threads before lazy imports.
    # See dbms_baselines.py main() for rationale.
    t = str(args.threads)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = t

    paths = dataset_paths(args.dataset)

    print(f"\n{'='*60}")
    print(f"  Additional Baselines: {args.dataset} ({args.max_points} pts)")
    print(f"{'='*60}\n")

    print("Loading dataset...")
    base, n, dim = read_fvecs(paths["base"], args.max_points)
    queries, nq, _ = read_fvecs(paths["query"])
    gt = read_ivecs(paths["gt"])
    if not nq or args.k > n or queries.shape[1] != dim:
        parser.error("Empty queries, k exceeds base count, or mismatched dimensions")
    gt = evaluation_ground_truth(base, queries, gt, args.k, paths["metric"])
    print(f"  {n} base, {nq} queries, dim={dim}\n")

    if paths["metric"] == "cos":
        from dbms_baselines import faiss_vectors
        base = faiss_vectors(base, "cos")
        queries = faiss_vectors(queries, "cos")

    # Load existing baseline JSON
    out_path = f"visualizations/baseline_{args.dataset}.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            output = json.load(f)
    else:
        output = {"dataset": args.dataset, "dimension": dim, "num_points": n,
                  "threads": args.threads, "k": args.k}

    # ── FAISS IVF-Flat ──
    nlist = max(1, min(256, n // 40))
    nprobe_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    nprobe_list = [p for p in nprobe_list if p <= nlist]
    print("── FAISS IVF-Flat ──")
    ivf_flat_data, ivf_flat_build = run_faiss_ivf_flat(
        base, queries, gt, args.k, nprobe_list, nlist, args.threads, args.trials, metric=paths["metric"])
    output["faiss_ivf_flat"] = ivf_flat_data

    # ── FAISS IVF-PQ ──
    m_pq = max(m for m in range(1, max(1, dim // 8) + 1) if dim % m == 0)
    print(f"\n── FAISS IVF-PQ (m={m_pq}) ──")
    ivf_pq_data, ivf_pq_build = run_faiss_ivf_pq(
        base, queries, gt, args.k, nprobe_list, nlist, m_pq, args.threads, args.trials, metric=paths["metric"])
    output["faiss_ivf_pq"] = ivf_pq_data

    # ── Annoy ──
    search_k_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    print("\n── Annoy ──")
    annoy_data, annoy_build = run_annoy(
        base, queries, gt, args.k, search_k_list, n_trees=50, num_threads=args.threads, trials=args.trials, metric=paths["metric"])
    output["annoy"] = annoy_data

    # Save build times
    if "build_time" not in output:
        output["build_time"] = {}
    output["build_time"]["faiss_ivf_flat"] = round(ivf_flat_build, 3)
    output["build_time"]["faiss_ivf_pq"] = round(ivf_pq_build, 3)
    output["build_time"]["annoy"] = round(annoy_build, 3)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
