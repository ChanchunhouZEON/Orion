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
import struct
import time
import numpy as np

def read_fvecs(path, max_n=0):
    """Vectorised .fvecs reader (np.fromfile + reshape)."""
    with open(path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
    record_floats = 1 + dim
    raw = np.fromfile(path, dtype=np.float32)
    total = raw.size // record_floats
    n = min(total, max_n) if max_n > 0 else total
    data = raw[: n * record_floats].reshape(n, record_floats)[:, 1:].copy()
    return data, n, dim

def read_ivecs(path, max_n=0):
    """Vectorised .ivecs reader."""
    with open(path, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
    record_ints = 1 + dim
    raw = np.fromfile(path, dtype=np.int32)
    total = raw.size // record_ints
    n = min(total, max_n) if max_n > 0 else total
    return raw[: n * record_ints].reshape(n, record_ints)[:, 1:].copy()

def recall_at_k(results, ground_truth, k):
    n = min(len(results), len(ground_truth))
    total = 0.0
    for i in range(n):
        gt_set = set(int(x) for x in ground_truth[i][:k])
        hits = sum(1 for r in results[i][:k] if int(r) in gt_set)
        total += hits / k
    return total / n

def run_faiss_ivf_flat(base, queries, gt, k, nprobe_list, nlist=100, num_threads=8, trials=5):
    import faiss
    faiss.omp_set_num_threads(num_threads)
    dim = base.shape[1]
    n = base.shape[0]

    print(f"  Building FAISS IVF-Flat (nlist={nlist}, {n} pts)...")
    t0 = time.time()
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
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

def run_faiss_ivf_pq(base, queries, gt, k, nprobe_list, nlist=100, m_pq=16, num_threads=8, trials=5):
    import faiss
    faiss.omp_set_num_threads(num_threads)
    dim = base.shape[1]
    n = base.shape[0]

    print(f"  Building FAISS IVF-PQ (nlist={nlist}, m={m_pq}, {n} pts)...")
    t0 = time.time()
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8)
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

def run_annoy(base, queries, gt, k, search_k_list, n_trees=50, num_threads=-1, trials=5):
    from annoy import AnnoyIndex
    dim = base.shape[1]
    n = base.shape[0]

    print(f"  Building Annoy (n_trees={n_trees}, {n} pts)...")
    t0 = time.time()
    index = AnnoyIndex(dim, 'euclidean')
    for i in range(n):
        index.add_item(i, base[i])
    index.build(n_trees)
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
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    paths = DATASET_PATHS[args.dataset]

    print(f"\n{'='*60}")
    print(f"  Additional Baselines: {args.dataset} ({args.max_points} pts)")
    print(f"{'='*60}\n")

    print("Loading dataset...")
    base, n, dim = read_fvecs(paths["base"], args.max_points)
    queries, nq, _ = read_fvecs(paths["query"])
    gt = read_ivecs(paths["gt"])
    if n < gt.shape[0] or np.max(gt) >= n:
        print(f"Recomputing ground truth for {n} points...")
        from scipy.spatial.distance import cdist
        dists = cdist(queries, base, metric='sqeuclidean')
        gt = np.argsort(dists, axis=1)[:, :100].astype(np.int32)
    print(f"  {n} base, {nq} queries, dim={dim}\n")

    # Load existing baseline JSON
    out_path = f"visualizations/baseline_{args.dataset}.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            output = json.load(f)
    else:
        output = {"dataset": args.dataset, "dimension": dim, "num_points": n,
                  "threads": args.threads, "k": args.k}

    # ── FAISS IVF-Flat ──
    nlist = min(256, n // 40)
    nprobe_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    nprobe_list = [p for p in nprobe_list if p <= nlist]
    print("── FAISS IVF-Flat ──")
    ivf_flat_data, ivf_flat_build = run_faiss_ivf_flat(
        base, queries, gt, args.k, nprobe_list, nlist, args.threads, args.trials)
    output["faiss_ivf_flat"] = ivf_flat_data

    # ── FAISS IVF-PQ ──
    m_pq = max(1, dim // 8)  # sub-quantizer count
    if dim % m_pq != 0:
        m_pq = max(1, dim // 4)
    if dim % m_pq != 0:
        m_pq = max(1, dim // 2)
    print(f"\n── FAISS IVF-PQ (m={m_pq}) ──")
    ivf_pq_data, ivf_pq_build = run_faiss_ivf_pq(
        base, queries, gt, args.k, nprobe_list, nlist, m_pq, args.threads, args.trials)
    output["faiss_ivf_pq"] = ivf_pq_data

    # ── Annoy ──
    search_k_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    print("\n── Annoy ──")
    annoy_data, annoy_build = run_annoy(
        base, queries, gt, args.k, search_k_list, n_trees=50, trials=args.trials)
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
