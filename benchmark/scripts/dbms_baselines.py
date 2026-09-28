#!/usr/bin/env python3
"""In-process baseline panel — LanceDB + USearch + Faiss HNSW+SQ.

In-process engines:

* **Faiss** — HNSW with trained 8-bit scalar quantization by default,
  M=16, efConstruction=200, and no exact reranking. Cosine uses
  normalized vectors with inner product; raw IP is not normalized.

* **LanceDB** — mainstream commercial vector database (Rust core,
  Python in-process via `pyarrow`). The HNSW config we use is
  `IVF_HNSW_SQ` with `num_partitions = 1`, which degenerates the IVF
  layer to a pass-through and leaves a plain HNSW (M=16, efC=200) as
  the only active index — closest apples-to-apples comparison with
  the `hnswlib` / `usearch_hnsw` rows. Search-time `ef` sweeps the
  recall-vs-QPS curve.

* **USearch** — Unum's HNSW search-engine library (open source;
  embedded in some commercial products, not a standalone DB
  itself). C++ + SIMD distance kernels (`simsimd`), same algorithm
  as hnswlib. Labelled as a library, not a DB, in the panel.

Note: Milvus Lite was attempted as a third engine but its embedded
build silently ignores the search-time `nprobe` override, so the
sweep collapsed to a single point per dataset (recall flat across
all `nprobe`). The full Milvus server (via docker / loopback gRPC)
would work but loses the in-process framing, so it's not part of
this panel.

The per-dataset metric is read from `DATASET_PATHS[ds]["metric"]`:
  * `l2`   — Euclidean; matches SIFT / GIST / Deep10M / Fashion-MNIST.
  * `cos`  — Cosine; matches GloVe / Wiki-ada / any unit-norm angular.
  * `ip`   — raw dot product; matches MS-MARCO BERT (non-unit-norm).

Outputs land in `visualizations/baseline_<dataset>.json` under the
keys `lancedb_hnsw`, `usearch_hnsw`, and `faiss_hnsw_sq`, next to whatever
else (hnswlib / faiss / annoy) already lives there.

Usage:
  python dbms_baselines.py --dataset sift --max-points 0 --threads 8
"""

import argparse
import json
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path

import numpy as np


# ── fvecs / ivecs readers (vectorised; copies from additional_baselines.py) ──

def read_fvecs(path, max_n=0):
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    record_floats = 1 + dim
    raw = np.fromfile(path, dtype=np.float32)
    total = raw.size // record_floats
    n = min(total, max_n) if max_n > 0 else total
    return raw[: n * record_floats].reshape(n, record_floats)[:, 1:].copy(), n, dim


def read_ivecs(path, max_n=0):
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    record_ints = 1 + dim
    raw = np.fromfile(path, dtype=np.int32)
    total = raw.size // record_ints
    n = min(total, max_n) if max_n > 0 else total
    return raw[: n * record_ints].reshape(n, record_ints)[:, 1:].copy()


def recall_at_k(results, gt, k):
    n = min(len(results), len(gt))
    total = 0.0
    for i in range(n):
        gt_set = set(int(x) for x in gt[i][:k])
        hits = sum(1 for r in results[i][:k] if int(r) in gt_set)
        total += hits / k
    return total / n


def faiss_vectors(vectors, metric):
    vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(vectors).all():
        raise ValueError("Vectors must be finite")
    if metric == "cos":
        norms = np.sqrt(np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64))
        if np.any(norms == 0):
            raise ValueError("Cosine requires nonzero vectors")
        vectors /= norms[:, None]
    return vectors


def run_faiss_hnsw_sq(base, queries, gt, k, ef_list, *, metric, threads,
                      trials, m=16, ef_construction=200, sq="8bit"):
    """Batch search including Python binding cost; no exact reranking."""
    import faiss

    if min(k, threads, trials, m, ef_construction) <= 0 or k > len(base):
        raise ValueError("Invalid k, thread count, trials, or HNSW parameters")
    if not ef_list or any(ef < k for ef in ef_list):
        raise ValueError("Every efSearch must be >= k")
    metric_id = {"l2": faiss.METRIC_L2, "ip": faiss.METRIC_INNER_PRODUCT,
                 "cos": faiss.METRIC_INNER_PRODUCT}[metric]
    qtype = {"8bit": faiss.ScalarQuantizer.QT_8bit,
             "fp16": faiss.ScalarQuantizer.QT_fp16}[sq]
    faiss.omp_set_num_threads(threads)
    t0 = time.perf_counter()
    xb = faiss_vectors(base, metric)
    index = faiss.IndexHNSWSQ(xb.shape[1], qtype, m, metric_id)
    index.hnsw.efConstruction = ef_construction
    index.train(xb)
    index.add(xb)
    build_time = time.perf_counter() - t0
    del xb
    # Include cosine query normalization in every timed batch.
    def search():
        xq = faiss_vectors(queries, metric) if metric == "cos" else queries
        return index.search(xq, k)[1]

    results = []
    for ef in ef_list:
        index.hnsw.efSearch = ef
        search()
        samples = []
        for _ in range(trials):
            start = time.perf_counter()
            ids = search()
            samples.append(len(queries) / (time.perf_counter() - start))
        recall = recall_at_k(ids, gt, k)
        qps = sorted(samples)[trials // 2]
        results.append([round(recall, 6), round(qps)])
        print(f"    ef={ef:>4}  R@{k}={recall:.6f}  QPS={qps:.0f}")
    metadata = dict(version=faiss.__version__, sq=sq, m=m,
                    ef_construction=ef_construction, ef_search=ef_list,
                    metric=metric, normalized=metric == "cos", rerank=False,
                    threads=threads, trials=trials, k=k,
                    timing="batch search; cosine query normalization included",
                    build_timing="base preparation + SQ training + graph construction")
    return results, build_time, metadata


# ── LanceDB (in-process, file-backed Rust core) ─────────────────────────────

def run_lancedb(base, queries, gt, k, ef_list, *, metric, threads, trials):
    """LanceDB HNSW via `IVF_HNSW_SQ` with `num_partitions = 1`. The
    `num_partitions = 1` knob degenerates the IVF layer to a pass-
    through (one centroid, every vector lives in it) and leaves a
    pure HNSW (M=16, efC=200) as the only active index — apples-to-
    apples with the `hnswlib` / `usearch_hnsw` rows in the same
    panel. The `SQ` suffix is scalar quantization on the HNSW edges'
    distance precomputes; the underlying base data stays f32."""
    import lancedb
    import pyarrow as pa

    distance = {"l2": "L2", "cos": "cosine", "ip": "dot"}[metric]
    dim = int(base.shape[1])
    n = int(base.shape[0])

    db_dir = Path(tempfile.mkdtemp(prefix="lancedb_"))
    db = lancedb.connect(str(db_dir))

    print(f"  Building LanceDB IVF_HNSW_SQ (M=16, efC=200, partitions=1, {n} pts, metric={distance})...")
    t0 = time.time()
    # Build the Arrow table in one shot — LanceDB's writer is happy
    # with millions of rows in a single batch as long as memory holds.
    # The vector column type must be a fixed-size list for the index
    # builder to know the dimension up front.
    vec_array = pa.FixedSizeListArray.from_arrays(
        pa.array(base.flatten(), type=pa.float32()),
        list_size=dim,
    )
    table = db.create_table(
        "bench",
        data=pa.table({"id": pa.array(range(n), type=pa.int64()), "vec": vec_array}),
    )
    table.create_index(
        metric=distance,
        vector_column_name="vec",
        index_type="IVF_HNSW_SQ",
        num_partitions=1,
        m=16,
        ef_construction=200,
    )
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    results = []
    for ef in ef_list:
        qps_samples = []
        ids_last = None
        for _ in range(trials):
            t = time.time()
            per_query = []
            for i in range(len(queries)):
                hits = (
                    table.search(queries[i].tolist())
                    .limit(k)
                    .ef(ef)
                    .nprobes(1)  # IVF degenerate (single partition)
                    .select(["id"])
                    .to_list()
                )
                per_query.append([int(h["id"]) for h in hits])
            wall = time.time() - t
            qps_samples.append(len(queries) / wall)
            ids_last = per_query
        qps = sorted(qps_samples)[trials // 2]
        r = recall_at_k(ids_last, gt.tolist(), k)
        results.append([round(r, 4), round(qps)])
        print(f"    ef={ef:>4}  R@{k}={r:.4f}  QPS={qps:.0f}")

    db.drop_table("bench")
    shutil.rmtree(db_dir, ignore_errors=True)
    return results, build_time


# ── (Legacy) Milvus Lite — kept for reference; nprobe is silently ────────────
#     dropped in the embedded build so the curve collapses to a single
#     point. See module docstring for details.

def _run_milvus_disabled(base, queries, gt, k, nprobe_list, *, metric, threads, trials):
    """Returns a list of `[recall, qps]` pairs, one per `nprobe`.

    Milvus Lite is the in-process Python build of Milvus core. The
    embedded variant deliberately drops HNSW (only FLAT / IVF_FLAT /
    AUTOINDEX are supported in `milvus-lite`); we use **explicit
    `IVF_FLAT`** with `nlist = min(256, n // 40)` — the same recipe
    `additional_baselines.py` uses for the FAISS IVF-Flat row, so the
    two appear in the panel on identical footing. AUTOINDEX was tried
    first but its `nprobe` parameter is silently dropped in the
    embedded resolver (recall + QPS came out flat across the whole
    sweep); explicit IVF_FLAT honours the override correctly."""
    from pymilvus import MilvusClient, DataType

    metric_type = {"l2": "L2", "cos": "COSINE", "ip": "IP"}[metric]
    dim = int(base.shape[1])
    n = int(base.shape[0])

    nlist = min(256, max(1, n // 40))
    # nprobe_list comes in from the caller — cap each entry at nlist so
    # we don't request more probes than centroids exist.
    nprobe_list = [min(p, nlist) for p in nprobe_list]
    # Dedupe in-order (sweep saturates once we ask for `nprobe == nlist`).
    seen = set()
    nprobe_list = [p for p in nprobe_list if not (p in seen or seen.add(p))]

    db_dir = Path(tempfile.mkdtemp(prefix="milvus_lite_"))
    db_path = db_dir / "milvus.db"
    # Milvus Lite uses the .db file directly as the `uri`. No `sqlite://`
    # scheme; the path itself triggers the embedded backend.
    client = MilvusClient(uri=str(db_path))
    collection = "bench"

    print(f"  Building Milvus IVF_FLAT (nlist={nlist}, {n} pts, metric={metric_type})...")
    if client.has_collection(collection):
        client.drop_collection(collection)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)
    client.create_collection(collection, schema=schema)

    # Build the index then load. Milvus Lite serializes inserts so we
    # batch in 50 k chunks to stay under the protobuf message limit.
    t0 = time.time()
    batch = 50_000
    for off in range(0, n, batch):
        end = min(off + batch, n)
        rows = [
            {"id": i, "vec": base[i].tolist()}
            for i in range(off, end)
        ]
        client.insert(collection, rows)
    client.flush(collection)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vec",
        index_type="IVF_FLAT",
        metric_type=metric_type,
        params={"nlist": nlist},
    )
    client.create_index(collection, index_params)
    client.load_collection(collection)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    queries_l = queries.tolist()
    results = []
    for nprobe in nprobe_list:
        qps_samples = []
        ids_last = None
        for _ in range(trials):
            t = time.time()
            search_res = client.search(
                collection,
                queries_l,
                anns_field="vec",
                limit=k,
                search_params={
                    "params": {"nprobe": nprobe},
                    "metric_type": metric_type,
                },
            )
            wall = time.time() - t
            qps_samples.append(len(queries_l) / wall)
            ids_last = [[hit["id"] for hit in row] for row in search_res]
        qps = sorted(qps_samples)[trials // 2]
        r = recall_at_k(ids_last, gt.tolist(), k)
        results.append([round(r, 4), round(qps)])
        print(f"    nprobe={nprobe:>4}  R@{k}={r:.4f}  QPS={qps:.0f}")

    client.drop_collection(collection)
    client.close()
    shutil.rmtree(db_dir, ignore_errors=True)
    return results, build_time


# ── USearch (Unum's in-process HNSW; SIMD C++ with Python bindings) ────────

def run_usearch(base, queries, gt, k, ef_list, *, metric, threads, trials):
    """USearch HNSW. The reference for **fast in-process HNSW** —
    Unum's single-file C++ implementation behind a Python binding,
    SIMD-tuned (the same `simsimd` distance kernels we'd reach for in
    raw NEON). Matches the M = 16, ef_construction = 200 recipe of the
    `hnswlib` row.

    (Qdrant's embedded Python backend was the original plan; it's
    officially capped at ~20 K points and bottlenecks on Python-level
    per-query serialisation at ~700 QPS at any scale we test, so it's
    not usable as a baseline. USearch is the closest in-process
    equivalent that holds up at 1 M+ points.)"""
    from usearch.index import Index, MetricKind

    mk = {"l2": MetricKind.L2sq, "cos": MetricKind.Cos, "ip": MetricKind.IP}[metric]
    dim = int(base.shape[1])
    n = int(base.shape[0])

    print(f"  Building USearch HNSW (M=16, efC=200, {n} pts, metric={mk})...")
    t0 = time.time()
    index = Index(
        ndim=dim,
        metric=mk,
        dtype="f32",
        connectivity=16,
        expansion_add=200,
        # expansion_search is set per-query via the `exact=False, threads=…`
        # path; we override it inside the sweep loop below.
    )
    # USearch wants u64 keys + a contiguous f32 array; both are zero-copy.
    keys = np.arange(n, dtype=np.uint64)
    index.add(keys, base.astype(np.float32), threads=threads)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")

    q_f32 = queries.astype(np.float32)
    results = []
    for ef in ef_list:
        index.expansion_search = ef
        qps_samples = []
        ids_last = None
        for _ in range(trials):
            t = time.time()
            matches = index.search(q_f32, count=k, threads=threads)
            wall = time.time() - t
            qps_samples.append(len(queries) / wall)
            ids_last = matches.keys.tolist() if hasattr(matches, "keys") else [m.keys.tolist() for m in matches]
        qps = sorted(qps_samples)[trials // 2]
        r = recall_at_k(ids_last, gt.tolist(), k)
        results.append([round(r, 4), round(qps)])
        print(f"    ef={ef:>4}  R@{k}={r:.4f}  QPS={qps:.0f}")

    return results, build_time


# ── Dataset paths ───────────────────────────────────────────────────────────

DATASET_PATHS = {
    "sift": {
        "base": "data/sift/sift_base.fvecs",
        "query": "data/sift/sift_query.fvecs",
        "gt": "data/sift/sift_groundtruth.ivecs",
        "metric": "l2",
    },
    "glove25": {
        "base": "data/glove25_norm/glove-25-angular_base.fvecs",
        "query": "data/glove25_norm/glove-25-angular_query.fvecs",
        "gt": "data/glove25_norm/glove-25-angular_groundtruth.ivecs",
        "metric": "cos",
    },
    "glove100": {
        "base": "data/glove100_norm/glove-100-angular_base.fvecs",
        "query": "data/glove100_norm/glove-100-angular_query.fvecs",
        "gt": "data/glove100_norm/glove-100-angular_groundtruth.ivecs",
        "metric": "cos",
    },
    "gist": {
        "base": "data/gist/gist_base.fvecs",
        "query": "data/gist/gist_query.fvecs",
        "gt": "data/gist/gist_groundtruth.ivecs",
        "metric": "l2",
    },
    "deep10m": {
        "base": "data/deep10m/deep10m_base.fvecs",
        "query": "data/deep10m/deep10m_query.fvecs",
        "gt": "data/deep10m/deep10m_groundtruth.ivecs",
        "metric": "l2",
    },
    "msmarco_bert_1M": {
        "base": "data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs",
        "query": "data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs",
        "gt": "data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs",
        "metric": "ip",
    },
    "wiki_ada_1M": {
        "base": "data/wiki_ada_1M/wiki_ada_1M_base.fvecs",
        "query": "data/wiki_ada_1M/wiki_ada_1M_query.fvecs",
        "gt": "data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs",
        "metric": "cos",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_PATHS.keys()))
    ap.add_argument(
        "--max-points", type=int, default=0,
        help="0 = full dataset.",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-list", type=int, nargs="+")
    ap.add_argument("--faiss-m", type=int, default=16)
    ap.add_argument("--faiss-ef-construction", type=int, default=200)
    ap.add_argument("--faiss-sq", choices=["8bit", "fp16"], default="8bit")
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--engines", default="lancedb,usearch,faiss",
        help="Comma-separated subset of {lancedb, usearch, faiss}. Default all.",
    )
    args = ap.parse_args()
    selected = [s.strip() for s in args.engines.split(",")]
    if not set(selected) <= {"lancedb", "usearch", "faiss"}:
        ap.error("Unknown engine; choose lancedb, usearch, faiss")
    if min(args.k, args.threads, args.trials, args.faiss_m,
           args.faiss_ef_construction) <= 0 or args.max_points < 0:
        ap.error("Counts must be positive; max-points must be nonnegative")

    # ── Cap every runtime to args.threads BEFORE we touch any lazy
    # imports (lancedb/usearch/faiss). LanceDB's Rust core spawns
    # Tokio + Rayon worker pools whose default is `num_cpus`, which
    # silently broke fairness against the explicitly-threaded
    # hnswlib / FAISS / USearch lanes. Tokio/Rayon read these env
    # vars on first runtime init, so setting them here (pre-import)
    # is what actually pins the thread count.
    t = str(args.threads)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS",
              "LANCE_IO_THREADS", "LANCE_CPU_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[k] = t

    paths = DATASET_PATHS[args.dataset]
    metric = paths["metric"]
    # ef ladder for HNSW (LanceDB + USearch). Mirrors the `hnswlib`
    # row in the existing baseline panel so the three HNSW lanes are
    # apples-to-apples.
    ef_list = [16, 20, 24, 32, 40, 48, 56, 64, 80, 100, 128, 160, 200, 256]
    if args.k > 10:
        ef_list = sorted(set([args.k] + [ef for ef in ef_list if ef >= args.k]
                             + [ef for ef in [384, 512, 768, 1024, 1536, 2048]
                                if ef >= args.k]))
    if args.ef_list is not None:
        ef_list = args.ef_list
    if any(ef < args.k for ef in ef_list) or ef_list != sorted(set(ef_list)):
        ap.error("ef-list must be strictly increasing and every value >= k")

    print(f"\n{'='*60}")
    print(f"  DBMS baselines: {args.dataset} (metric={metric}, max_points={args.max_points})")
    print(f"{'='*60}\n")

    print("Loading dataset...")
    base, n, dim = read_fvecs(paths["base"], args.max_points)
    queries, nq, _ = read_fvecs(paths["query"])
    gt = read_ivecs(paths["gt"])
    if not nq or args.k > n or queries.shape[1] != dim:
        ap.error("Empty queries, k exceeds base count, or mismatched dimensions")
    if (gt.shape[0] != nq or gt.shape[1] < args.k or
            np.any(gt < 0) or np.any(gt >= n)):
        print(f"Recomputing ground truth for {n} points...")
        from scipy.spatial.distance import cdist
        gt = np.empty((nq, args.k), dtype=np.int32)
        # Bound the distance matrix to approximately 64 MiB, not nq * n.
        batch = max(1, min(nq, (64 * 1024 * 1024) // (8 * n)))
        for start in range(0, nq, batch):
            q = queries[start:start + batch]
            if metric == "ip":
                dists = -(q @ base.T)
            else:
                dists = cdist(q, base, metric={"l2": "sqeuclidean", "cos": "cosine"}[metric])
            gt[start:start + len(q)] = np.argsort(dists, axis=1)[:, :args.k]
    print(f"  {n} base, {nq} queries, dim={dim}\n")

    suffix = "" if args.k == 10 else f"_k{args.k}"
    out_path = args.output or Path(f"visualizations/baseline_{args.dataset}{suffix}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            output = json.load(f)
        for field, value in dict(dataset=args.dataset, dimension=dim,
                                 num_points=n, threads=args.threads, k=args.k).items():
            if output.get(field) != value:
                ap.error(f"Existing output has incompatible {field}; use --output")
    else:
        output = {
            "dataset": args.dataset,
            "dimension": dim,
            "num_points": n,
            "threads": args.threads,
            "k": args.k,
        }
    output.setdefault("build_time", {})
    output.setdefault("engine_config", {})

    selected = [s.strip() for s in args.engines.split(",")]

    if "faiss" in selected:
        print("── Faiss HNSW+SQ ──")
        data, bt, metadata = run_faiss_hnsw_sq(
            base, queries, gt, args.k, ef_list, metric=metric,
            threads=args.threads, trials=args.trials, m=args.faiss_m,
            ef_construction=args.faiss_ef_construction, sq=args.faiss_sq,
        )
        output["faiss_hnsw_sq"] = data
        output["build_time"]["faiss_hnsw_sq"] = round(bt, 3)
        output["engine_config"]["faiss_hnsw_sq"] = metadata

    if "lancedb" in selected:
        print("── LanceDB (IVF_HNSW_SQ, partitions=1) ──")
        try:
            data, bt = run_lancedb(
                base, queries, gt, args.k, ef_list,
                metric=metric, threads=args.threads, trials=args.trials,
            )
            output["lancedb_hnsw"] = data
            output["build_time"]["lancedb_hnsw"] = round(bt, 3)
            # Drop stale Milvus rows from previous runs so the plot
            # script doesn't render dead lanes.
            output.pop("milvus_auto", None)
            output.pop("milvus_ivf", None)
        except Exception as e:
            print(f"  [LanceDB error] {e}")

    if "usearch" in selected:
        print("\n── USearch (HNSW M=16 efC=200) ──")
        try:
            data, bt = run_usearch(
                base, queries, gt, args.k, ef_list,
                metric=metric, threads=args.threads, trials=args.trials,
            )
            output["usearch_hnsw"] = data
            output["build_time"]["usearch_hnsw"] = round(bt, 3)
        except Exception as e:
            print(f"  [USearch error] {e}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
