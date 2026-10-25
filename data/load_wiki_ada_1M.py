#!/usr/bin/env python3
"""Build a 1M base + 10K query slice of Wikipedia + OpenAI ada-002
embeddings from `nlpkevinl/wikipedia_openai_embeddings`.

Why we're not running the embedding model: ada-002 is closed-weights.
We consume the precomputed embeddings published in nlpkevinl's HF repo
(confirmed `model: "text-embedding-ada-002"` in the request half of
each record).

The repo's shards are `.jsonl.gz`, one *request/response* pair per
line:

    [
      {"model": "text-embedding-ada-002", "input": "<passage text>"},
      {"object": "list",
       "data": [{"object": "embedding", "index": 0,
                 "embedding": [<1536 floats>]}],
       "model": "text-embedding-ada-002",
       "usage": {...}}
    ]

Each shard is ~7.7 GB compressed; two shards comfortably cover the
~1.01M target. Streaming through them avoids needing 15+ GB of RAM.

Output (matches `data/{dataset}/{dataset}_*` convention so the rest of
the harness picks it up unchanged):

    data/wiki_ada_1M/wiki_ada_1M_base.fvecs    1M × 1536 × f32 ≈ 5.7 GB
    data/wiki_ada_1M/wiki_ada_1M_query.fvecs   10K × 1536 × f32 ≈ 60 MB
    data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs   10K × 100 × i32

Queries are *held-out from the corpus* (ann-benchmarks self-query
convention) — sampled deterministically with the same RNG so the
split is reproducible.
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

REPO = "nlpkevinl/wikipedia_openai_embeddings"
REVISION = "99ceeb82bed4b59fe9412245cf450da992f5b198"
SHARD_SHA256 = {
    "wikipedia_passages_shard_1-00.jsonl.gz": "522a2e4866fa7920e8cd663efb6402138a6545217db274287a573137752185aa",
    "wikipedia_passages_shard_1-01.jsonl.gz": "38d463740abe5d11d40e1022c9a1dcbde61ffba9f5d053bb602cfc28380e5679",
    "wikipedia_passages_shard_1-02.jsonl.gz": "7587a9245198b5d8df29b932f1b68e84b8176669e3d9d9c51992ba878bcbf26f",
    "wikipedia_passages_shard_1-03.jsonl.gz": "5b339c0200de011dee07d00cfe3e22a1434211ff7564c02ed42d720240d49528",
}
# Use the LFS resolve URL — works without an HF token at ~50 MB/s.
SHARD_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
)
SHARD_NAMES = [
    "wikipedia_passages_shard_1-00.jsonl.gz",
    "wikipedia_passages_shard_1-01.jsonl.gz",
    "wikipedia_passages_shard_1-02.jsonl.gz",
    "wikipedia_passages_shard_1-03.jsonl.gz",
]


def ensure_shard(local_dir: Path, shard_name: str) -> Path:
    """Download a single shard via `curl -C -` (resumable, single
    stream — Hugging Face LFS does not throttle the way Azure
    static-web does, so multi-stream gives no benefit)."""
    dst = local_dir / shard_name
    if dst.exists():
        verify_shard(dst)
        return dst
    url = SHARD_URL.format(repo=REPO, revision=REVISION, name=quote(shard_name))
    print(f"[shard] downloading {shard_name} from {REPO}")
    local_dir.mkdir(parents=True, exist_ok=True)
    # A partial download is resumable but must never look like a completed shard.
    import subprocess
    partial = dst.with_suffix(dst.suffix + ".partial")
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--connect-timeout", "30", "--progress-bar", "-C", "-",
                    "-o", str(partial), url], check=True)
    verify_shard(partial, shard_name)
    partial.replace(dst)
    return dst


def verify_shard(path: Path, name=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHARD_SHA256[name or path.name]:
        raise ValueError(f"Shard checksum mismatch: {path}")


def stream_records(shard_path: Path):
    """Yield ``(text, embedding[1536])`` from a `.jsonl.gz` shard.

    Skips malformed lines / null embeddings rather than crashing —
    the input is a raw API dump and a small number of rate-limit
    error responses are mixed in.
    """
    with gzip.open(shard_path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (isinstance(obj, list) and len(obj) == 2):
                continue
            req, resp = obj
            try:
                if req["model"] != "text-embedding-ada-002":
                    continue
                text = req["input"]
                emb = resp["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError):
                continue
            if not isinstance(emb, list) or len(emb) != 1536:
                continue
            try:
                vector = np.asarray(emb, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                continue
            yield text, vector


from vector_io import write_fvecs, write_ivecs



def collect_embeddings(shard_paths, n_target: int) -> np.ndarray:
    """Keep one preallocated matrix, not passage text or duplicate arrays."""
    embs = np.empty((n_target, 1536), dtype=np.float32)
    count = 0
    t0 = time.time()
    for shard in shard_paths:
        print(f"[collect] streaming {shard.name}...")
        for _, emb in stream_records(shard):
            embs[count] = emb
            count += 1
            if count % 100_000 == 0:
                rate = count / max(1.0, time.time() - t0)
                print(f"  collected {count:,} ({rate:.0f}/s)")
            if count >= n_target:
                break
        if count >= n_target:
            break
    if count < n_target:
        raise RuntimeError(
            f"only {count} valid records found; need {n_target} "
            "(download more shards or lower --n-total)"
        )
    print(f"[collect] done: {count:,} records in {time.time()-t0:.0f}s")
    return embs


def faiss_mips_topk(base, query, k, threads=8):
    """Exhaustive float32 inner-product search, with bounded query batches."""
    import faiss
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    out = np.empty((len(query), k), dtype=np.int32)
    for start in range(0, len(query), 256):
        stop = min(start + 256, len(query))
        _, ids = index.search(query[start:stop], k)
        out[start:stop] = ids
        print(f"[gt] {stop}/{len(query)}", flush=True)
    return out


def brute_force_mips_topk(
    base: np.ndarray, query: np.ndarray, k: int, device: str
) -> np.ndarray:
    """Exact top-k dot-product NN. Streams 256 queries per matmul."""
    import torch

    print(f"[gt] brute-force MIPS top-{k} on {device}")
    base_t = torch.from_numpy(base).to(device)
    query_t = torch.from_numpy(query).to(device)
    Q = query_t.shape[0]

    QCHUNK = 256
    out = np.empty((Q, k), dtype=np.int32)
    t0 = time.time()
    for q0 in range(0, Q, QCHUNK):
        q1 = min(q0 + QCHUNK, Q)
        scores = query_t[q0:q1] @ base_t.T
        _, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        out[q0:q1] = idx.to("cpu").numpy().astype(np.int32)
        if q0 % (QCHUNK * 4) == 0:
            print(f"  gt: {q1}/{Q} ({100*q1/Q:.0f}%, {time.time()-t0:.0f}s)")
    print(f"[gt] done ({time.time()-t0:.1f}s)")
    return out


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="data/wiki_ada_1M/_shards",
                    help="Cache directory for the downloaded jsonl.gz shards.")
    ap.add_argument("--out-dir", default="data/wiki_ada_1M")
    ap.add_argument("--dataset-name", default="wiki_ada_1M")
    ap.add_argument("--n-base", type=int, default=1_000_000)
    ap.add_argument("--n-query", type=int, default=10_000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--num-shards", type=int, default=2,
                    help="How many of the SHARD_NAMES to consider; "
                         "stops downloading early once n-base + n-query "
                         "valid records are collected.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gt-backend", choices=["faiss", "torch"], default="faiss")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--raw-inner-product", action="store_true",
                    help="Keep original norms instead of the default cosine normalization.")
    ap.add_argument("--skip-gt", action="store_true")
    args = ap.parse_args()

    if args.n_base < args.top_k or min(args.n_query, args.top_k, args.threads) < 1:
        ap.error("Require n_base >= top_k and positive query count, k, threads")
    if not 1 <= args.num_shards <= len(SHARD_NAMES):
        ap.error("num-shards is outside the available shard list")
    device = (args.device or pick_device()) if args.gt_backend == "torch" and not args.skip_gt else "cpu"
    print(f"[main] device={device}, seed={args.seed}")

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    base_path = out_dir / f"{args.dataset_name}_base.fvecs"
    query_path = out_dir / f"{args.dataset_name}_query.fvecs"
    gt_path = out_dir / f"{args.dataset_name}_groundtruth.ivecs"
    for path in (base_path, query_path, gt_path, out_dir / "manifest.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new out-dir")

    # Download up to num_shards; the collect loop stops as soon as it
    # has enough records, so on a typical shard density only 1–2 are
    # read end-to-end.
    shards_to_use = SHARD_NAMES[: args.num_shards]
    used_shards = []
    def shard_paths():
        for name in shards_to_use:
            path = ensure_shard(shard_dir, name)
            used_shards.append(name)
            yield path

    n_total = args.n_base + args.n_query
    embs = collect_embeddings(shard_paths(), n_total)
    print(f"[main] collected {len(embs):,} valid 1536-D vectors")
    if not args.raw_inner_product:
        for start in range(0, len(embs), 16384):
            batch = embs[start:start + 16384]
            batch /= np.linalg.norm(batch, axis=1, keepdims=True)

    # Deterministic disjoint split of base vs query.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(embs))
    base_idx = perm[: args.n_base]
    query_idx = perm[args.n_base : args.n_base + args.n_query]

    base_emb = embs[base_idx]
    query_emb = embs[query_idx]
    del embs
    print(f"[main] base={base_emb.shape}, query={query_emb.shape}")

    write_fvecs(base_path, base_emb)
    write_fvecs(query_path, query_emb)

    if not args.skip_gt:
        ids = (faiss_mips_topk(base_emb, query_emb, args.top_k, args.threads)
               if args.gt_backend == "faiss" else
               brute_force_mips_topk(base_emb, query_emb, args.top_k, device))
        write_ivecs(gt_path, ids)
    manifest = dict(source=REPO, revision=REVISION,
                    shards={name: SHARD_SHA256[name] for name in used_shards},
                    model="text-embedding-ada-002", dimension=1536,
                    n_base=args.n_base, n_query=args.n_query, seed=args.seed,
                    split="Seeded permutation of the first n_base+n_query valid records; disjoint row IDs",
                    metric="raw-inner-product" if args.raw_inner_product else "normalized-inner-product",
                    gt_backend=None if args.skip_gt else args.gt_backend, top_k=args.top_k,
                    numpy_version=np.__version__)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("[main] done.")


if __name__ == "__main__":
    main()
