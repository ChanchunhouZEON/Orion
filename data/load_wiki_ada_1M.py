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
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

REPO = "nlpkevinl/wikipedia_openai_embeddings"
# Use the LFS resolve URL — works without an HF token at ~50 MB/s.
SHARD_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
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
        print(f"[shard] already present: {dst} ({dst.stat().st_size/1024/1024:.0f} MB)")
        return dst
    url = SHARD_URL.format(repo=REPO, name=quote(shard_name))
    print(f"[shard] downloading {shard_name} from {REPO}")
    local_dir.mkdir(parents=True, exist_ok=True)
    rc = os.system(
        f"curl -L --fail --progress-bar -C - -o {dst!s} {url!r}"
    )
    if rc != 0:
        raise RuntimeError(f"curl failed (rc={rc}) for {shard_name}")
    return dst


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
                text = req["input"]
                emb = resp["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError):
                continue
            if not isinstance(emb, list) or len(emb) != 1536:
                continue
            yield text, emb


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    """`.fvecs`: per-row [u32 dim][dim × f32]."""
    n, dim = arr.shape
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    out = np.empty((n, dim + 1), dtype=np.float32)
    out.view(np.uint32)[:, 0] = dim
    out[:, 1:] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fo:
        fo.write(out.tobytes())
    print(f"  → {path}  ({n} × {dim}, {out.nbytes/1024/1024:.1f} MB)")


def write_ivecs(path: Path, ids: np.ndarray) -> None:
    n, k = ids.shape
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    out = np.empty((n, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = ids
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fo:
        fo.write(out.tobytes())
    print(f"  → {path}  ({n} queries × top-{k})")


def collect_embeddings(shard_paths, n_target: int) -> tuple[list[str], np.ndarray]:
    """Read shards in order until `n_target` valid records are collected.
    Returns the parallel lists of text + a stacked embedding matrix."""
    texts: list[str] = []
    embs: list[np.ndarray] = []
    t0 = time.time()
    for shard in shard_paths:
        print(f"[collect] streaming {shard.name}...")
        for text, emb in stream_records(shard):
            texts.append(text)
            embs.append(np.asarray(emb, dtype=np.float32))
            if len(texts) % 100_000 == 0:
                rate = len(texts) / max(1.0, time.time() - t0)
                print(f"  collected {len(texts):,} ({rate:.0f}/s)")
            if len(texts) >= n_target:
                break
        if len(texts) >= n_target:
            break
    if len(texts) < n_target:
        raise RuntimeError(
            f"only {len(texts)} valid records found; need {n_target} "
            "(download more shards or lower --n-total)"
        )
    print(f"[collect] done: {len(texts):,} records in {time.time()-t0:.0f}s")
    return texts, np.stack(embs)


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
    ap.add_argument("--skip-gt", action="store_true")
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"[main] device={device}, seed={args.seed}")

    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    base_path = out_dir / f"{args.dataset_name}_base.fvecs"
    query_path = out_dir / f"{args.dataset_name}_query.fvecs"
    gt_path = out_dir / f"{args.dataset_name}_groundtruth.ivecs"

    # Download up to num_shards; the collect loop stops as soon as it
    # has enough records, so on a typical shard density only 1–2 are
    # read end-to-end.
    shards_to_use = SHARD_NAMES[: args.num_shards]
    shard_paths = [ensure_shard(shard_dir, n) for n in shards_to_use]

    n_total = args.n_base + args.n_query
    texts, embs = collect_embeddings(shard_paths, n_total)
    print(f"[main] collected {len(texts):,} valid (text, 1536-D) pairs")

    # Deterministic disjoint split of base vs query.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(texts))
    base_idx = perm[: args.n_base]
    query_idx = perm[args.n_base : args.n_base + args.n_query]

    base_emb = embs[base_idx]
    query_emb = embs[query_idx]
    print(f"[main] base={base_emb.shape}, query={query_emb.shape}")

    write_fvecs(base_path, base_emb)
    write_fvecs(query_path, query_emb)

    if not args.skip_gt:
        ids = brute_force_mips_topk(base_emb, query_emb, args.top_k, device)
        write_ivecs(gt_path, ids)

    print("[main] done.")


if __name__ == "__main__":
    main()
