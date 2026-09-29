#!/usr/bin/env python3
"""Embed a 1M-passage / 10K-query slice of MS-MARCO with a
sentence-transformers BERT model and emit Yael `.fvecs` + `.ivecs`
files compatible with `orion`.

Pipeline:
  1. Pull the `corpus` and `queries` subsets of
     `sentence-transformers/msmarco` from HuggingFace
     (raw text + IDs, ~3 GB parquet for corpus, ~50 MB for queries).
  2. Random-sample 1M passages / 10K queries with a fixed seed so the
     split is reproducible.
  3. Encode both with `sentence-transformers/msmarco-bert-base-dot-v5`
     (BERT-base, 768D, dot-product / MIPS). Runs on MPS when available.
  4. Emit `base.fvecs`, `query.fvecs`, and `groundtruth.ivecs`
     (top-100 exact MIPS NN via a single batched matmul).

Output layout matches the `data/{dataset}/{dataset}_base.fvecs` style
already used by `prepare_parlayann_data.sh` and `orion.rs`.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


from vector_io import write_fvecs, write_ivecs



def load_texts(split_name: str, sample_n: int, seed: int, text_col: str, id_col: str):
    """Returns ``(ids, texts)`` for the requested split, both as Python lists."""
    from datasets import load_dataset

    print(f"[{split_name}] loading sentence-transformers/msmarco split={split_name}...")
    t0 = time.time()
    ds = load_dataset("sentence-transformers/msmarco", split_name, split="train")
    n_total = len(ds)
    print(f"[{split_name}] {n_total} rows ({time.time()-t0:.1f}s)")

    if sample_n >= n_total:
        idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=sample_n, replace=False)
        idx.sort()
    print(f"[{split_name}] sampling {len(idx)} rows (seed={seed})")

    # `.select(...)` returns a Dataset; pull text/id cols out into Python lists
    # in a single arrow→pylist conversion to avoid per-row __getitem__ overhead.
    sub = ds.select(idx)
    return sub[id_col], sub[text_col]


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed(texts, model_name: str, device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[embed] model={model_name} device={device} batch={batch_size}")
    model = SentenceTransformer(model_name, device=device)
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    dt = time.time() - t0
    print(f"[embed] {emb.shape} ({dt:.1f}s, {len(texts)/dt:.1f} it/s)")
    assert emb.dtype == np.float32, f"unexpected dtype {emb.dtype}"
    return emb


def brute_force_mips_topk(
    base: np.ndarray, query: np.ndarray, k: int, device: str
) -> np.ndarray:
    """Exact top-k inner-product neighbors. Streams queries in chunks
    so the (10K × 1M) score matrix never materialises in full."""
    import torch

    print(f"[gt] brute-force MIPS top-{k} on {device}")
    base_t = torch.from_numpy(base).to(device)  # [N, D]
    query_t = torch.from_numpy(query).to(device)  # [Q, D]
    N, D = base_t.shape
    Q = query_t.shape[0]

    # Chunk so a single matmul fits in MPS budget. 512 queries × 1M base ×
    # f32 = 2 GB scores per chunk; halve if MPS OOMs.
    QCHUNK = 256
    out = np.empty((Q, k), dtype=np.int32)
    t0 = time.time()
    for q0 in range(0, Q, QCHUNK):
        q1 = min(q0 + QCHUNK, Q)
        scores = query_t[q0:q1] @ base_t.T  # [q, N]
        # topk gives sorted descending — exactly what we want for MIPS.
        _, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        out[q0:q1] = idx.to("cpu").numpy().astype(np.int32)
        if q0 % (QCHUNK * 4) == 0:
            done = q1 / Q
            elapsed = time.time() - t0
            print(f"  gt: {q1}/{Q} ({100*done:.0f}%, {elapsed:.0f}s)")
    print(f"[gt] done ({time.time()-t0:.1f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/msmarco_bert_1M")
    ap.add_argument("--dataset-name", default="msmarco_bert_1M",
                    help="Used to name files: <dataset>_base.fvecs etc.")
    ap.add_argument("--model", default="sentence-transformers/msmarco-bert-base-dot-v5")
    ap.add_argument("--n-base", type=int, default=1_000_000)
    ap.add_argument("--n-query", type=int, default=10_000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                    help="mps / cuda / cpu; auto-detect by default.")
    ap.add_argument("--skip-base", action="store_true",
                    help="Reuse an existing base.fvecs; skip corpus encode.")
    ap.add_argument("--skip-query", action="store_true",
                    help="Reuse an existing query.fvecs; skip query encode.")
    ap.add_argument("--skip-gt", action="store_true",
                    help="Skip groundtruth (used when iterating on the encode).")
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"[main] device={device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = str(out_dir / f"{args.dataset_name}_base.fvecs")
    query_path = str(out_dir / f"{args.dataset_name}_query.fvecs")
    gt_path = str(out_dir / f"{args.dataset_name}_groundtruth.ivecs")

    base_emb = None
    query_emb = None

    if not args.skip_base:
        _, base_texts = load_texts(
            "corpus", args.n_base, args.seed, text_col="passage", id_col="passage_id"
        )
        base_emb = embed(base_texts, args.model, device, args.batch_size)
        write_fvecs(base_path, base_emb)

    if not args.skip_query:
        _, query_texts = load_texts(
            "queries", args.n_query, args.seed, text_col="query", id_col="query_id"
        )
        query_emb = embed(query_texts, args.model, device, args.batch_size)
        write_fvecs(query_path, query_emb)

    if not args.skip_gt:
        # If we skipped one or both, fall back to mmap'ing the existing
        # .fvecs (drop the per-row dim header to recover the matrix).
        def load_fvecs(path):
            print(f"[gt] loading {path}")
            raw = np.fromfile(path, dtype=np.float32)
            # First u32 is the dim; deduce per-row record size from it.
            dim = int(raw.view(np.uint32)[0])
            rec = dim + 1
            assert raw.size % rec == 0
            n = raw.size // rec
            return raw.reshape(n, rec)[:, 1:].copy()

        if base_emb is None:
            base_emb = load_fvecs(base_path)
        if query_emb is None:
            query_emb = load_fvecs(query_path)

        ids = brute_force_mips_topk(base_emb, query_emb, args.top_k, device)
        write_ivecs(gt_path, ids)

    print("[main] done.")


if __name__ == "__main__":
    main()
