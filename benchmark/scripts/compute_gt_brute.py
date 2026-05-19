#!/usr/bin/env python3
"""Brute-force top-k ground-truth computer — used when we truncate a
dataset to a size different from the one the provided `.ivecs` gt was
computed on (e.g. GIST-100k sampled from GIST-1M).

Computes squared-L2 distances from each query to every base point
chunk-wise (keeps memory bounded), records top-k by distance, writes
in ParlayANN's `.bin` gt format:
    uint32 num_queries
    uint32 k
    uint32 neighbor_ids[num_queries][k]
    float32 distances[num_queries][k]
"""

import argparse
import numpy as np


def read_fbin(path):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(8), dtype=np.uint32)
        n, d = int(hdr[0]), int(hdr[1])
        arr = np.frombuffer(f.read(), dtype=np.float32).reshape(n, d)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-fbin", required=True)
    ap.add_argument("--query-fbin", required=True)
    ap.add_argument("--out", required=True, help="Output gt.bin path")
    ap.add_argument("-k", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=8192,
                    help="Base rows per distance-matrix chunk")
    args = ap.parse_args()

    base = read_fbin(args.base_fbin)
    query = read_fbin(args.query_fbin)
    nb, d = base.shape
    nq = query.shape[0]
    print(f"base: {nb} × {d}   query: {nq} × {d}   k={args.k}")

    # ‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b. Keeps a chunked matmul kernel fast.
    base_norm2 = (base * base).sum(axis=1)
    q_norm2 = (query * query).sum(axis=1)

    top_ids = np.zeros((nq, args.k), dtype=np.uint32)
    top_d = np.full((nq, args.k), np.inf, dtype=np.float32)

    for s in range(0, nb, args.chunk):
        e = min(s + args.chunk, nb)
        chunk = base[s:e]                                       # [c, d]
        # d² = q_norm² + b_norm² − 2 q·b  (shape [nq, c])
        cross = query @ chunk.T
        d2 = q_norm2[:, None] + base_norm2[None, s:e] - 2.0 * cross

        # Merge this chunk's distances against the running top-k.
        # `cur` concatenates old top-k with this chunk's distances, then
        # argpartition picks the new top-k.
        cur_d = np.concatenate([top_d, d2], axis=1)
        cur_ids = np.concatenate(
            [top_ids, np.broadcast_to(np.arange(s, e, dtype=np.uint32),
                                      (nq, e - s))],
            axis=1,
        )
        idx = np.argpartition(cur_d, args.k, axis=1)[:, : args.k]
        # Advanced-index back into cur_d and cur_ids
        rows = np.arange(nq)[:, None]
        top_d = cur_d[rows, idx]
        top_ids = cur_ids[rows, idx]
        if (s // args.chunk) % 4 == 0:
            print(f"  processed {e}/{nb}")

    # Sort each row's top-k by distance ascending.
    order = np.argsort(top_d, axis=1)
    top_d = np.take_along_axis(top_d, order, axis=1)
    top_ids = np.take_along_axis(top_ids, order, axis=1)

    with open(args.out, "wb") as f:
        f.write(np.array([nq, args.k], dtype=np.uint32).tobytes())
        f.write(top_ids.astype(np.uint32, copy=False).tobytes())
        f.write(top_d.astype(np.float32, copy=False).tobytes())
    print(f"wrote {args.out}: {nq} × {args.k}")


if __name__ == "__main__":
    main()
