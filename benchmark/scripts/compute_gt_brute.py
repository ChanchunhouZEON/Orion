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
from benchmark_support import exact_top_k


def read_fbin(path):
    """Map validated little-endian fbin input without copying the whole base."""
    from pathlib import Path
    import struct
    with open(path, "rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"{path}: missing fbin header")
    n, d = struct.unpack("<II", header)
    if not n or not d or Path(path).stat().st_size != 8 + n * d * 4:
        raise ValueError(f"{path}: invalid fbin shape or payload length")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(n, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-fbin", required=True)
    ap.add_argument("--query-fbin", required=True)
    ap.add_argument("--out", required=True, help="Output gt.bin path")
    ap.add_argument("-k", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=8192,
                    help="Base rows per distance-matrix chunk")
    ap.add_argument("--metric", choices=("l2", "ip", "cos"), default="l2")
    args = ap.parse_args()

    base = read_fbin(args.base_fbin)
    query = read_fbin(args.query_fbin)
    nb, d = base.shape
    nq = query.shape[0]
    print(f"base: {nb} × {d}   query: {nq} × {d}   k={args.k}")

    top_ids, top_d = exact_top_k(base, query, args.k, args.metric, args.chunk)

    with open(args.out, "wb") as f:
        f.write(np.array([nq, args.k], dtype="<u4").tobytes())
        f.write(top_ids.astype("<u4", copy=False).tobytes())
        f.write(top_d.astype("<f4", copy=False).tobytes())
    print(f"wrote {args.out}: {nq} × {args.k}")


if __name__ == "__main__":
    main()
