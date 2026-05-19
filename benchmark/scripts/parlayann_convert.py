#!/usr/bin/env python3
"""Convert SIFT fvecs/ivecs to ParlayANN's fbin format.

ParlayANN .fbin layout (from utils/point_range.h:86-100):
    uint32 num_points
    uint32 dim
    float32 data[num_points][dim]

ParlayANN groundtruth layout (from utils/types.h:54-72):
    uint32 num_queries
    uint32 k
    uint32 neighbor_ids[num_queries][k]
    float32 distances[num_queries][k]  (unused for recall; fill with zeros)

fvecs source layout (per vector):
    uint32 dim
    float32 data[dim]

ivecs source layout (per query):
    uint32 k
    uint32 neighbor_ids[k]
"""

import argparse
import os
import numpy as np


def read_fvecs(path):
    with open(path, "rb") as f:
        buf = f.read()
    dim = np.frombuffer(buf, dtype=np.uint32, count=1)[0]
    vec_size = 4 + 4 * dim
    n = len(buf) // vec_size
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, vec_size)
    return arr[:, 4:].view(np.float32).reshape(n, dim), int(dim)


def read_ivecs(path):
    with open(path, "rb") as f:
        buf = f.read()
    k = np.frombuffer(buf, dtype=np.uint32, count=1)[0]
    vec_size = 4 + 4 * k
    n = len(buf) // vec_size
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, vec_size)
    return arr[:, 4:].view(np.uint32).reshape(n, k), int(k)


def write_fbin(path, arr):
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(np.array([n, d], dtype=np.uint32).tobytes())
        f.write(arr.astype(np.float32, copy=False).tobytes())
    return n, d


def write_gt_bin(path, neighbors_u32, k):
    n = neighbors_u32.shape[0]
    dists = np.zeros((n, k), dtype=np.float32)
    with open(path, "wb") as f:
        f.write(np.array([n, k], dtype=np.uint32).tobytes())
        f.write(neighbors_u32.astype(np.uint32, copy=False).tobytes())
        f.write(dists.tobytes())
    return n, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-fvecs", required=True)
    ap.add_argument("--query-fvecs", required=True)
    ap.add_argument("--gt-ivecs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-base-points", type=int, default=0,
                    help="Truncate base to this many points (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base, bdim = read_fvecs(args.base_fvecs)
    query, qdim = read_fvecs(args.query_fvecs)
    gt, gtk = read_ivecs(args.gt_ivecs)

    if args.max_base_points and args.max_base_points < base.shape[0]:
        base = base[: args.max_base_points]

    assert bdim == qdim, f"base dim {bdim} != query dim {qdim}"

    base_out = os.path.join(args.out_dir, "base.fbin")
    query_out = os.path.join(args.out_dir, "query.fbin")
    gt_out = os.path.join(args.out_dir, "gt.bin")

    n, d = write_fbin(base_out, base)
    print(f"base:  {n} pts × {d} dim → {base_out}")
    n, d = write_fbin(query_out, query)
    print(f"query: {n} pts × {d} dim → {query_out}")
    n, k = write_gt_bin(gt_out, gt, gtk)
    print(f"gt:    {n} queries × {k} NN → {gt_out} (distances zero-filled)")


if __name__ == "__main__":
    main()
