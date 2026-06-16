#!/usr/bin/env python3
"""Convert BigANN-style `.fbin` (Yandex/Microsoft) → `.fvecs` (Yael format).

Used to bring Deep10M (and similar BigANN-track datasets) into the
fvecs format that the rust-side `staged_diskann` / `staged_sweep` bins
expect.

## File formats

`.fbin` (BigANN / Yandex):
    [u32 num_vectors]
    [u32 dim]
    [num_vectors × dim × f32]

`.fvecs` (Yael / standard):
    Per vector:
        [u32 dim]
        [dim × f32]

For groundtruth, the BigANN GT format `.bin` is:
    [u32 num_queries]
    [u32 K]
    [num_queries × K × i32  ids]
    [num_queries × K × f32  distances]   ← optional; some dumps skip

The `.ivecs` output keeps only the IDs (which is what `mean_recall`
reads). Distances are dropped on conversion.

## Usage

    # Base + query (vector .fbin → .fvecs)
    python3 data/convert_fbin.py --vec base.fbin       data/deep10m/deep10m_base.fvecs
    python3 data/convert_fbin.py --vec query.fbin      data/deep10m/deep10m_query.fvecs

    # Groundtruth (BigANN .bin → .ivecs)
    python3 data/convert_fbin.py --gt  groundtruth.bin data/deep10m/deep10m_groundtruth.ivecs
"""

import argparse
import os
import struct
import sys

import numpy as np


def convert_vec_fbin_to_fvecs(
    fbin_path: str, fvecs_path: str, pad_to_dim=None
) -> None:
    """Vector .fbin → .fvecs. Streams in chunks so 10M × 96 = 3.8 GB
    files don't have to fit in RAM.

    `pad_to_dim`: when set, zero-pad each vector to this dim. Used to
    match the rust-side const-generic monomorphisations (32/100/128/960)
    when the source dim is non-supported (e.g. Deep10M's D=96 → 100).
    Zero padding contributes 0 to both L2 and IP distances, so the
    ranking is bit-identical to the native-D path.
    """
    os.makedirs(os.path.dirname(fvecs_path) or ".", exist_ok=True)
    with open(fbin_path, "rb") as fi, open(fvecs_path, "wb") as fo:
        header = fi.read(8)
        if len(header) != 8:
            raise ValueError(f"{fbin_path}: truncated header")
        n, dim = struct.unpack("<II", header)
        out_dim = pad_to_dim if pad_to_dim is not None else dim
        if out_dim < dim:
            raise ValueError(f"pad_to_dim={out_dim} < source dim={dim}")
        print(
            f"  {fbin_path}: {n} vectors × {dim} dims"
            + (f"  →  pad to {out_dim}" if out_dim != dim else "")
        )

        # 64K vectors per chunk → 64K × ~100 × 4 ≈ 25 MB working set.
        CHUNK = 65_536
        rec_bytes = dim * 4
        for c0 in range(0, n, CHUNK):
            c = min(CHUNK, n - c0)
            buf = fi.read(c * rec_bytes)
            if len(buf) != c * rec_bytes:
                raise ValueError(f"{fbin_path}: truncated at vector {c0}")
            vecs = np.frombuffer(buf, dtype=np.float32).reshape(c, dim)
            # Per-vector record: [u32 out_dim][out_dim × f32 values].
            out = np.zeros((c, out_dim + 1), dtype=np.float32)
            out.view(np.uint32)[:, 0] = out_dim
            out[:, 1 : dim + 1] = vecs
            # out[:, dim + 1 : out_dim + 1] stays zero (the padding).
            fo.write(out.tobytes())
    print(f"  → {fvecs_path}")


def convert_gt_fbin_to_ivecs(gt_path: str, ivecs_path: str) -> None:
    """BigANN groundtruth `.bin` → `.ivecs` (IDs only, distances dropped)."""
    os.makedirs(os.path.dirname(ivecs_path) or ".", exist_ok=True)
    with open(gt_path, "rb") as fi:
        header = fi.read(8)
        if len(header) != 8:
            raise ValueError(f"{gt_path}: truncated header")
        nq, k = struct.unpack("<II", header)
        print(f"  {gt_path}: {nq} queries × top-{k}")
        ids = np.frombuffer(fi.read(nq * k * 4), dtype=np.int32).reshape(nq, k)

    with open(ivecs_path, "wb") as fo:
        out = np.empty((nq, k + 1), dtype=np.int32)
        out[:, 0] = k
        out[:, 1:] = ids
        fo.write(out.tobytes())
    print(f"  → {ivecs_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="BigANN .fbin → Yael .fvecs converter."
    )
    p.add_argument(
        "--vec",
        action="store_true",
        help="Convert a vector .fbin (default mode).",
    )
    p.add_argument(
        "--gt",
        action="store_true",
        help="Convert a BigANN groundtruth .bin → .ivecs (IDs only).",
    )
    p.add_argument("input", help="Source .fbin / .bin file.")
    p.add_argument("output", help="Destination .fvecs / .ivecs path.")
    p.add_argument(
        "--pad-to-dim",
        type=int,
        default=None,
        help="Zero-pad vectors to this dim (e.g. 96→100 for Deep10M).",
    )
    args = p.parse_args()

    if args.gt:
        convert_gt_fbin_to_ivecs(args.input, args.output)
    else:
        convert_vec_fbin_to_fvecs(args.input, args.output, pad_to_dim=args.pad_to_dim)


if __name__ == "__main__":
    main()
