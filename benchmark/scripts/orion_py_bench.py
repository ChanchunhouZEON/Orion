#!/usr/bin/env python3
"""Measure FFI overhead of the Orion Python binding.

Loads the SAME cached index the Rust `benchmark` binary loads,
sweeps the same `search_list_size` ladder, runs the same batch
queries through `orion_py.OrionSift.search_batch`, and
reports (recall, QPS) per point. Compared against
`visualizations/sweep_orion_vs_parlayann_sift.json` (the Rust
direct numbers) the per-point QPS gap is the per-batch FFI
overhead — the same overhead USearch's Python binding adds to
USearch's QPS numbers we use elsewhere in the panel.

Usage:
  RAYON_NUM_THREADS=8 python orion_py_bench_bench.py
"""

import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Match the Rust sweep ladder verbatim (benchmark/configs/sweep.yaml,
# `search_list_sizes` line). 14 of the 17 entries — drop the tail
# (512/768/1024) since calibration drifts on extreme L and the
# comparison signal lives in the mid-band.
SEARCH_LIST_SIZES = [16, 20, 24, 32, 40, 48, 56, 64, 80, 100, 128, 160, 200, 256]
CACHE_PATH = "cache/orion_parlayann/sift_n1000000_r64_l128_a1_15_ex16_pct60.bin"
DATA_DIR = "data/sift"
REF_SWEEP = "visualizations/sweep_orion_vs_parlayann_sift.json"
OUT_PATH = "visualizations/orion_py_bench_sift.json"
K = 10
TRIALS = 3
WINDOW_SIZE = 8
CALIB_SAMPLE = 200


def read_fvecs(path):
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    record = 1 + dim
    raw = np.fromfile(path, dtype=np.float32)
    n = raw.size // record
    return raw[: n * record].reshape(n, record)[:, 1:].copy(), n, dim


def read_ivecs(path):
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    record = 1 + dim
    raw = np.fromfile(path, dtype=np.int32)
    n = raw.size // record
    return raw[: n * record].reshape(n, record)[:, 1:].copy()


def recall_at_k(ids, gt, k):
    n = min(len(ids), len(gt))
    tot = 0.0
    for i in range(n):
        gt_set = set(int(x) for x in gt[i][:k])
        hits = sum(1 for r in ids[i][:k] if int(r) in gt_set)
        tot += hits / k
    return tot / n


def main():
    import orion_py as sdp

    print(f"rayon threads: {sdp.OrionSift.rayon_threads()}")
    print(f"loading SIFT base + queries + GT...")
    base, n, dim = read_fvecs(f"{DATA_DIR}/sift_base.fvecs")
    queries, nq, _ = read_fvecs(f"{DATA_DIR}/sift_query.fvecs")
    gt = read_ivecs(f"{DATA_DIR}/sift_groundtruth.ivecs")
    print(f"  base={n}×{dim}  queries={nq}  gt={gt.shape}")

    t0 = time.time()
    idx = sdp.OrionSift.load_cache(CACHE_PATH, base)
    print(f"loaded cache in {time.time() - t0:.2f}s")

    # Calibrate ONCE at the canonical CALIB_L=48, mirroring the Rust
    # benchmark binary (main.rs:534 — `recalibrate(queries, 200)` after
    # `set_search_list_size(48)`, before the sweep loop). The sweep
    # then just changes L without re-calibrating; the L=48 params
    # are reused for every SLS. See also `orion`'s `CALIB_L=48`.
    t0 = time.time()
    eps, ee = idx.calibrate(queries[:CALIB_SAMPLE], 48, WINDOW_SIZE)
    print(f"calibrated at L=48 in {time.time() - t0:.2f}s: epsilon={eps:.4f} early_exit={ee}")

    # Warmup pass at the smallest L — matches main.rs:1446-1454 shape.
    print("warmup...")
    _ = idx.search_batch(queries, K, SEARCH_LIST_SIZES[0])

    # Match the Rust binary's measurement methodology exactly:
    # `flush_cache()` (40 MB random-shuffle, ~10 ms) before every
    # timed search_batch so each trial starts from a cold L1/L2/L3.
    # See `benchmark/src/bin/orion.rs` (TRIALS=1 + per-trial
    # flush). Without this, trial 2+ of any L benefits from the
    # working set the prior trial just paged in — a warm-cache
    # advantage Python had over the Rust JSON's cold-cache numbers.
    print("\nsweeping search_list_size (params from single L=48 calibration, "
          "flush_cache before each trial):")
    points = []
    for sls in SEARCH_LIST_SIZES:
        qps_samples = []
        ids = None
        for _ in range(TRIALS):
            sdp.flush_cache()
            t = time.time()
            ids = idx.search_batch(queries, K, sls)
            wall = time.time() - t
            qps_samples.append(nq / wall)
        qps = sorted(qps_samples)[TRIALS // 2]
        r = recall_at_k(ids.tolist(), gt.tolist(), K)
        points.append([round(r, 4), round(qps)])
        print(f"  L={sls:>5}  R@{K}={r:.4f}  QPS={qps:>9,.0f}")

    out = {
        "dataset": "sift",
        "binding": "pyo3-numpy-batch",
        "search_list_sizes": SEARCH_LIST_SIZES,
        "calibration": "once at CALIB_L=48 (matches Rust main.rs:534)",
        "cache_flush_per_trial": True,
        "epsilon": eps,
        "early_exit_limit": ee,
        "window_size": WINDOW_SIZE,
        "threads": sdp.OrionSift.rayon_threads(),
        "k": K,
        "trials": TRIALS,
        "orion_py": points,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_PATH}")

    # ── FFI overhead vs Rust direct ────────────────────────────────
    if not Path(REF_SWEEP).exists():
        print(f"[skip] reference sweep file missing: {REF_SWEEP}")
        return

    ref = json.loads(Path(REF_SWEEP).read_text())
    rust_pts = ref.get("orion", [])
    if not rust_pts:
        print("[skip] no orion series in reference sweep")
        return

    # Compare iso-recall: for each Python point, find the Rust QPS at
    # the closest recall and compute ratio.
    rust_sorted = sorted(rust_pts, key=lambda p: p[0])
    print(f"\n{'L':>5}  {'pyR@10':>7}  {'pyQPS':>9}  {'rustR@10':>9}  {'rustQPS':>9}  {'gap%':>6}")
    gaps = []
    for sls, (r_py, q_py) in zip(SEARCH_LIST_SIZES, points):
        # nearest-recall Rust point
        nearest = min(rust_sorted, key=lambda p: abs(p[0] - r_py))
        r_ru, q_ru = nearest
        gap = (q_ru - q_py) / q_ru * 100 if q_ru else 0.0
        gaps.append(gap)
        print(f"  {sls:>5}  {r_py:>7.4f}  {q_py:>9,.0f}  {r_ru:>9.4f}  {q_ru:>9,.0f}  {gap:>5.1f}%")

    avg = sum(gaps) / len(gaps)
    print(f"\nMean FFI overhead (Rust − Python) / Rust: {avg:+.1f}%")
    if avg > 10:
        print("  ⚠️  >10% — investigate (numpy copy? FFI per-call? thread mismatch?)")
    elif avg > 5:
        print("  ⚠️  >5% — moderate, worth a footnote")
    elif avg > -5:
        print("  ✓  within ±5% — Python binding overhead is negligible at this batch size")
    else:
        print("  ⚠️  Python is FASTER than Rust direct — calibration drift, not FFI")


if __name__ == "__main__":
    sys.exit(main() or 0)
