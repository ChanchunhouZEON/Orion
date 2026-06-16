#!/usr/bin/env python3
"""SIFT 1M multi-algorithm QPS-vs-Recall@10 comparison at 8 threads.

Reads:
  * `visualizations/baseline_<ds>.json` — HNSW, FAISS IVF-Flat,
    FAISS IVF-PQ, Annoy (from `baseline_comparison.py` +
    `additional_baselines.py`).
  * `visualizations/sweep_staged_vs_parlayann_<ds>.json` — Staged,
    Microsoft Vamana (the `diskann` core crate, in-memory), ParlayANN
    Vamana (from `sweep_staged_vs_diskann_vs_parlayann.sh`).

Renders one figure: log-QPS vs Recall@10 with all available series.
StagedDiskANN drawn last so its line stays visually on top.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from chart_style import PALETTE, PALETTE_VIVID, style_ax


VIS_DIR = Path(__file__).resolve().parent

# Each entry: (key, label, color, marker, linewidth, marker_size, zorder).
# StagedDiskANN gets the boldest line + top zorder; the bit-quantised
# tree-style indices (Annoy) and product-quantised IVF (FAISS IVF-PQ)
# sit at lower zorder because they overlap each other in the mid-recall
# band and matter less for the headline narrative.
SERIES = [
    ("annoy",          "Annoy",                "#94A3B8", "v", 1.7, 6.0, 2),
    ("faiss_ivf_pq",   "FAISS IVF-PQ",         "#A78BFA", "P", 1.7, 6.5, 2),
    ("faiss_ivf_flat", "FAISS IVF-Flat",       "#FBBF24", "X", 1.8, 6.5, 2),
    ("hnsw",           "HNSW (hnswlib)",       "#10B981", "D", 1.9, 6.5, 3),
    ("diskann",        "Microsoft Vamana",     PALETTE_VIVID["diskann"], "^", 1.9, 6.0, 3),
    ("parlayann",      "ParlayANN Vamana",     PALETTE_VIVID["parlay"],  "o", 2.0, 6.5, 3),
    ("staged",         "StagedDiskANN (ours)", PALETTE_VIVID["staged"],  "s", 2.8, 7.5, 5),
]


def load(ds):
    """Merge baseline_<ds>.json with sweep_staged_vs_parlayann_<ds>.json
    into a single key→[(recall, qps), ...] dict. Sweep-file series take
    precedence (they're the head-to-head 3-engine source of truth)."""
    merged = {}

    baseline_path = VIS_DIR / f"baseline_{ds}.json"
    sweep_path    = VIS_DIR / f"sweep_staged_vs_parlayann_{ds}.json"

    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        for key in ("hnsw", "faiss_ivf_flat", "faiss_ivf_pq", "annoy",
                    "diskann", "staged", "parlayann"):
            if key in baseline:
                merged[key] = baseline[key]

    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        for key in ("staged", "diskann", "parlayann"):
            if key in sweep:
                merged[key] = sweep[key]

    return merged


def subsample_series(points, target_count=14):
    """Subsample a sorted-by-recall point list down to ~`target_count`
    representative points while preserving the endpoints. Used for the
    Staged curve, which sweeps 35 search-list sizes — far more density
    than the other series and visually noisy at thumbnail scale."""
    n = len(points)
    if n <= target_count:
        return list(points)
    # Evenly-spaced indices including first and last.
    import numpy as _np
    idx = _np.linspace(0, n - 1, target_count).round().astype(int).tolist()
    seen = []
    out = []
    for i in idx:
        if i not in seen:
            seen.append(i)
            out.append(points[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--out", default=None,
                    help="Override output PNG path.")
    args = ap.parse_args()

    series_data = load(args.dataset)
    if not series_data:
        print(f"No data for dataset '{args.dataset}' — looked for "
              f"baseline_{args.dataset}.json and "
              f"sweep_staged_vs_parlayann_{args.dataset}.json")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 5.8))

    staged_drawn = None  # remember the drawn Staged polyline for the arrow annotation
    for key, label, color, marker, lw, msize, zorder in SERIES:
        if key not in series_data:
            continue
        pts = series_data[key]
        if not pts:
            continue
        # Sort by recall so the polyline doesn't backtrack on the x-axis
        # when an algorithm's parameter sweep crosses itself.
        pts = sorted(pts, key=lambda p: p[0])
        # Staged has ~35 sweep points — far denser than the other series.
        # Subsample to ~14 evenly-spaced points so markers don't visually
        # cluster into a band at low / mid recall.
        if key == "staged":
            pts = subsample_series(pts, target_count=14)
            staged_drawn = pts
        rs = [p[0] for p in pts]
        qs = [p[1] for p in pts]
        ax.plot(
            rs, qs,
            marker=marker, linewidth=lw, markersize=msize,
            color=color, label=label,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=zorder,
        )

    # ── "ours" arrow annotation on the Staged line — mirrors the
    # treatment in `plot_dataset_all.py` so the figure communicates
    # which curve is ours even in monochrome / accessibility mode. ──
    if staged_drawn and len(staged_drawn) >= 5:
        # Anchor the arrow at a mid-band point (not too low, not too
        # high) where the Staged curve sits well above HNSW / Vamana
        # — there's headroom for the label without overlapping other
        # lines.
        anchor_idx = min(4, len(staged_drawn) - 1)
        ar_x, ar_y = staged_drawn[anchor_idx]
        ax.annotate(
            "ours",
            xy=(ar_x, ar_y),
            xytext=(ar_x - 0.05, ar_y * 2.1),
            color=PALETTE_VIVID["staged_d"],
            fontsize=12, fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE_VIVID["staged_d"],
                lw=1.7,
            ),
            zorder=6,
        )

    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (queries / sec)", fontsize=11)
    ax.set_yscale("log")

    # Recall x-axis: clip to the lowest recall point present across all
    # rendered series so Annoy / IVF-PQ low-recall tails aren't cut off.
    all_recalls = [r for series in series_data.values() for r, _ in series]
    if all_recalls:
        ax.set_xlim(max(0.0, min(all_recalls) - 0.02), 1.0)

    title = f"SIFT 1M — QPS vs Recall@10 (8 threads, k=10)"
    ax.set_title(title, fontsize=12, color=PALETTE_VIVID["text"], pad=10)
    ax.grid(True, which="both", alpha=0.35, color=PALETTE_VIVID["grid"])
    ax.legend(loc="lower left", frameon=True, fontsize=10, framealpha=0.92)
    style_ax(ax)

    out = args.out or (VIS_DIR / f"baseline_panel_{args.dataset}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
