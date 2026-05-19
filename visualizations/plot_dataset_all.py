#!/usr/bin/env python3
"""Per-dataset QPS vs Recall@10 plot — 3 implementations on one figure:
  - DiskANN (built at the dataset's staged params — apples-to-apples)
  - StagedDiskANN (our current best)
  - ParlayANN Vamana

Data sources:
  * visualizations/qps_recall_<dataset>.json
      DiskANN baseline now uses the same R/L/α as StagedDiskANN
      (see benchmark/configs/sweep.yaml `datasets.<name>.staged` block;
      override with a per-dataset `diskann:` block to opt out).
      The legacy `diskann_matched` JSON key is gone — there's only one
      DiskANN line now since the baseline already matches staged params.
  * visualizations/sweep_staged_vs_parlayann_<dataset>.json
      (StagedDiskANN + ParlayANN 3-run medians, produced by
      `benchmark/scripts/sweep_staged_vs_parlayann.sh`)

Usage:
  python3 visualizations/plot_dataset_all.py                  # default: sift (L2)
  DATASET=gist python3 visualizations/plot_dataset_all.py     # any dataset (L2)
  DATASET=glove100 METRIC=mips python3 visualizations/plot_dataset_all.py
      # — reads `sweep_staged_vs_parlayann_<ds>_mips.json` which holds the
      #   angular-aligned (MIPS + normalized) numbers for both sides.
"""

import json
import os
import matplotlib.pyplot as plt
from chart_style import PALETTE, style_ax


DATASET_LABELS = {
    "sift": "SIFT1M",
    "glove25": "GloVe-25-angular",
    "glove100": "GloVe-100-angular",
    "gist": "GIST-100k",
}


def main():
    dataset = os.environ.get("DATASET", "sift")
    metric = os.environ.get("METRIC", "l2").lower()
    pretty = DATASET_LABELS.get(dataset, dataset)

    # ── DiskANN baseline (built at staged params per sweep.yaml) ──
    d = json.load(open(f"visualizations/qps_recall_{dataset}.json"))
    diskann = d["diskann"]

    # ── StagedDiskANN + ParlayANN: 3-run medians from the sweep script ──
    suffix = "_mips" if metric == "mips" else ""
    sweep = json.load(
        # open(f"visualizations/sweep_staged_vs_parlayann_{dataset}{suffix}.json")
        open(f"visualizations/sweep_staged_vs_parlayann_{dataset}.json")
    )
    staged = sweep["staged"]
    parlay = sweep["parlayann"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    def plot(data, label, color, marker, lw=2.0):
        rs = [p[0] for p in data]
        qs = [p[1] for p in data]
        ax.plot(rs, qs, marker=marker, linewidth=lw, markersize=6,
                label=label, color=color)

    # Staged line uses the canonical "ours" color + square marker used in
    # `plot_qps_recall_bands.py` so this figure is visually consistent
    # with the rest of the project's comparison plots.
    OURS_COLOR = "#598392"

    # All three lines are now built at the same R/α/L (per sweep.yaml's
    # `staged:` block per dataset), so the labels carry just the
    # algorithm name — the only difference between the lines is the
    # search-time algorithm itself.
    plot(diskann, "DiskANN",                PALETTE['grey'], "^")
    plot(parlay,  "ParlayANN Vamana",       PALETTE['red'],  "D")
    plot(staged,  "StagedDiskANN (ours)",   OURS_COLOR,      "s", lw=2.6)

    # Annotate "ours" with an arrow at a mid-recall point so the
    # highlighted line is unambiguous even in B&W prints.
    if len(staged) >= 7:
        ar_x, ar_y = staged[6]  # L≈56, R≈0.988 region
        ax.annotate(
            "ours",
            xy=(ar_x, ar_y),
            xytext=(ar_x - 0.035, ar_y * 2.1),
            color=OURS_COLOR,
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=OURS_COLOR, lw=1.5),
        )

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (queries / sec)")
    ax.set_yscale("log")
    # Auto-fit x-axis to the leftmost point across all three curves
    # (ParlayANN typically starts at the lowest recall — Q=10 lands
    # near R=0.43 on SIFT — so a hard-coded `xlim(0.88, 1.0)` would
    # silently lop off its full low-recall sweep). Pad 0.01 on the
    # left for breathing room; pin the right at 1.0 since that's
    # the natural recall ceiling.
    min_recall = min(
        min(p[0] for p in diskann),
        min(p[0] for p in parlay),
        min(p[0] for p in staged),
    )
    ax.set_xlim(max(0.0, min_recall - 0.01), 1.0)
    ax.set_title(f"{pretty} — QPS vs Recall@10 (8 threads)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left")
    style_ax(ax)

    out_suffix = "_mips" if metric == "mips" else ""
    out = f"visualizations/qps_recall_{dataset}{out_suffix}_all4.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
