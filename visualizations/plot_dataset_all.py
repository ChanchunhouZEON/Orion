#!/usr/bin/env python3
"""Per-dataset QPS-vs-Recall@10 plot — three engines on one figure:

  * **Orion** (ours)
  * **Microsoft DiskANN** (in-process via the `diskann` core crate)
  * **ParlayANN Vamana** (PA's published per-dataset recipe)

All three are built on the same Vamana topology (identical R / L<sub>build</sub>
/ α / num_passes, per the dataset's entry in
`benchmark/scripts/prepare_parlayann_data.sh`).

## Data source

Three series are read from a single JSON per dataset, produced by
`benchmark/scripts/sweep_orion_vs_diskann_vs_parlayann.sh` and parsed
by `benchmark/scripts/collect_sweep_medians.py`:

    visualizations/sweep_orion_vs_parlayann_<dataset>.json
        {
          "orion":    [[recall, qps], ...],
          "diskann":   [[recall, qps], ...],
          "parlayann": [[recall, qps], ...],
        }

The old separate `qps_recall_<ds>.json` (DiskANN-only baseline from the
thread-sweep harness) is retired — the unified sweep script produces
all three series at once, so this script reads only the unified file.

## Usage

    # Single dataset
    python3 visualizations/plot_dataset_all.py                  # default: sift
    DATASET=glove100 python3 visualizations/plot_dataset_all.py
    DATASET=glove100 METRIC=mips python3 visualizations/plot_dataset_all.py
        # — reads `sweep_orion_vs_parlayann_glove100_mips.json` if present

    # Batch — every public dataset except fashion-mnist (where orion
    # is dominated by PA at the dataset's tiny scale; figure suppressed
    # so the headline curves communicate the production regime cleanly).
    python3 visualizations/plot_dataset_all.py --all

## Palette

`chart_style.PALETTE_VIVID` — the GLM-vivid palette: sky-bright cyan for
Orion, rose-magenta for ParlayANN, amber for DiskANN. Tuned for
white background + log-scale axes, where overlapping mid-recall regions
need brighter hues to keep each line unambiguous.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from chart_style import PALETTE_VIVID, save_png_and_pdf, style_ax


# Datasets the script knows how to render. `fashion-mnist` is
# excluded from `--all` because at 60K vectors the orion cascade
# overhead becomes proportionally significant — PA's tighter beam
# wins, and a paper-grade comparison figure leads with the regime
# where the headline result lives.
HEADLINE_DATASETS = [
    "sift",
    "glove25",
    "glove100",
    "gist",
    "deep10m",
    "msmarco_bert_1M",
    "wiki_ada_1M",
]

DATASET_LABELS = {
    "sift":            "SIFT1M",
    "glove25":         "GloVe-25-angular",
    "glove100":        "GloVe-100-angular",
    "gist":            "GIST1M",
    "deep10m":         "Deep10M",
    "fashion-mnist":   "Fashion-MNIST",
    "msmarco_bert_1M": "MS-MARCO BERT 1M",
    "wiki_ada_1M":     "Wikipedia ada-002 1M",
}


VIS_DIR = Path(__file__).resolve().parent


def render_one(dataset: str, metric: str = "l2") -> Optional[str]:
    """Render the three-engine QPS-vs-recall plot for one dataset.
    Returns the output path on success, or `None` if the source JSON
    is missing / lacks required keys. Paths resolve relative to this
    script's directory so the entry point works from any cwd.
    """
    suffix = "_mips" if metric == "mips" else ""
    src = VIS_DIR / f"sweep_orion_vs_parlayann_{dataset}{suffix}.json"
    if not src.exists():
        print(f"[skip] {dataset}: no source JSON ({src})", file=sys.stderr)
        return None

    sweep = json.loads(src.read_text())
    required = {"orion", "parlayann"}
    if not required.issubset(sweep.keys()):
        print(
            f"[skip] {dataset}: JSON missing required keys (have {list(sweep)})",
            file=sys.stderr,
        )
        return None

    orion = sweep["orion"]
    parlay = sweep["parlayann"]
    # `diskann` is optional — older two-engine sweeps don't write it.
    # When absent, the plot degrades to 2 lines instead of failing.
    diskann = sweep.get("diskann")

    pretty = DATASET_LABELS.get(dataset, dataset)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    def plot(data, label, color, marker, *, lw=2.2, msize=6.5, zorder=2):
        rs = [p[0] for p in data]
        qs = [p[1] for p in data]
        ax.plot(
            rs,
            qs,
            marker=marker,
            linewidth=lw,
            markersize=msize,
            label=label,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=zorder,
        )

    # Draw order: DiskANN first (lowest zorder), then ParlayANN, then
    # Orion on top so the "ours" line never gets visually buried by
    # the others in overlap regions.
    if diskann:
        # Label is "Microsoft Vamana" — not "Microsoft DiskANN" —
        # because we run the in-memory Vamana index here (no SSD I/O
        # path). The "DiskANN" name historically refers to the
        # disk-resident variant; in-memory Vamana is the algorithm
        # both Microsoft and ParlayANN actually share, so labelling
        # the chart with the algorithm name keeps the apples-to-
        # apples framing honest.
        plot(
            diskann,
            "Microsoft Vamana",
            PALETTE_VIVID["diskann"],
            "^",
            lw=2.0,
            msize=6.0,
            zorder=2,
        )
    plot(
        parlay,
        "ParlayANN Vamana",
        PALETTE_VIVID["parlay"],
        "D",
        lw=2.0,
        msize=6.0,
        zorder=3,
    )
    plot(
        orion,
        "Orion (ours)",
        PALETTE_VIVID["orion"],
        "s",
        lw=2.8,
        msize=7.0,
        zorder=4,
    )

    # Annotation arrow on the "ours" line — keeps the figure
    # interpretable in monochrome / accessibility mode.
    if len(orion) >= 7:
        ar_x, ar_y = orion[6]
        ax.annotate(
            "ours",
            xy=(ar_x, ar_y),
            xytext=(ar_x - 0.035, ar_y * 2.1),
            color=PALETTE_VIVID["orion_d"],
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE_VIVID["orion_d"],
                lw=1.6,
            ),
        )

    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (queries / sec)", fontsize=11)
    ax.set_yscale("log")

    # Auto-fit x to the leftmost recall point across all available series
    # so PA's low-Q sweep (which can dip to R≈0.40) isn't clipped.
    series_for_xlim = [orion, parlay]
    if diskann:
        series_for_xlim.append(diskann)
    min_recall = min(min(p[0] for p in s) for s in series_for_xlim)
    ax.set_xlim(max(0.0, min_recall - 0.01), 1.0)

    ax.set_title(
        f"{pretty} — QPS vs Recall@10 (8 threads)",
        fontsize=12,
        color=PALETTE_VIVID["text"],
        pad=10,
    )
    ax.grid(True, which="both", alpha=0.35, color=PALETTE_VIVID["grid"])
    ax.legend(loc="lower left", frameon=True, fontsize=10, framealpha=0.92)
    style_ax(ax)

    out_suffix = "_mips" if metric == "mips" else ""
    out = str(VIS_DIR / f"qps_recall_{dataset}{out_suffix}_all3.png")
    plt.tight_layout()
    # Sibling PDF for paper use — fonts upscaled (see chart_style).
    png_path, pdf_path = save_png_and_pdf(fig, out)
    plt.close(fig)
    print(f"Saved: {png_path}  +  {pdf_path}")
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all",
        action="store_true",
        help=(
            "Render every dataset listed in HEADLINE_DATASETS "
            "(excludes fashion-mnist by design — see module docstring)."
        ),
    )
    ap.add_argument(
        "--dataset",
        default=os.environ.get("DATASET", "sift"),
        help="Dataset name (default: $DATASET or 'sift').",
    )
    ap.add_argument(
        "--metric",
        default=os.environ.get("METRIC", "l2").lower(),
        choices=["l2", "mips"],
        help="Reads `_mips` JSON variant when set to 'mips'.",
    )
    args = ap.parse_args()

    if args.all:
        rendered = 0
        for ds in HEADLINE_DATASETS:
            out = render_one(ds, metric="l2")
            if out:
                rendered += 1
        print(f"\nRendered {rendered}/{len(HEADLINE_DATASETS)} datasets")
    else:
        render_one(args.dataset, metric=args.metric)


if __name__ == "__main__":
    main()
