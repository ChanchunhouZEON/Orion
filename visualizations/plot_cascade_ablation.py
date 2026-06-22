#!/usr/bin/env python3
"""Cascade-stage ablation panel: QPS vs Recall@10 for four cascade
variants on the same built graph.

Reads `visualizations/cascade_ablation_<dataset>.json` (produced by
`benchmark --algorithms cascade-ablation`) and renders one panel with
four lines:

  * **full**           — per-dataset default cascade (baseline)
  * **no-prefilter**   — prefilter forced to None
  * **no-rerank**      — rerank forced to None (top-k taken from
                          admission PQ ordering directly)
  * **admission-only** — both auxiliary stages off

Each line is a parameter sweep over `search_list_size`. Read the
panel by comparing two lines at iso-recall:

  * `full` vs `no-prefilter` at the same recall → prefilter QPS contribution
  * `full` vs `no-rerank`    at the same L      → rerank QPS cost + recall loss
  * `full` vs `admission-only`                  → combined contribution

The plot mirrors `plot_baseline_comparison.py`'s log-QPS-vs-recall
treatment so the staged_diskann figure family stays visually
consistent across the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from chart_style import PALETTE, PALETTE_VIVID, save_png_and_pdf, style_ax

VIS_DIR = Path(__file__).resolve().parent

# (json_key, label, color, marker, linewidth, marker_size, zorder)
# `full` gets the boldest line + top zorder — it's the baseline every
# other curve is read against.
SERIES = [
    ("admission_only", "Admission only (no prefilter, no rerank)",
        PALETTE["grey"],  "v", 1.7, 6.0, 2),
    ("no_rerank",      "No rerank (prefilter + admission)",
        PALETTE["purple"], "X", 1.8, 6.5, 3),
    ("no_prefilter",   "No prefilter (admission + rerank)",
        PALETTE["green"],  "D", 1.8, 6.5, 3),
    ("full",           "Full cascade (default)",
        PALETTE_VIVID["staged"], "s", 2.6, 7.5, 5),
]

HEADLINE_DATASETS = [
    ("sift",     "SIFT 1M"),
    ("glove25",  "GloVe-25 (cosine)"),
    ("glove100", "GloVe-100 (cosine)"),
    ("gist",     "GIST 1M"),
]


def render_one(dataset: str, title_label: str, *, out_path=None) -> str | None:
    json_path = VIS_DIR / f"cascade_ablation_{dataset}.json"
    if not json_path.exists():
        print(f"[skip] {dataset}: {json_path} not found")
        return None

    payload = json.loads(json_path.read_text())
    cascade = payload.get("default_cascade", {})

    fig, ax = plt.subplots(figsize=(9, 5.8))

    for key, label, color, marker, lw, msize, zorder in SERIES:
        if key not in payload:
            continue
        pts = payload[key]
        if not pts:
            continue
        # Sort by recall so the polyline doesn't backtrack.
        pts = sorted(pts, key=lambda p: p[0])
        rs = [p[0] for p in pts]
        qs = [p[1] for p in pts]
        ax.plot(
            rs, qs,
            marker=marker, linewidth=lw, markersize=msize,
            color=color, label=label,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=zorder,
        )

    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (queries / sec)", fontsize=11)
    ax.set_yscale("log")

    # Clip the x-axis to the rendered data so the low-recall
    # admission-only tail doesn't dominate the panel.
    all_recalls = [r for k, *_ in SERIES if k in payload
                       for r, _ in payload[k]]
    if all_recalls:
        ax.set_xlim(max(0.0, min(all_recalls) - 0.02), 1.0)

    title = f"{title_label} — Cascade-stage ablation (8 threads, k=10)"
    if cascade:
        # Sub-title note recording the per-dataset default cascade so
        # the reader can map "default" back to a concrete triple
        # without cross-referencing sweep.yaml.
        cascade_line = "default: {} → {} → {}".format(
            cascade.get("prefilter", "?"),
            cascade.get("admission", "?"),
            cascade.get("rerank", "?"),
        )
        title = f"{title}\n{cascade_line}"

    ax.set_title(title, fontsize=12, color=PALETTE_VIVID["text"], pad=10)
    ax.grid(True, which="both", alpha=0.35, color=PALETTE_VIVID["grid"])
    # fontsize=11 → 16.5pt under save_png_and_pdf's 1.5× scale (matches
    # the ablation_study figure's legend treatment).
    ax.legend(loc="lower left", frameon=True, fontsize=11, framealpha=0.92)
    style_ax(ax)

    out = str(out_path or (VIS_DIR / f"cascade_ablation_{dataset}.png"))
    plt.tight_layout()
    png_path, pdf_path = save_png_and_pdf(fig, out)
    plt.close(fig)
    print(f"Saved {png_path}  +  {pdf_path}")
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--out", default=None, help="Override output PNG path.")
    ap.add_argument(
        "--all", action="store_true",
        help="Render every dataset in HEADLINE_DATASETS that has a JSON.",
    )
    args = ap.parse_args()

    if args.all:
        rendered = 0
        for ds, label in HEADLINE_DATASETS:
            if render_one(ds, label):
                rendered += 1
        print(f"\nRendered {rendered}/{len(HEADLINE_DATASETS)} datasets.")
        return

    label = dict(HEADLINE_DATASETS).get(args.dataset, args.dataset)
    if render_one(args.dataset, label, out_path=args.out) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
