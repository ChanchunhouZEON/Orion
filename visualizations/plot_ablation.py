#!/usr/bin/env python3
"""Convergence-side ablation panel: QPS vs Recall@10 for the four
convergence variants on the per-dataset default cascade.

Reads `visualizations/ablation_<dataset>.json` (produced by
`benchmark --algorithms ablation`) and renders one panel with four
lines:

  * **origin**         — vanilla DiskANN Vamana (params from
                         sweep.yaml's `defaults.diskann` block, with
                         per-dataset overrides applied)
  * **no-extra**       — default cascade on a `max_extra = 0` graph
                         (extras zone gone — rerank visits local only)
  * **no-early-stop**  — default cascade + convergence on, `ee = MAX`
                         (beam exhausts the PQ, no early termination)
  * **full**           — default cascade + convergence + early-exit
                         + extras enabled (the production path)

Companion to `plot_cascade_ablation.py`. That panel ablates the
cascade *tiers* (prefilter / admission / rerank); this one ablates
the convergence-side machinery (early-stop, extras, vs the
pre-cascade DiskANN baseline) on top of the same default cascade.
The visual style mirrors `plot_cascade_ablation.py` so the two
panels read side-by-side in the README.
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
# other curve is read against. `origin` here is **cascade-only** — the
# same cascade as the other three variants with both convergence-side
# knobs disabled (no extras + no early-stop). Forms the bare reference
# point of a 2×2 factorial on early-stop × extras. NOT the pre-cascade
# Microsoft Vamana — that lives in the 3-engine sweep panel.
SERIES = [
    ("origin",        "Origin (cascade-only: no extras, no early-stop)",
        PALETTE["grey"],   "v", 1.7, 6.0, 2),
    ("no_extra",      "No extras (post-convergence: local+remote, not local+extra)",
        PALETTE["green"],  "D", 1.8, 6.5, 3),
    ("no_early_stop", "No early-stop (ee = MAX, extras on)",
        PALETTE["purple"], "X", 1.8, 6.5, 3),
    ("full",          "Full (default cascade + early-stop + extras)",
        PALETTE_VIVID["staged"], "s", 2.6, 7.5, 5),
]

HEADLINE_DATASETS = [
    ("sift",            "SIFT 1M"),
    ("glove25",         "GloVe-25 (cosine)"),
    ("glove100",        "GloVe-100 (cosine)"),
    ("gist",            "GIST 1M"),
    ("msmarco_bert_1M", "MSMARCO BERT 1M"),
]


def render_one(dataset: str, title_label: str, *, out_path=None) -> str | None:
    json_path = VIS_DIR / f"ablation_{dataset}.json"
    if not json_path.exists():
        print(f"[skip] {dataset}: {json_path} not found")
        return None

    payload = json.loads(json_path.read_text())
    cascade = payload.get("default_cascade", {})
    # `origin_params` is a legacy field from the Microsoft-Vamana-as-origin
    # design; recent runs omit it. The plot still tolerates older JSONs
    # but no longer relies on it.

    fig, ax = plt.subplots(figsize=(9, 5.8))

    for key, label, color, marker, lw, msize, zorder in SERIES:
        if key not in payload:
            continue
        pts = payload[key]
        if not pts:
            continue
        # Sort by recall so the polyline doesn't backtrack on any L
        # where two variants happen to land at the same QPS but
        # different recalls.
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

    # Clip the x-axis to the rendered data so the low-recall tail
    # (origin/no-extra at L=16) doesn't dominate the panel.
    all_recalls = [r for k, *_ in SERIES if k in payload
                       for r, _ in payload[k]]
    if all_recalls:
        ax.set_xlim(max(0.0, min(all_recalls) - 0.02), 1.0)

    title = f"{title_label} — Convergence-side ablation (8 threads, k=10)"
    sub_lines: list[str] = []
    if cascade:
        # Sub-title note recording the per-dataset default cascade — the
        # ablation is performed ON TOP of this cascade, so the reader
        # can map "full" back to the concrete cascade triple.
        sub_lines.append("default cascade: {} → {} → {}".format(
            cascade.get("prefilter", "?"),
            cascade.get("admission", "?"),
            cascade.get("rerank", "?"),
        ))
    # No origin-params subtitle: with origin now being cascade-only
    # (same code path as the other three variants), there's nothing
    # variant-specific to declare in the title — every variant is
    # a `search_compose` call with the cascade shown above and the
    # two knobs noted in the legend.
    if sub_lines:
        title = f"{title}\n" + "  |  ".join(sub_lines)

    ax.set_title(title, fontsize=12, color=PALETTE_VIVID["text"], pad=10)
    ax.grid(True, which="both", alpha=0.35, color=PALETTE_VIVID["grid"])
    # fontsize=11 → 16.5pt under save_png_and_pdf's 1.5× scale (matches
    # plot_cascade_ablation.py's legend treatment so the two panels read
    # at the same visual weight).
    ax.legend(loc="lower left", frameon=True, fontsize=11, framealpha=0.92)
    style_ax(ax)

    out = str(out_path or (VIS_DIR / f"ablation_{dataset}.png"))
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
