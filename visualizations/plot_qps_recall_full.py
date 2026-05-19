#!/usr/bin/env python3
"""Plot QPS vs Recall@10 on full datasets (SIFT1M / GLoVe-25 / GLoVe-100 / GIST).

Reads raw per-run JSONs from `visualizations/runs/qps_recall_{slug}_run*.json`
and renders a 2x2 grid with median line + min/max shaded band per dataset.
Also saves one standalone PNG/PDF per dataset.
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from chart_style import save_png_and_pdf

VIZ_DIR  = os.path.dirname(__file__)
RUNS_DIR = os.path.join(VIZ_DIR, "runs")

DATASETS = [
    ("sift1m",      "SIFT (dim=128)"),
    ("glove25full", "GLoVe-25 (dim=32)"),
    ("glove100full","GLoVe-100 (dim=100)"),
    ("gistfull",    "GIST (dim=960)"),
]


def load_runs(slug):
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, f"qps_recall_{slug}_run*.json")))
    return [json.load(open(p)) for p in paths]


def median_curve(runs, key):
    arrs = np.stack([np.array(r[key]) for r in runs])  # (N_runs, N_L, 2)
    med_recall = np.median(arrs[:, :, 0], axis=0)
    med_qps    = np.median(arrs[:, :, 1], axis=0)
    return med_recall, med_qps


def render(ax, runs, title, show_ours=True):
    if not runs:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    n_runs = len(runs)
    num_points = runs[0]["num_points"]
    threads    = runs[0]["threads"]

    # DiskANN baseline uses the dataset's staged params (sweep.yaml
    # `datasets.<name>.staged` block) — apples-to-apples with Staged.
    d_recall, d_med = median_curve(runs, "diskann")
    ax.plot(d_recall, d_med, 'o-', color='#EEC170',
            label='DiskANN', markersize=6, linewidth=2.5)

    s_recall, s_med = median_curve(runs, "staged")
    ax.plot(s_recall, s_med, 's-', color='#598392',
            label='StagedDiskANN (ours)', markersize=6, linewidth=2.5)

    # Optional ParlayANN overlay, if `parlayann_{slug}.json` exists.
    slug = runs[0].get("dataset", "")
    pa_path = os.path.join(VIZ_DIR, f"parlayann_{slug}.json")
    if slug == "sift" and num_points == 1_000_000:
        pa_path = os.path.join(VIZ_DIR, "parlayann_sift1m.json")
    if os.path.exists(pa_path):
        with open(pa_path) as pf:
            pa = json.load(pf)
        pa_curve = pa.get("parlayann_vamana", [])
        if pa_curve:
            pa_r = [p[0] for p in pa_curve]
            pa_q = [p[1] for p in pa_curve]
            ax.plot(pa_r, pa_q, '^--', color='#4E6E8E',
                    label='ParlayANN Vamana', markersize=5, linewidth=2.0)

    if show_ours:
        # Anchor at the Staged curve's peak (argmax QPS) with a small up-right
        # offset — lands above the peak, slightly to the right.
        ax.margins(x=0.08, y=0.14)
        peak = int(np.argmax(s_med))
        ax.annotate('ours', xy=(s_recall[peak], s_med[peak]),
                    xytext=(14, 6), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#598392',
                    ha='left', va='bottom')

    # Scale label: "1M" when divisible by 1M, else "NK".
    if num_points >= 1_000_000 and num_points % 1_000_000 == 0:
        size_label = f"{num_points // 1_000_000}M"
    elif num_points >= 100_000:
        size_label = f"{num_points // 1000}K"
    else:
        size_label = str(num_points)

    ax.set_title(f"{title}\n({size_label} pts, {threads} threads, "
                 f"median of {n_runs} runs)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)


def main():
    # ── Combined 2x2 panel ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for ax, (slug, title) in zip(axes, DATASETS):
        render(ax, load_runs(slug), title)
    plt.tight_layout()
    out = os.path.join(VIZ_DIR, "qps_recall_full.png")
    save_png_and_pdf(fig, out)
    print(f"Saved {out} (+ .pdf)")
    plt.close(fig)

    # ── Individual panels ─────────────────────────────────────────────
    for slug, title in DATASETS:
        runs = load_runs(slug)
        if not runs:
            print(f"Skipped {slug}: no runs")
            continue
        fig2, ax2 = plt.subplots(figsize=(9.5, 5))
        render(ax2, runs, title)
        out2 = os.path.join(VIZ_DIR, f"qps_recall_{slug}.png")
        save_png_and_pdf(fig2, out2)
        print(f"Saved {out2} (+ .pdf)")
        plt.close(fig2)


if __name__ == "__main__":
    main()
