#!/usr/bin/env python3
"""Plot QPS vs Recall@10 with confidence bands from multiple runs."""

import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from chart_style import save_png_and_pdf

datasets = [
    ("sift",    "SIFT (dim=128)"),
    ("glove25", "GloVe-25 (dim=32)"),
    ("glove100","GloVe-100 (dim=100)"),
    ("gist",    "GIST (dim=960)"),
]

def load_runs(name):
    """Load all run files for a dataset, return list of dicts."""
    pattern = f"visualizations/runs/qps_recall_{name}_run*.json"
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs

def annotate_ours(ax, recall, qps):
    """Place a small 'ours' label to the left of the Staged curve's start.

    The starting point (leftmost, highest-QPS) sits at the upper-left of the
    plot; a left offset lands in the upper-left margin, diagonally opposite
    the upper-right legend so the two never collide."""
    if len(recall) == 0:
        return
    i = 0
    ax.annotate('ours', xy=(recall[i], qps[i]),
                xytext=(-10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#598392',
                ha='right', va='center')
    ax.margins(x=0.05)


def compute_bands(runs, key):
    """Compute median, min, max QPS per L value across runs."""
    if not runs:
        return [], [], [], []
    # All runs should have same recall points (same L values)
    n_points = len(runs[0][key])
    recalls = [runs[0][key][i][0] for i in range(n_points)]

    qps_matrix = np.array([[run[key][i][1] for i in range(n_points)] for run in runs])

    median = np.median(qps_matrix, axis=0)
    lo = np.min(qps_matrix, axis=0)
    hi = np.max(qps_matrix, axis=0)

    # Use median recall too (should be very similar across runs)
    recall_matrix = np.array([[run[key][i][0] for i in range(n_points)] for run in runs])
    recalls = np.median(recall_matrix, axis=0)

    return recalls, median, lo, hi

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, title) in enumerate(datasets):
    ax = axes[idx]
    runs = load_runs(name)

    if not runs:
        # Fallback to single-run JSON
        path = f"visualizations/qps_recall_{name}.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            d_recall = [p[0] for p in data["diskann"]]
            d_qps = [p[1] for p in data["diskann"]]
            s_recall = [p[0] for p in data["staged"]]
            s_qps = [p[1] for p in data["staged"]]
            ax.plot(d_recall, d_qps, 'o-', color='#EEC170', label='DiskANN', markersize=5, linewidth=2)
            ax.plot(s_recall, s_qps, 's-', color='#598392', label='StagedDiskANN (ours)', markersize=5, linewidth=2)
            annotate_ours(ax, s_recall, s_qps)
            num_points = data.get('num_points', 0)
            threads = data.get('threads', 0)
            ax.set_title(f"{title} ({num_points//1000}K, {threads}T) [1 run]", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{title} (no data)")
    else:
        n_runs = len(runs)
        num_points = runs[0].get('num_points', 0)
        threads = runs[0].get('threads', 0)

        # DiskANN bands. Built at the dataset's staged params (sweep.yaml
        # `datasets.<name>.staged` block), so this single line is the
        # apples-to-apples comparison vs Staged. The legacy
        # `diskann_matched` JSON key from older runs is no longer read.
        d_recall, d_med, d_lo, d_hi = compute_bands(runs, 'diskann')
        ax.fill_between(d_recall, d_lo, d_hi, alpha=0.15, color='#EEC170')
        ax.plot(d_recall, d_med, 'o-', color='#EEC170', label='DiskANN', markersize=5, linewidth=2)

        # Staged bands
        s_recall, s_med, s_lo, s_hi = compute_bands(runs, 'staged')
        ax.fill_between(s_recall, s_lo, s_hi, alpha=0.15, color='#598392')
        ax.plot(s_recall, s_med, 's-', color='#598392', label='StagedDiskANN (ours)', markersize=5, linewidth=2)
        annotate_ours(ax, s_recall, s_med)

        ax.set_title(f"{title} ({num_points//1000}K, {threads}T, {n_runs} runs)", fontsize=12, fontweight='bold')

    # Overlay baselines if available
    baseline_path = f"visualizations/baseline_{name}.json"
    baseline_styles = {
        "hnsw":           ('^--', '#2E7D32', 'HNSW (hnswlib)'),
        "faiss_ivf_flat": ('v--', '#7B1FA2', 'FAISS IVF-Flat'),
        "faiss_ivf_pq":   ('d--', '#C62828', 'FAISS IVF-PQ'),
        "annoy":          ('x--', '#F57F17', 'Annoy'),
    }
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        for key, (marker, color, label) in baseline_styles.items():
            if key in bl and bl[key]:
                r = [p[0] for p in bl[key]]
                q = [p[1] for p in bl[key]]
                ax.plot(r, q, marker, color=color, label=label,
                        markersize=4, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    # ann-benchmarks–style legend: horizontal row below the axes, no frame.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=min(len(handles), 4),
              frameon=False, borderaxespad=0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.subplots_adjust(hspace=0.55, wspace=0.25)
out = "visualizations/qps_recall_all.png"
save_png_and_pdf(fig, out, pdf_font_scale=1.0)
print(f"Saved {out} (+ .pdf)")
plt.close(fig)

# ── DiskANN vs StagedDiskANN only (clean comparison) ──
fig_da, axes_da = plt.subplots(2, 2, figsize=(14, 10))
axes_da = axes_da.flatten()

for idx, (name, title) in enumerate(datasets):
    ax = axes_da[idx]
    runs = load_runs(name)

    if not runs:
        path = f"visualizations/qps_recall_{name}.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            d_recall = [p[0] for p in data["diskann"]]
            d_qps = [p[1] for p in data["diskann"]]
            s_recall = [p[0] for p in data["staged"]]
            s_qps = [p[1] for p in data["staged"]]
            ax.plot(d_recall, d_qps, 'o-', color='#EEC170', label='DiskANN', markersize=5, linewidth=2)
            ax.plot(s_recall, s_qps, 's-', color='#598392', label='StagedDiskANN (ours)', markersize=5, linewidth=2)
            annotate_ours(ax, s_recall, s_qps)
            num_points = data.get('num_points', 0)
            threads = data.get('threads', 0)
            ax.set_title(f"{title} ({num_points//1000}K, {threads}T)", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{title} (no data)")
    else:
        n_runs = len(runs)
        num_points = runs[0].get('num_points', 0)
        threads = runs[0].get('threads', 0)

        d_recall, d_med, d_lo, d_hi = compute_bands(runs, 'diskann')
        ax.fill_between(d_recall, d_lo, d_hi, alpha=0.15, color='#EEC170')
        ax.plot(d_recall, d_med, 'o-', color='#EEC170', label='DiskANN', markersize=5, linewidth=2)

        s_recall, s_med, s_lo, s_hi = compute_bands(runs, 'staged')
        ax.fill_between(s_recall, s_lo, s_hi, alpha=0.15, color='#598392')
        ax.plot(s_recall, s_med, 's-', color='#598392', label='StagedDiskANN (ours)', markersize=5, linewidth=2)
        annotate_ours(ax, s_recall, s_med)

        ax.set_title(f"{title} ({num_points//1000}K, {threads}T, {n_runs} runs)", fontsize=12, fontweight='bold')

    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    # ann-benchmarks–style legend: horizontal row below the axes, no frame.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.16), ncol=min(len(handles), 4),
              frameon=False, borderaxespad=0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_da.subplots_adjust(hspace=0.55, wspace=0.25)
out_da = "visualizations/qps_recall_diskann_vs_staged.png"
save_png_and_pdf(fig_da, out_da, pdf_font_scale=1.0)
print(f"Saved {out_da} (+ .pdf)")
plt.close(fig_da)

# Also save individual plots
for name, title in datasets:
    runs = load_runs(name)
    if not runs:
        continue

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    n_runs = len(runs)
    num_points = runs[0].get('num_points', 0)
    threads = runs[0].get('threads', 0)

    d_recall, d_med, d_lo, d_hi = compute_bands(runs, 'diskann')
    ax2.fill_between(d_recall, d_lo, d_hi, alpha=0.15, color='#EEC170')
    ax2.plot(d_recall, d_med, 'o-', color='#EEC170', label='DiskANN', markersize=6, linewidth=2.5)

    s_recall, s_med, s_lo, s_hi = compute_bands(runs, 'staged')
    ax2.fill_between(s_recall, s_lo, s_hi, alpha=0.15, color='#598392')
    ax2.plot(s_recall, s_med, 's-', color='#598392', label='StagedDiskANN (ours)', markersize=6, linewidth=2.5)
    annotate_ours(ax2, s_recall, s_med)

    # Overlay baselines if available
    baseline_path = f"visualizations/baseline_{name}.json"
    baseline_styles = {
        "hnsw":           ('^--', '#2E7D32', 'HNSW (hnswlib)'),
        "faiss_ivf_flat": ('v--', '#7B1FA2', 'FAISS IVF-Flat'),
        "faiss_ivf_pq":   ('d--', '#C62828', 'FAISS IVF-PQ'),
        "annoy":          ('x--', '#F57F17', 'Annoy'),
    }
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        for key, (marker, color, label) in baseline_styles.items():
            if key in bl and bl[key]:
                r = [p[0] for p in bl[key]]
                q = [p[1] for p in bl[key]]
                ax2.plot(r, q, marker, color=color, label=label,
                         markersize=5, linewidth=2, alpha=0.8)

    ax2.set_xlabel('Recall@10', fontsize=12)
    ax2.set_ylabel('QPS', fontsize=12)
    ax2.set_title(f"QPS vs Recall@10 -- {title}\n({num_points//1000}K points, {threads} threads, {n_runs} runs)",
                  fontsize=13, fontweight='bold')
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, fontsize=9, loc='upper center',
               bbox_to_anchor=(0.5, -0.16), ncol=min(len(handles2), 4),
               frameon=False, borderaxespad=0)
    ax2.grid(True, alpha=0.3)
    # Shrink axes to preserve figure size while making room for bottom legend.
    fig2.subplots_adjust(bottom=0.28)

    out2 = f"visualizations/qps_recall_{name}.png"
    save_png_and_pdf(fig2, out2, pdf_font_scale=1.0)
    plt.close(fig2)
    print(f"Saved {out2}")
