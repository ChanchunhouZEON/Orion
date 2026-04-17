#!/usr/bin/env python3
"""Plot QPS vs Recall@10 with confidence bands from multiple runs."""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
            ax.plot(s_recall, s_qps, 's-', color='#598392', label='StagedDiskANN', markersize=5, linewidth=2)
            num_points = data.get('num_points', 0)
            threads = data.get('threads', 0)
            ax.set_title(f"{title} ({num_points//1000}K, {threads}T) [1 run]", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{title} (no data)")
    else:
        n_runs = len(runs)
        num_points = runs[0].get('num_points', 0)
        threads = runs[0].get('threads', 0)

        # DiskANN bands
        d_recall, d_med, d_lo, d_hi = compute_bands(runs, 'diskann')
        ax.fill_between(d_recall, d_lo, d_hi, alpha=0.15, color='#EEC170')
        ax.plot(d_recall, d_med, 'o-', color='#EEC170', label='DiskANN', markersize=5, linewidth=2)

        # Staged bands
        s_recall, s_med, s_lo, s_hi = compute_bands(runs, 'staged')
        ax.fill_between(s_recall, s_lo, s_hi, alpha=0.15, color='#598392')
        ax.plot(s_recall, s_med, 's-', color='#598392', label='StagedDiskANN', markersize=5, linewidth=2)

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
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "visualizations/qps_recall_all.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
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
            ax.plot(s_recall, s_qps, 's-', color='#598392', label='StagedDiskANN', markersize=5, linewidth=2)
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
        ax.plot(s_recall, s_med, 's-', color='#598392', label='StagedDiskANN', markersize=5, linewidth=2)

        ax.set_title(f"{title} ({num_points//1000}K, {threads}T, {n_runs} runs)", fontsize=12, fontweight='bold')

    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_da = "visualizations/qps_recall_diskann_vs_staged.png"
fig_da.savefig(out_da, dpi=150, bbox_inches='tight')
print(f"Saved {out_da}")
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
    ax2.plot(s_recall, s_med, 's-', color='#598392', label='StagedDiskANN', markersize=6, linewidth=2.5)

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
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    out2 = f"visualizations/qps_recall_{name}.png"
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {out2}")
