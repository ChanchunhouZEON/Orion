#!/usr/bin/env python3
"""Plot QPS vs Recall@10 curves for all datasets."""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datasets = [
    ("sift",    "SIFT (dim=128)"),
    ("glove25", "GloVe-25 (dim=32)"),
    ("glove100","GloVe-100 (dim=100)"),
    ("gist",    "GIST (dim=960)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, title) in enumerate(datasets):
    ax = axes[idx]
    path = f"visualizations/qps_recall_{name}.json"
    if not os.path.exists(path):
        ax.set_title(f"{title} (no data)")
        continue

    with open(path) as f:
        data = json.load(f)

    diskann = data["diskann"]
    staged = data["staged"]

    d_recall = [p[0] for p in diskann]
    d_qps    = [p[1] for p in diskann]
    s_recall = [p[0] for p in staged]
    s_qps    = [p[1] for p in staged]

    ax.plot(d_recall, d_qps, 'o-', color='#2196F3', label='DiskANN', markersize=5, linewidth=2)
    ax.plot(s_recall, s_qps, 's-', color='#FF5722', label='StagedDiskANN', markersize=5, linewidth=2)

    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    ax.set_title(f"{title} ({data['num_points']//1000}K, {data['threads']}T)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=min(d_recall + s_recall) - 0.01)

plt.tight_layout()
out = "visualizations/qps_recall_all.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")

# Also save individual plots
for name, title in datasets:
    path = f"visualizations/qps_recall_{name}.json"
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    diskann = data["diskann"]
    staged = data["staged"]

    ax2.plot([p[0] for p in diskann], [p[1] for p in diskann],
             'o-', color='#2196F3', label='DiskANN', markersize=6, linewidth=2.5)
    ax2.plot([p[0] for p in staged], [p[1] for p in staged],
             's-', color='#FF5722', label='StagedDiskANN', markersize=6, linewidth=2.5)

    ax2.set_xlabel('Recall@10', fontsize=12)
    ax2.set_ylabel('QPS', fontsize=12)
    ax2.set_title(f"QPS vs Recall@10 — {title}\n({data['num_points']//1000}K points, {data['threads']} threads)",
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    out2 = f"visualizations/qps_recall_{name}.png"
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {out2}")
