#!/usr/bin/env python3
"""4-way ADSampling comparison: DiskANN / DiskANN+ADS / Orion / Orion+ADS."""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from chart_style import PALETTE, DATASET_COLORS, style_ax, style_fig, save_png_and_pdf

datasets = [
    ("sift",     "SIFT (128-d)"),
    ("glove25",  "GloVe-25 (32-d)"),
    ("glove100", "GloVe-100 (100-d)"),
    ("gist",     "GIST (960-d)"),
]

variants = [
    ("diskann",     "DiskANN",      PALETTE['cyan'],   'o-',  2.0),
    ("diskann_ads", "DiskANN+ADS",  PALETTE['orange'], 's--', 2.0),
    ("orion",      "Orion",PALETTE['blue'],   'D-',  2.3),
    ("orion_ads",  "Orion+ADS",   PALETTE['red'],    '^--', 2.3),
]

present = [(n, t) for n, t in datasets if os.path.exists(f"visualizations/ads_{n}.json")]
if not present:
    print("No ads_*.json data. Run: cargo run ... --algorithms ads-comparison")
    sys.exit(1)

n_ds = len(present)
cols = 2 if n_ds > 1 else 1
rows = (n_ds + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.5 * rows), squeeze=False)
style_fig(fig)

for idx, (name, title) in enumerate(present):
    ax = axes[idx // cols][idx % cols]
    with open(f"visualizations/ads_{name}.json") as f:
        data = json.load(f)

    for key, label, color, style, lw in variants:
        if key not in data or not data[key]:
            continue
        pts = data[key]
        recalls = [p[0] for p in pts]
        qpses = [p[1] for p in pts]
        ax.plot(recalls, qpses, style, color=color, label=label,
                markersize=6, linewidth=lw, alpha=0.9)

    ax.set_xlabel('Recall@10', fontsize=11)
    ax.set_ylabel('QPS (log)', fontsize=11)
    n_pts = data.get('num_points', 0)
    threads = data.get('threads', 0)
    eps = data.get('ads_epsilon', 2.1)
    ax.set_title(f'{title} ({n_pts // 1000}K, {threads}T, ε_ads={eps})',
                 fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='lower left', facecolor='white',
              edgecolor='#E0E0E0', labelcolor=PALETTE['text'])
    ax.grid(True, alpha=0.3, which='both')
    style_ax(ax)

# Hide any unused subplot slot.
for j in range(len(present), rows * cols):
    axes[j // cols][j % cols].axis('off')

plt.tight_layout()
out = 'visualizations/ads_comparison.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')
