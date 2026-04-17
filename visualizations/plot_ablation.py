#!/usr/bin/env python3
"""Ablation study: grouped rounded bar chart with DiskANN baseline."""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from chart_style import *

datasets = [
    ("sift",    "SIFT (128-dim)"),
    ("glove100","GloVe-100 (100-dim)"),
]

variants = [
    ("diskann",       "DiskANN"),
    ("full",          "Full Staged"),
    ("no_early_exit", "No Early Exit"),
    ("no_extra",      "No Extra"),
]
variant_colors = [PALETTE['grey'], PALETTE['blue'], PALETTE['green'], PALETTE['purple']]
target_Ls = [32, 64, 128, 256]

present = [(n, t) for n, t in datasets if os.path.exists(f"visualizations/ablation_{n}.json")]
n_ds = len(present)
if not n_ds:
    print("No ablation data."); exit()

fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5.5))
style_fig(fig)
if n_ds == 1: axes = [axes]

for idx, (name, title) in enumerate(present):
    ax = axes[idx]
    with open(f"visualizations/ablation_{name}.json") as f:
        data = json.load(f)

    all_Ls = data.get("search_list_sizes", [])
    L_indices = [all_Ls.index(l) for l in target_Ls if l in all_Ls]
    L_labels = [f"L={all_Ls[i]}" for i in L_indices]

    n_L = len(L_indices)
    n_var = len(variants)
    x = np.arange(n_L) * 1.15
    w = 0.18
    gap = 0.03

    handles = []
    all_qps = []
    for vi, (key, label) in enumerate(variants):
        if key not in data: continue
        points = data[key]
        qps_vals = [points[i][1] for i in L_indices]
        all_qps.extend(qps_vals)
        positions = x + (vi - (n_var - 1) / 2) * (w + gap)
        h = rounded_bars(ax, positions, qps_vals, w, variant_colors[vi], label=label)
        if h: handles.append(h)

        if key != "diskann" and "diskann" in data:
            da_qps = [data["diskann"][i][1] for i in L_indices]
            for j, (q, dq) in enumerate(zip(qps_vals, da_qps)):
                if dq > 0:
                    pct = (q / dq - 1) * 100
                    ax.annotate(f'{pct:+.0f}%', xy=(positions[j], q),
                                ha='center', va='bottom', fontsize=7,
                                color=PALETTE['annot'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(L_labels)
    ax.set_ylabel('QPS', fontsize=11)
    n_pts = data.get('num_points', 0)
    threads = data.get('threads', 0)
    ax.set_title(f'{title} ({n_pts//1000}K, {threads}T)', fontsize=12, fontweight='bold')
    make_legend(ax, handles, loc='upper right', fontsize=8)
    style_ax(ax)
    apply_ylim(ax, all_qps, headroom=1.35)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

plt.tight_layout()
out = 'visualizations/ablation_study.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
