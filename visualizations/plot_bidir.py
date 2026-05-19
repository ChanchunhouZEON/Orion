#!/usr/bin/env python3
"""Bidirectional edge distribution: edge composition + per-node bidir fraction."""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from chart_style import *

datasets = [
    ("sift",    "SIFT"), ("glove25", "GloVe-25"),
    ("glove100","GloVe-100"), ("gist",    "GIST"),
]

bidir_data = {}
for name, _ in datasets:
    path = f"visualizations/bidir_distribution_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            bidir_data[name] = json.load(f)

present = [(n, t) for n, t in datasets if n in bidir_data]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
style_fig(fig)

# ── Left: Edge composition (stacked rounded bars) ──
ax = axes[0]
x = np.arange(len(present))
w = 0.45

avg_locals = []; avg_remotes = []; labels = []; bidir_ratios = []
for name, title in present:
    d = bidir_data[name]
    fracs = np.array(d['bidir_fractions'])
    rates = d['bidir_rate_by_rank']
    avg_frac = fracs.mean()
    bidir_ratios.append(avg_frac)
    deg = len(rates)
    avg_locals.append(avg_frac * deg)
    avg_remotes.append((1 - avg_frac) * deg)
    labels.append(f'{title}\n(a={d["alpha"]:.1f})')

h1 = rounded_bars(ax, x, avg_locals, w, PALETTE['blue'], label='Local (bidir)')
# Stack remote on top — draw as separate bars at offset height
for i in range(len(present)):
    rounded_bar(ax, x[i], avg_locals[i] + avg_remotes[i], w, PALETTE['orange'], alpha=0.85)
# Redraw local on top so it's visible
for i in range(len(present)):
    rounded_bar(ax, x[i], avg_locals[i], w, PALETTE['blue'], alpha=0.9)

h2 = rounded_patch(PALETTE['orange'], alpha=0.85, label='Remote (unidir)')

for i, ratio in enumerate(bidir_ratios):
    total = avg_locals[i] + avg_remotes[i]
    ax.annotate(f'{ratio:.0%} bidir', xy=(x[i], total),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=PALETTE['annot'])

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Avg Neighbors per Node')
ax.set_title('Edge Composition: Local vs Remote', fontweight='bold')
make_legend(ax, [h1, h2], fontsize=10)
style_ax(ax)
apply_ylim(ax, [l + r for l, r in zip(avg_locals, avg_remotes)], headroom=1.45)
ax.set_xlim(-0.6, len(present) - 0.4)

# ── Right: Per-node bidir fraction boxplot ──
ax = axes[1]
box_data = []; box_labels = []; box_colors = []
for name, title in present:
    d = bidir_data[name]
    box_data.append(np.array(d['bidir_fractions']))
    box_labels.append(f'{title}\n(a={d["alpha"]:.1f})')
    box_colors.append(DATASET_COLORS.get(name, PALETTE['grey']))

bp = ax.boxplot(box_data, positions=range(len(box_data)),
                widths=0.25, patch_artist=True,
                medianprops=dict(color=PALETTE['red'], linewidth=2),
                whiskerprops=dict(color='#999'), capprops=dict(color='#999'),
                flierprops=dict(marker='.', markersize=2, alpha=0.3))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(box_colors[i])
    patch.set_edgecolor('#999')
    patch.set_alpha(0.7)

ax.set_xticks(range(len(box_labels)))
ax.set_xticklabels(box_labels)
ax.set_ylabel('Bidir Fraction (local / degree)')
ax.set_title('Per-Node Bidir Fraction Distribution', fontweight='bold')
ax.set_ylim(0, 1.05)
style_ax(ax)

plt.tight_layout()
out = 'visualizations/bidir_analysis.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')
