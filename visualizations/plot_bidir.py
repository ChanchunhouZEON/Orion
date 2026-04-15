#!/usr/bin/env python3
"""Plot bidirectional edge distribution: bidir rate by neighbor rank + per-node bidir fraction."""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

datasets = [
    ("sift",    "SIFT (128-dim)"),
    ("glove25", "GloVe-25 (32-dim)"),
    ("glove100","GloVe-100 (100-dim)"),
    ("gist",    "GIST (960-dim)"),
]

colors = {'sift': '#CBF3F0', 'glove25': '#CBF3F0', 'glove100': '#CBF3F0', 'gist': '#CBF3F0'}

bidir_data = {}
for name, _ in datasets:
    path = f"visualizations/bidir_distribution_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            bidir_data[name] = json.load(f)

present = [(n, t) for n, t in datasets if n in bidir_data]
n_ds = len(present)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Per-dataset edge composition (stacked bar) ──
ax = axes[0]
x = np.arange(n_ds)
w = 0.5

avg_locals = []
avg_remotes = []
avg_extras = []
labels = []
bidir_ratios = []

for name, title in present:
    d = bidir_data[name]
    fracs = np.array(d['bidir_fractions'])
    rates = d['bidir_rate_by_rank']
    n_pts = d['num_points']
    deg = len(rates)

    # Compute aggregate: total bidir edges / total edges
    total_local = sum(int(f * deg + 0.5) for f in fracs)
    total_edges = sum(min(deg, len(d['bidir_rate_by_rank'])) for _ in range(n_pts))
    # More accurate: sum degree per node
    # degree per node ~ len(rates) but capped; use fractions * degree
    avg_frac = fracs.mean()
    bidir_ratios.append(avg_frac)

    # Per-node averages for stacked bar
    avg_deg = deg  # max degree
    avg_lc = avg_frac * avg_deg
    avg_remote = avg_deg - avg_lc
    # extra from the data
    avg_locals.append(avg_lc)
    avg_remotes.append(avg_remote)
    labels.append(f'{title.split("(")[0].strip()}\n(a={d["alpha"]:.1f})')

bars_local = ax.bar(x, avg_locals, w, label='Local (bidir edges)', color='#0466C8')
bars_remote = ax.bar(x, avg_remotes, w, bottom=avg_locals, label='Remote (unidir edges)', color='#023E7D')

# Annotate bidir ratio on each bar
for i, ratio in enumerate(bidir_ratios):
    total = avg_locals[i] + avg_remotes[i]
    ax.annotate(f'{ratio:.0%} bidir',
                xy=(i, total + 0.3), ha='center', fontsize=10, fontweight='bold', color='#023E7D')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Avg Neighbors per Node')
ax.set_title('Graph Edge Composition: Local (Bidir) vs Remote (Unidir)', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# ── Right: Per-node bidir fraction violin plot ──
ax = axes[1]
violin_data = []
violin_labels = []
violin_colors = []
violin_means = []

for name, title in present:
    d = bidir_data[name]
    frac_arr = np.array(d['bidir_fractions'])
    violin_data.append(frac_arr)
    violin_labels.append(f'{title.split("(")[0].strip()}\n(a={d["alpha"]:.1f})')
    violin_colors.append(colors[name])
    violin_means.append(frac_arr.mean())

bp = ax.boxplot(violin_data, positions=range(len(violin_data)),
                widths=0.2, patch_artist=True,
                medianprops=dict(color='#2EC4B6', linewidth=2),
                whiskerprops=dict(color='#333'), capprops=dict(color='#333'),
                flierprops=dict(marker='.', markersize=2, alpha=0.3))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(violin_colors[i])
    patch.set_edgecolor('#333')
    patch.set_alpha(0.7)

# Annotate mean
# for i, m in enumerate(violin_means):
#     ax.annotate(f'mean={m:.2f}', xy=(i, m - 0.06),
#                 ha='center', fontsize=9, fontweight='bold', color=violin_colors[i])

ax.set_xticks(range(len(violin_labels)))
ax.set_xticklabels(violin_labels)
ax.set_ylabel('Bidir Fraction (local / degree)')
ax.set_title('Per-Node Bidir Fraction Distribution', fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = 'visualizations/bidir_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
