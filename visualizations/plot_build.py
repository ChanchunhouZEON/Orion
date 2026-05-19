#!/usr/bin/env python3
"""Build overhead: DiskANN vs StagedDiskANN build time."""

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

def load_build(name):
    """Prefer build_profile (multi-trial); fall back to calibration_diag (single-shot)."""
    bp = f"visualizations/build_profile_{name}.json"
    cd = f"visualizations/calibration_diag_{name}.json"
    if os.path.exists(bp):
        with open(bp) as f:
            d = json.load(f)
        return {
            'diskann_s': d['diskann_s'],
            'staged_graph_s': d['staged_graph_s'],
            'staged_overhead_s': d['staged_overhead_s'],
        }
    if os.path.exists(cd):
        with open(cd) as f:
            return json.load(f)['build']
    return None

diags = {}
for name, _ in datasets:
    b = load_build(name)
    if b is not None:
        diags[name] = {'build': b}

present = [(n, t) for n, t in datasets if n in diags]
short_labels = [t for _, t in present]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
style_fig(fig)

# ── Chart 1: Build time breakdown ──
ax = axes[0]
x = np.arange(len(present))
w = 0.3
gap = 0.05

da_times = [diags[n]['build']['diskann_s'] for n, _ in present]
st_graph = [diags[n]['build']['staged_graph_s'] for n, _ in present]
st_overhead = [diags[n]['build']['staged_overhead_s'] for n, _ in present]

h1 = rounded_bars(ax, x - w/2 - gap/2, da_times, w, PALETTE['cyan'], label='DiskANN (a=2.0)')
h2 = rounded_bars(ax, x + w/2 + gap/2, [g + o for g, o in zip(st_graph, st_overhead)], w, PALETTE['orange'], label='Staged (graph+overhead)')

for i in range(len(present)):
    total = st_graph[i] + st_overhead[i]
    pct = st_overhead[i] / total * 100 if total > 0 else 0
    ax.annotate(f'+{pct:.1f}%', xy=(x[i] + w/2 + gap/2, total),
                ha='center', va='bottom', fontsize=12, color=PALETTE['annot'], fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(short_labels)
ax.set_ylabel('Build Time (seconds)', fontsize=11)
ax.set_title('Build Time: DiskANN vs StagedDiskANN', fontweight='bold')
make_legend(ax, [h for h in [h1, h2] if h], fontsize=10)
style_ax(ax)
apply_ylim(ax, da_times + [g + o for g, o in zip(st_graph, st_overhead)], headroom=1.2)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

# ── Chart 2: Build speed ratio ──
ax = axes[1]
speedup = []
overhead_pct = []
for n, _ in present:
    d = diags[n]['build']
    total_staged = d['staged_graph_s'] + d['staged_overhead_s']
    speedup.append(d['diskann_s'] / total_staged)
    overhead_pct.append(d['staged_overhead_s'] / total_staged * 100)

ds_colors = [DATASET_COLORS.get(n, PALETTE['grey']) for n, _ in present]
handles = []
for i, ((n, t), s, c) in enumerate(zip(present, speedup, ds_colors)):
    h = rounded_bars(ax, [x[i]], [s], 0.45, c, label=t)
    if h: handles.append(h)

ax.axhline(y=1.0, color='#ccc', linestyle='--', alpha=0.5, linewidth=1)
for i, (s, o) in enumerate(zip(speedup, overhead_pct)):
    ax.annotate(f'{s:.2f}x', xy=(x[i], s),
                ha='center', va='bottom', fontsize=12, fontweight='bold', color=PALETTE['annot'])

ax.set_xticks(x); ax.set_xticklabels(short_labels)
ax.set_ylabel('Build Speed Ratio (DiskANN / Staged)', fontsize=11)
ax.set_title('Staged Build is Faster (Lower Alpha)', fontweight='bold')
make_legend(ax, handles, fontsize=10)
style_ax(ax)
apply_ylim(ax, speedup, headroom=1.4)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

plt.tight_layout()
out = 'visualizations/build_analysis.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')
