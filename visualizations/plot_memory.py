#!/usr/bin/env python3
"""Peak memory comparison across datasets: α-matched DiskANN vs StagedDiskANN.
Both engines build at the dataset's PA-aligned `scfg.alpha`; the only
delta is the `compute_candidate_sets` flag plus the per-node 60/40
partition that materialises the PhasedGraph. The figure isolates that
delta on peak / final RSS."""

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

data = {}
for name, _ in datasets:
    path = f"visualizations/memory_profile_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            data[name] = json.load(f)

present = [(n, t) for n, t in datasets if n in data]
if not present:
    print("No memory profile data. Run: cargo run ... --algorithms memory-profile")
    sys.exit(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
style_fig(fig)

MB = 1024 * 1024

# ── Left: Absolute peak memory (DiskANN vs Staged peak vs Staged final) ──
ax = axes[0]
n = len(present)
x = np.arange(n) * 1.15
w = 0.25
gap = 0.03

diskann_mb = [data[k]['diskann_peak_b'] / MB for k, _ in present]
staged_peak_mb = [data[k]['staged_peak_b'] / MB for k, _ in present]
staged_final_mb = [data[k]['staged_final_b'] / MB for k, _ in present]

h1 = rounded_bars(ax, x - (w + gap), diskann_mb, w, PALETTE['cyan'], label='DiskANN peak (α=match, no candidates)')
h2 = rounded_bars(ax, x, staged_peak_mb, w, PALETTE['blue'], label='Staged peak')
h3 = rounded_bars(ax, x + (w + gap), staged_final_mb, w, PALETTE['green'], label='Staged final')

ax.set_xticks(x); ax.set_xticklabels([t for _, t in present])
ax.set_ylabel('Memory (MB)', fontsize=11)
ax.set_title('Peak Memory: α-matched DiskANN vs StagedDiskANN', fontweight='bold')
make_legend(ax, [h for h in [h1, h2, h3] if h], fontsize=10)
style_ax(ax)
apply_ylim(ax, diskann_mb + staged_peak_mb + staged_final_mb, headroom=1.25)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

# ── Right: Ratio (Staged / DiskANN) ──
ax = axes[1]
ratio_peak = [sp / da if da > 0 else 1 for sp, da in zip(staged_peak_mb, diskann_mb)]

ds_colors = [DATASET_COLORS.get(k, PALETTE['grey']) for k, _ in present]
handles = []
for i, ((k, t), r, c) in enumerate(zip(present, ratio_peak, ds_colors)):
    h = rounded_bars(ax, [x[i]], [r], 0.45, c, label=t)
    if h: handles.append(h)

ax.axhline(y=1.0, color='#ccc', linestyle='--', alpha=0.5, linewidth=1)
for i, r in enumerate(ratio_peak):
    ax.annotate(f'{r:.2f}x', xy=(x[i], r),
                ha='center', va='bottom', fontsize=12, fontweight='bold', color=PALETTE['annot'])

ax.set_xticks(x); ax.set_xticklabels([t for _, t in present])
ax.set_ylabel('Peak Memory Ratio (Staged / DiskANN)', fontsize=11)
ax.set_title('Peak Memory Ratio at Matched α (Lower is Better)', fontweight='bold')
make_legend(ax, handles, fontsize=10)
style_ax(ax)
apply_ylim(ax, ratio_peak, headroom=1.4)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

plt.tight_layout()
out = 'visualizations/memory_analysis.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')
