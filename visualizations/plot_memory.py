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

# Paper figure: single panel — the right "Peak Memory Ratio" plot.
# The left absolute-bars panel was useful in earlier iterations but
# the headline (Staged uses ~X× of DiskANN at matched α) reads more
# cleanly off the ratio bars. We compute the absolute MBs only as
# inputs to the ratio.
fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
style_fig(fig)

MB = 1024 * 1024

n = len(present)
x = np.arange(n) * 1.15

diskann_mb = [data[k]['diskann_peak_b'] / MB for k, _ in present]
staged_peak_mb = [data[k]['staged_peak_b'] / MB for k, _ in present]

# ── Ratio (Staged / DiskANN) ──
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
make_legend(ax, handles, fontsize=11)
style_ax(ax)
apply_ylim(ax, ratio_peak, headroom=1.4)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

plt.tight_layout()
out = 'visualizations/memory_analysis.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')
