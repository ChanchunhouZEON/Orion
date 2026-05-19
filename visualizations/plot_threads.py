#!/usr/bin/env python3
"""Multi-thread scaling per dataset: median QPS line + min/max band."""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from chart_style import *

datasets = [
    ("sift",    "SIFT"), ("gist",     "GIST"),
]

data = {}
for name, _ in datasets:
    path = f"visualizations/thread_sweep_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            data[name] = json.load(f)

present = [(n, t) for n, t in datasets if n in data]
if not present:
    print("No thread_sweep data. Run: cargo run ... --algorithms thread-sweep")
    sys.exit(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
style_fig(fig)
axes = axes.flatten()

C_DISKANN = PALETTE['cyan']
C_STAGED = PALETTE['blue']

for idx, (name, title) in enumerate(present):
    ax = axes[idx]
    d = data[name]
    ts = d['thread_counts']
    trials = d['trials']

    d_med = d['diskann']['qps_median']
    d_min = d['diskann']['qps_min']
    d_max = d['diskann']['qps_max']
    s_med = d['staged']['qps_median']
    s_min = d['staged']['qps_min']
    s_max = d['staged']['qps_max']

    # Shaded bands (min-max range across trials).
    ax.fill_between(ts, d_min, d_max, color=C_DISKANN, alpha=0.18, zorder=2)
    ax.fill_between(ts, s_min, s_max, color=C_STAGED,  alpha=0.18, zorder=2)

    ax.plot(ts, d_med, marker='s', markersize=7, linewidth=1.8, color=C_DISKANN,
            linestyle='--', label='DiskANN', zorder=4)
    ax.plot(ts, s_med, marker='o', markersize=7, linewidth=2.2, color=C_STAGED,
            label='Staged', zorder=5)

    # Annotate speedup at T=8 on Staged line.
    if 8 in ts:
        i8 = ts.index(8)
        sp = d['staged']['speedup'][i8]
        eff = d['staged']['efficiency'][i8] * 100
        ax.annotate(f'{sp:.1f}× @8T ({eff:.0f}% eff.)',
                    xy=(8, s_med[i8]), xytext=(6, s_med[i8] * 1.25),
                    fontsize=9, fontweight='bold', color=C_STAGED,
                    arrowprops=dict(arrowstyle='->', color=C_STAGED, lw=1, alpha=0.7))

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(ts)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Threads', fontsize=11)
    ax.set_ylabel('QPS', fontsize=11)
    n_pts = d.get('num_points', 0)
    ax.set_title(f'{title} ({n_pts // 1000}K pts, {trials} trials)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, facecolor='white', edgecolor='#E0E0E0',
              labelcolor=PALETTE['text'], loc='lower right')
    style_ax(ax)

plt.tight_layout()
out = 'visualizations/thread_scaling.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')

print("\n─── Parallel Efficiency @ 8 threads ───")
for name, title in present:
    d = data[name]
    if 8 not in d['thread_counts']:
        continue
    idx = d['thread_counts'].index(8)
    print(f"  {title:<10}  DiskANN: {d['diskann']['speedup'][idx]:.2f}× ({d['diskann']['efficiency'][idx]*100:.0f}%)  "
          f"Staged: {d['staged']['speedup'][idx]:.2f}× ({d['staged']['efficiency'][idx]*100:.0f}%)")
