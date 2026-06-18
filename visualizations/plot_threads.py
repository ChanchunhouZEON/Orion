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

# Apple M4 Max topology — 10 P-cores + 4 E-cores. The P-core
# saturation knee (T=10) is the most informative annotation point;
# after T=10 the next workers spill onto E-cores (~3× slower per op),
# and after T=14 the scheduler over-subscribes physical cores.
P_CORES = 10
TOTAL_CORES = 14

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

    # Mark the P-core saturation knee (T=10) and the all-physical-cores
    # boundary (T=14) so the eye picks up the topology transitions
    # without having to back-trace through tick labels.
    for x_mark, color, label in [
        (P_CORES,     '#7BC4A8', f'P-core knee (T={P_CORES})'),
        (TOTAL_CORES, '#E0A24A', f'all cores (T={TOTAL_CORES})'),
    ]:
        if min(ts) <= x_mark <= max(ts):
            ax.axvline(x=x_mark, color=color, linestyle=':', linewidth=1.4,
                       alpha=0.65, zorder=1)

    # (Per-panel speedup/efficiency annotation removed for the paper
    # figure — it sat at `s_med[ik] * 1.4` which collided with the
    # subplot title. The same numbers are printed to stdout at the
    # bottom of this script for the caption text.)

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

print(f"\n─── Parallel Efficiency @ P-core knee (T={P_CORES}) ───")
for name, title in present:
    d = data[name]
    if P_CORES not in d['thread_counts']:
        continue
    idx = d['thread_counts'].index(P_CORES)
    print(f"  {title:<10}  DiskANN: {d['diskann']['speedup'][idx]:.2f}× ({d['diskann']['efficiency'][idx]*100:.0f}%)  "
          f"Staged: {d['staged']['speedup'][idx]:.2f}× ({d['staged']['efficiency'][idx]*100:.0f}%)")
