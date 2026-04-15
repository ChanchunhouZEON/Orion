#!/usr/bin/env python3
"""Plot build overhead: DiskANN vs StagedDiskANN build time and overhead ratio."""

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

diags = {}
for name, _ in datasets:
    path = f"visualizations/calibration_diag_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            diags[name] = json.load(f)

colors = {'sift': '#1565C0', 'glove25': '#2E7D32', 'glove100': '#E65100', 'gist': '#BF360C'}

names_present = [n for n, _ in datasets if n in diags]
titles_present = [t for n, t in datasets if n in diags]
short_labels = [t.split('(')[0].strip() for t in titles_present]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Chart 1: Build time breakdown ──
ax = axes[0]
x = np.arange(len(names_present))
w = 0.35

da_times = [diags[n]['build']['diskann_s'] for n in names_present]
st_graph = [diags[n]['build']['staged_graph_s'] for n in names_present]
st_overhead = [diags[n]['build']['staged_overhead_s'] for n in names_present]

ax.bar(x - w/2, da_times, w, label='DiskANN (a=2.0)', color='#90CAF9')
ax.bar(x + w/2, st_graph, w, label='Staged Graph Build', color='#FF8A65')
ax.bar(x + w/2, st_overhead, w, bottom=st_graph, label='Staged Overhead', color='#BF360C')

for i in range(len(names_present)):
    total = st_graph[i] + st_overhead[i]
    pct = st_overhead[i] / total * 100 if total > 0 else 0
    ax.annotate(f'+{pct:.1f}%', xy=(i + w/2, total + 0.15),
                ha='center', fontsize=9, color='#BF360C', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(short_labels)
ax.set_ylabel('Build Time (seconds)')
ax.set_title('Build Time: DiskANN vs StagedDiskANN', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# ── Chart 2: Build speed ratio ──
ax = axes[1]
speedup = []
overhead_pct = []
for n in names_present:
    d = diags[n]['build']
    total_staged = d['staged_graph_s'] + d['staged_overhead_s']
    speedup.append(d['diskann_s'] / total_staged)
    overhead_pct.append(d['staged_overhead_s'] / total_staged * 100)

bars = ax.bar(np.arange(len(names_present)), speedup, 0.5,
              color=[colors[n] for n in names_present])
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xticks(np.arange(len(names_present)))
ax.set_xticklabels(short_labels)
ax.set_ylabel('Build Speed Ratio (DiskANN / Staged)')
ax.set_title('Staged Build is Faster (Lower Alpha)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (s, o) in enumerate(zip(speedup, overhead_pct)):
    ax.annotate(f'{s:.2f}x\n(+{o:.1f}% overhead)', xy=(i, s + 0.02),
                ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
out = 'visualizations/build_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
