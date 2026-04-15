#!/usr/bin/env python3
"""Generate diagnostic charts: early exit savings, distance computation breakdown."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data from search-profile runs ──

L_values = [48, 100, 200]

# SIFT 128-dim (10K queries, threshold=0.21, ee=11)
sift = {
    'baseline_steps': [52.1, 103.6, 203.2],
    'staged_steps':   [49.3, 83.6, 132.2],
    'baseline_dist':  [893.0, 1505.1, 2476.4],
    'staged_dist':    [854.5, 1288.3, 1818.4],
    'baseline_recall':[0.9836, 0.9965, 0.9993],
    'staged_recall':  [0.9829, 0.9954, 0.9985],
}

# GIST 960-dim (1K queries, threshold=0.23, ee=14)
gist = {
    'baseline_steps': [52.1, 103.1, 202.5],
    'staged_steps':   [51.5, 96.8, 170.0],
    'baseline_dist':  [1185.6, 2059.6, 3482.2],
    'staged_dist':    [1157.3, 1934.8, 3018.5],
    'baseline_recall':[0.8988, 0.9605, 0.9850],
    'staged_recall':  [0.8953, 0.9559, 0.9807],
}

# Graph structure
graph_info = {
    'sift': {'dim': 128, 'avg_degree': 24.7, 'avg_local': 19.9, 'avg_extra': 3.5, 'alpha': 1.2},
    'gist': {'dim': 960, 'avg_degree': 25.9, 'avg_local': 18.1, 'avg_extra': 2.6, 'alpha': 1.5},
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Chart 1: Steps saved (early exit effectiveness) ──
ax = axes[0, 0]
x = np.arange(len(L_values))
w = 0.18
for i, (name, data, color) in enumerate([
    ('SIFT baseline', sift['baseline_steps'], '#90CAF9'),
    ('SIFT staged',   sift['staged_steps'],   '#1565C0'),
    ('GIST baseline', gist['baseline_steps'],  '#FFAB91'),
    ('GIST staged',   gist['staged_steps'],    '#BF360C'),
]):
    ax.bar(x + (i - 1.5) * w, data, w, label=name, color=color)

ax.set_xticks(x)
ax.set_xticklabels([f'L={l}' for l in L_values])
ax.set_ylabel('Avg Steps per Query')
ax.set_title('Early Exit: Steps Saved', fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')

# Add % saved annotations
for i, l in enumerate(L_values):
    sift_pct = (1 - sift['staged_steps'][i] / sift['baseline_steps'][i]) * 100
    gist_pct = (1 - gist['staged_steps'][i] / gist['baseline_steps'][i]) * 100
    ax.annotate(f'-{sift_pct:.0f}%', xy=(i - 0.5*w, sift['staged_steps'][i]),
                ha='center', va='bottom', fontsize=8, color='#1565C0', fontweight='bold')
    ax.annotate(f'-{gist_pct:.0f}%', xy=(i + 1.5*w, gist['staged_steps'][i]),
                ha='center', va='bottom', fontsize=8, color='#BF360C', fontweight='bold')

# ── Chart 2: Distance calls saved ──
ax = axes[0, 1]
for i, (name, data, color) in enumerate([
    ('SIFT baseline', sift['baseline_dist'], '#90CAF9'),
    ('SIFT staged',   sift['staged_dist'],   '#1565C0'),
    ('GIST baseline', gist['baseline_dist'],  '#FFAB91'),
    ('GIST staged',   gist['staged_dist'],    '#BF360C'),
]):
    ax.bar(x + (i - 1.5) * w, data, w, label=name, color=color)

ax.set_xticks(x)
ax.set_xticklabels([f'L={l}' for l in L_values])
ax.set_ylabel('Avg Distance Computations per Query')
ax.set_title('Distance Computation Reduction', fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')

for i, l in enumerate(L_values):
    sift_pct = (1 - sift['staged_dist'][i] / sift['baseline_dist'][i]) * 100
    gist_pct = (1 - gist['staged_dist'][i] / gist['baseline_dist'][i]) * 100
    ax.annotate(f'-{sift_pct:.0f}%', xy=(i - 0.5*w, sift['staged_dist'][i]),
                ha='center', va='bottom', fontsize=8, color='#1565C0', fontweight='bold')
    ax.annotate(f'-{gist_pct:.0f}%', xy=(i + 1.5*w, gist['staged_dist'][i]),
                ha='center', va='bottom', fontsize=8, color='#BF360C', fontweight='bold')

# ── Chart 3: Graph structure comparison ──
ax = axes[1, 0]
categories = ['Avg Degree', 'Avg Local', 'Avg Extra', 'Avg Rerank\n(Local+Extra)']
sift_vals = [24.7, 19.9, 3.5, 23.3]
gist_vals = [25.9, 18.1, 2.6, 20.7]
x3 = np.arange(len(categories))
w3 = 0.3
ax.bar(x3 - w3/2, sift_vals, w3, label=f'SIFT (dim=128, a=1.2)', color='#1565C0')
ax.bar(x3 + w3/2, gist_vals, w3, label=f'GIST (dim=960, a=1.5)', color='#BF360C')
ax.set_xticks(x3)
ax.set_xticklabels(categories)
ax.set_ylabel('Count')
ax.set_title('PhasedGraph Structure', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# ── Chart 4: Recall vs steps saved trade-off ──
ax = axes[1, 1]
for i, l in enumerate(L_values):
    sift_saved = (1 - sift['staged_steps'][i] / sift['baseline_steps'][i]) * 100
    gist_saved = (1 - gist['staged_steps'][i] / gist['baseline_steps'][i]) * 100
    sift_loss = (sift['baseline_recall'][i] - sift['staged_recall'][i]) * 100
    gist_loss = (gist['baseline_recall'][i] - gist['staged_recall'][i]) * 100

    ax.scatter(sift_saved, sift_loss, s=100, color='#1565C0', marker='o', zorder=5)
    ax.annotate(f'SIFT L={l}', (sift_saved + 0.5, sift_loss), fontsize=8, color='#1565C0')
    ax.scatter(gist_saved, gist_loss, s=100, color='#BF360C', marker='s', zorder=5)
    ax.annotate(f'GIST L={l}', (gist_saved + 0.5, gist_loss), fontsize=8, color='#BF360C')

ax.set_xlabel('Steps Saved (%)')
ax.set_ylabel('Recall Loss (percentage points)')
ax.set_title('Early Exit Trade-off: Savings vs Recall', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
out = 'visualizations/diagnostics.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
