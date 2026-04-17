#!/usr/bin/env python3
"""Extra enrichment + early stop analysis with GLM styling."""

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

# ═══ Chart 1: Zone Admission Rate ═══
fig1, ax1 = plt.subplots(figsize=(9, 5.5))
style_fig(fig1)

zone_colors = [PALETTE['blue'], PALETTE['green'], PALETTE['orange']]
zone_labels = ['Local', 'Extra', 'Remote']
zone_keys = ['rerank_local', 'rerank_extra', 'rerank_remote']

bar_data = []
for name, title in datasets:
    zpath = f"visualizations/zone_admission_{name}.json"
    if not os.path.exists(zpath): continue
    with open(zpath) as f:
        zd = json.load(f)
    z = zd["zones"]
    def ar(zone):
        return zone["admitted"] / zone["unseen"] * 100 if zone["unseen"] > 0 else 0.0
    bar_data.append({
        "title": title, "rerank_local": ar(z["rerank_local"]),
        "rerank_extra": ar(z["rerank_extra"]), "rerank_remote": ar(z["rerank_remote"]),
    })

n_ds = len(bar_data)
x = np.arange(n_ds) * 1.2
w = 0.22; gap = 0.04

legend_handles = []
for zi, (zkey, zlabel, zcolor) in enumerate(zip(zone_keys, zone_labels, zone_colors)):
    positions = x + (zi - 1) * (w + gap)
    values = [d[zkey] for d in bar_data]
    h = rounded_bars(ax1, positions, values, w, zcolor, label=zlabel)
    if h: legend_handles.append(h)
    for i, v in enumerate(values):
        ax1.text(positions[i], v + 0.03, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=PALETTE['annot'])

ax1.set_xticks(x); ax1.set_xticklabels([d["title"] for d in bar_data])
ax1.set_ylabel('Admission Rate (%)', fontsize=11)
ax1.set_title('Post-Convergence Admission Rate by Neighbor Zone', fontsize=13, fontweight='bold')
all_vals = [d[k] for d in bar_data for k in zone_keys]
ax1.set_ylim(0, max(all_vals) * 1.35)
make_legend(ax1, legend_handles, fontsize=10)
style_ax(ax1); ax1.set_xlim(x[0] - 0.6, x[-1] + 0.6)

plt.tight_layout()
fig1.savefig('visualizations/extra_enrichment.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved visualizations/extra_enrichment.png')
plt.close(fig1)

# ═══ Chart 2: Top-K Coverage ═══
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
style_fig(fig2)

ee_annotations = []
for name, title in datasets:
    cpath = f"visualizations/calibration_diag_{name}.json"
    if not os.path.exists(cpath): continue
    with open(cpath) as f:
        cd = json.load(f)
    coverage = cd.get("topk_coverage_by_step", [])
    ee = cd.get("early_exit_limit", 0)
    thr = cd.get("threshold", 0)
    rates = cd.get("admission_rates", [])
    if not coverage: continue

    color = DATASET_COLORS[name]
    ax2.plot(np.arange(len(coverage)), coverage, linewidth=2.5, color=color, label=title, zorder=4)
    conv_step = next((i for i, r in enumerate(rates) if i > 10 and r < thr), len(rates)//2)
    ee_step = conv_step + ee
    ax2.axvline(x=ee_step, color=color, linestyle='--', alpha=0.5, linewidth=1.5, zorder=3)
    if ee_step < len(coverage):
        ee_annotations.append((ee_step, coverage[min(ee_step, len(coverage)-1)], title, color))

if ee_annotations:
    ee_annotations.sort(key=lambda t: -t[1])
    x_text = max(a[0] for a in ee_annotations) + 10
    y_positions = np.linspace(0.92, 0.92 - 0.05 * (len(ee_annotations) - 1), len(ee_annotations))
    for (es, cov, ttl, col), yt in zip(ee_annotations, y_positions):
        ax2.annotate(f'{ttl}: {cov:.1%}', xy=(es, cov), xytext=(x_text, yt),
                    fontsize=9, color=col, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=col, alpha=0.5, lw=1))

ax2.axhline(y=1.0, color='#ccc', linestyle='-', alpha=0.3, zorder=1)
ax2.set_xlabel('Search Step', fontsize=11)
ax2.set_ylabel('Fraction of Top-10 Found', fontsize=11)
ax2.set_title('Cumulative Top-K Coverage — Early Exit Cuts Negligible Tail', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, facecolor='white', edgecolor='#E0E0E0', loc='lower right')
style_ax(ax2); ax2.set_ylim(0.5, 1.02)

plt.tight_layout()
fig2.savefig('visualizations/earlystop_coverage.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved visualizations/earlystop_coverage.png')
plt.close(fig2)
