#!/usr/bin/env python3
"""Convergence diagnostics (SIFT + GIST): steps saved, distance calls saved,
graph structure, trade-off.

Data source: visualizations/convergence_diag_{name}.json.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from chart_style import *

datasets = [("sift", "SIFT"), ("gist", "GIST")]

data = {}
for name, _ in datasets:
    path = f"visualizations/convergence_diag_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            data[name] = json.load(f)

present = [(n, t) for n, t in datasets if n in data]
if not present:
    print("No convergence_diag data. Run: cargo run ... --algorithms convergence-diag")
    sys.exit(1)

L_values = data[present[0][0]]['L_values']
n_L = len(L_values)
x = np.arange(n_L)
w = 0.18
gap = 0.03

fig, axes = plt.subplots(1, 4, figsize=(26, 6.5))
style_fig(fig)
axes = axes.reshape(1, 4)

# Per-dataset color pair: lighter = no-ee baseline, solid = staged.
color_pairs = {
    'sift': (PALETTE['cyan'],   PALETTE['blue']),
    'gist': (PALETTE['orange'], PALETTE['red']),
}

# ── Chart 1: Avg steps per query ──
ax = axes[0, 0]  # Chart 1
handles = []
all_step_vals = []
for di, (name, title) in enumerate(present):
    d = data[name]
    no_ee = d['no_early_exit']['steps']
    staged = d['staged']['steps']
    all_step_vals.extend(no_ee + staged)
    c_bl, c_st = color_pairs.get(name, (PALETTE['grey'], PALETTE['blue']))

    pos_bl = x + (2 * di - len(present) + 0.5) * (w + gap) - (w + gap) / 2
    pos_st = x + (2 * di - len(present) + 0.5) * (w + gap) + (w + gap) / 2

    h_bl = rounded_bars(ax, pos_bl, no_ee, w, c_bl, label=f'{title} no-ee')
    h_st = rounded_bars(ax, pos_st, staged, w, c_st, label=f'{title} staged')
    if h_bl: handles.append(h_bl)
    if h_st: handles.append(h_st)

    for j in range(n_L):
        pct = (staged[j] / no_ee[j] - 1) * 100
        ax.annotate(f'{pct:+.0f}%', xy=(pos_st[j], staged[j]),
                    ha='center', va='bottom', fontsize=12, color=c_st, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels([f'L={l}' for l in L_values])
ax.set_ylabel('Avg Steps per Query')
ax.set_title('Early Exit: Steps Saved', fontweight='bold')
make_legend(ax, handles, ncol=2, fontsize=9)
style_ax(ax)
apply_ylim(ax, all_step_vals, headroom=1.25)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

# ── Chart 2: Avg distance computations per query ──
ax = axes[0, 1]  # Chart 2
handles2 = []
all_ndc_vals = []
for di, (name, title) in enumerate(present):
    d = data[name]
    no_ee = d['no_early_exit']['ndc']
    staged = d['staged']['ndc']
    all_ndc_vals.extend(no_ee + staged)
    c_bl, c_st = color_pairs.get(name, (PALETTE['grey'], PALETTE['blue']))

    pos_bl = x + (2 * di - len(present) + 0.5) * (w + gap) - (w + gap) / 2
    pos_st = x + (2 * di - len(present) + 0.5) * (w + gap) + (w + gap) / 2

    h_bl = rounded_bars(ax, pos_bl, no_ee, w, c_bl, label=f'{title} no-ee')
    h_st = rounded_bars(ax, pos_st, staged, w, c_st, label=f'{title} staged')
    if h_bl: handles2.append(h_bl)
    if h_st: handles2.append(h_st)

    for j in range(n_L):
        pct = (staged[j] / no_ee[j] - 1) * 100
        ax.annotate(f'{pct:+.0f}%', xy=(pos_st[j], staged[j]),
                    ha='center', va='bottom', fontsize=12, color=c_st, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels([f'L={l}' for l in L_values])
ax.set_ylabel('Avg Distance Computations per Query')
ax.set_title('Distance Computation Reduction', fontweight='bold')
make_legend(ax, handles2, ncol=2, fontsize=9)
style_ax(ax)
apply_ylim(ax, all_ndc_vals, headroom=1.25)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

# ── Chart 3: PhasedGraph structure ──
ax = axes[0, 2]
categories = ['Avg Degree', 'Avg Local', 'Avg Extra', 'Avg Rerank']
x3 = np.arange(len(categories)) * 1.1
w3 = 0.3
gap3 = 0.04

handles3 = []
all_g_vals = []
for di, (name, title) in enumerate(present):
    g = data[name]['graph']
    vals = [g['avg_degree'], g['avg_local'], g['avg_extra'], g['avg_rerank']]
    all_g_vals.extend(vals)
    _, c_st = color_pairs.get(name, (PALETTE['grey'], PALETTE['blue']))
    pos = x3 + (di - (len(present) - 1) / 2) * (w3 + gap3)
    alpha_label = f"α={data[name]['alpha_staged']:.1f}"
    h = rounded_bars(ax, pos, vals, w3, c_st, label=f'{title} ({alpha_label})')
    if h: handles3.append(h)

ax.set_xticks(x3); ax.set_xticklabels(categories)
ax.set_ylabel('Count')
ax.set_title('PhasedGraph Structure', fontweight='bold')
make_legend(ax, handles3, fontsize=10)
style_ax(ax)
apply_ylim(ax, all_g_vals, headroom=1.2)
ax.set_xlim(x3[0] - 0.6, x3[-1] + 0.6)

# ── Chart 4: Steps saved vs recall loss ──
ax = axes[0, 3]
for name, title in present:
    d = data[name]
    _, c_st = color_pairs.get(name, (PALETTE['grey'], PALETTE['blue']))
    marker = 'o' if name == 'sift' else 's'
    for j, l in enumerate(L_values):
        saved_pct = (1 - d['staged']['steps'][j] / d['no_early_exit']['steps'][j]) * 100
        recall_loss_pp = (d['no_early_exit']['recall'][j] - d['staged']['recall'][j]) * 100
        ax.scatter(saved_pct, recall_loss_pp, s=110, color=c_st, marker=marker,
                   zorder=5, edgecolor='white', linewidth=1.5)
        ax.annotate(f'{title} L={l}', (saved_pct + 0.5, recall_loss_pp),
                    fontsize=12, color=c_st)

ax.set_xlabel('Steps Saved (%)')
ax.set_ylabel('Recall Loss (pp)')
ax.set_title('Early Exit Trade-off: Savings vs Recall', fontweight='bold')
ax.axhline(y=0, color='#ccc', linestyle='--', alpha=0.5)
style_ax(ax)

plt.tight_layout()
out = 'visualizations/diagnostics.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')
