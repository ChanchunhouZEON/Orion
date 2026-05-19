#!/usr/bin/env python3
"""Calibration diagnostics: per-dataset admission-rate curve + useful-gap PDF.

Original 4×2 layout (one row per dataset). Font sizes are pushed up so the
figure stays legible when scaled to half-page A4 in a paper.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from chart_style import PALETTE, DATASET_COLORS, style_ax, style_fig, save_png_and_pdf

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

present = [(n, t) for n, t in datasets if n in diags]
n_ds = len(present)

# Font sizes kept at natural (PNG-friendly) values; the PDF version produced
# by save_png_and_pdf scales everything up uniformly for the paper.
plt.rcParams.update({
    'font.size':       12,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

fig, axes = plt.subplots(n_ds, 2, figsize=(14, 4.6 * n_ds))
if n_ds == 1:
    axes = axes.reshape(1, -1)
style_fig(fig)

for row, (name, title) in enumerate(present):
    d = diags[name]
    rates = np.asarray(d['admission_rates'])
    thr = d['threshold']
    gaps = np.asarray(d.get('useful_gaps', d.get('tail_gaps', [])), dtype=float)
    ee = d['early_exit_limit']
    color = DATASET_COLORS.get(name, PALETTE['grey'])

    # ── Left: Admission rate curve ──
    ax = axes[row, 0]
    steps = np.arange(len(rates))
    ax.plot(steps, rates, linewidth=3.4, color=color, zorder=4,
            label='Admission rate')
    ax.axhline(y=thr, color=color, linestyle='--', linewidth=2.2, alpha=0.8,
               zorder=3)
    ax.fill_between(steps, 0, rates, where=np.array(rates) > thr,
                    alpha=0.18, color=color, label='Navigation (above τ)')
    ax.fill_between(steps, 0, rates, where=np.array(rates) <= thr,
                    alpha=0.12, color='gray', label='Converged (below τ)')
    # Inline τ label: anchored near the left edge where the admission-rate
    # curve is still ~1.0, so the area just above the threshold line is always
    # empty (even for GIST where the tail of the curve oscillates near τ).
    ax.annotate(f'τ = {thr:.2f}',
                xy=(len(rates) * 0.02, thr),
                xytext=(0, 6), textcoords='offset points',
                fontsize=16, fontweight='bold', color=color,
                va='bottom', ha='left', zorder=6)
    ax.set_xlabel('Search Step')
    ax.set_ylabel('Admission Rate')
    ax.set_title(f'{title} — Admission Rate', fontweight='bold')
    ax.set_ylim(-0.02, 1.08)
    ax.legend(loc='upper right', frameon=True, facecolor='white',
              edgecolor='#E0E0E0', labelcolor=PALETTE['text'])
    style_ax(ax)

    # ── Right: Useful-gap PDF via Gaussian KDE ──
    ax = axes[row, 1]
    if gaps.size:
        x_max = min(60, int(gaps.max()) + 5)
        x_range = np.linspace(0, x_max, 400)
        bw = max(1.0, gaps.std() * 0.2)
        density = np.zeros_like(x_range)
        for g in gaps:
            density += np.exp(-0.5 * ((x_range - g) / bw) ** 2)
        density /= (len(gaps) * bw * np.sqrt(2 * np.pi))

        ax.plot(x_range, density, linewidth=3.4, color=color, zorder=4,
                label='Useful-gap density')
        ax.fill_between(x_range, 0, density, where=x_range <= ee,
                        alpha=0.22, color=color, label='Covered by ee (P95)')
        ax.fill_between(x_range, 0, density, where=x_range > ee,
                        alpha=0.18, color='red', label='Beyond ee (5%)')
        ax.axvline(x=ee, color=color, linestyle='--', linewidth=2.2, alpha=0.85)
        # Inline ee label: mid-height (40% of peak) is always below the
        # upper-right legend and the density curve has already decayed there.
        ax.annotate(f'ee = {ee}',
                    xy=(ee, density.max() * 0.4),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=16, fontweight='bold', color=color,
                    va='center', ha='left', zorder=6)
        ax.set_xlim(0, x_max)
    ax.set_xlabel('Inter-Top-K Admission Gap (Steps)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} — Useful-Gap Distribution', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white',
              edgecolor='#E0E0E0', labelcolor=PALETTE['text'])
    style_ax(ax)

plt.tight_layout(pad=1.0, h_pad=3.5, w_pad=2.0)
out = 'visualizations/calibration_analysis.png'
save_png_and_pdf(fig, out)
print(f'Saved {out} (+ .pdf)')
