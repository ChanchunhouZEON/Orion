#!/usr/bin/env python3
"""Plot calibration diagnostics: admission rate curves and tail gap distribution."""

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

present = [(n, t) for n, t in datasets if n in diags]
n_ds = len(present)

fig, axes = plt.subplots(n_ds, 2, figsize=(14, 4 * n_ds))
if n_ds == 1:
    axes = axes.reshape(1, -1)

for row, (name, title) in enumerate(present):
    d = diags[name]
    rates = d['admission_rates']
    thr = d['threshold']
    gaps = sorted(d['tail_gaps'])
    ee = d['early_exit_limit']
    color = colors[name]

    # ── Left: Admission rate curve ──
    ax = axes[row, 0]
    steps = np.arange(len(rates))
    ax.plot(steps, rates, linewidth=2, color=color)
    ax.axhline(y=thr, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    ax.fill_between(steps, 0, rates, where=np.array(rates) > thr, alpha=0.15, color=color, label='Navigation (above threshold)')
    ax.fill_between(steps, 0, rates, where=np.array(rates) <= thr, alpha=0.10, color='gray', label='Converged (below threshold)')
    ax.annotate(f'threshold = {thr:.2f}', xy=(len(rates)*0.55, thr + 0.03),
                fontsize=10, color=color, fontweight='bold')
    ax.set_xlabel('Search Step')
    ax.set_ylabel('Admission Rate')
    ax.set_title(f'{title} — Admission Rate', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)

    # ── Right: Useful gap (inter top-k admission gap) PDF as KDE curve ──
    ax = axes[row, 1]
    ugaps = d.get('useful_gaps', gaps)  # fallback to tail_gaps if not present
    if ugaps:
        gap_arr = np.array(ugaps, dtype=float)
        x_max = min(60, int(gap_arr.max()) + 5)
        x_range = np.linspace(0, x_max, 300)
        # Gaussian KDE
        bw = max(1.0, gap_arr.std() * 0.2)
        density = np.zeros_like(x_range)
        for g in gap_arr:
            density += np.exp(-0.5 * ((x_range - g) / bw) ** 2)
        density /= (len(gap_arr) * bw * np.sqrt(2 * np.pi))
        ax.plot(x_range, density, linewidth=2, color=color)
        ax.fill_between(x_range, 0, density, where=x_range <= ee,
                        alpha=0.2, color=color, label=f'Covered by ee (P95)')
        ax.fill_between(x_range, 0, density, where=x_range > ee,
                        alpha=0.15, color='red', label=f'Beyond ee (5%)')
        ax.axvline(x=ee, color=color, linestyle='--', linewidth=2)
        ax.annotate(f'ee = {ee} (P95)', xy=(ee + 0.5, density.max() * 0.85),
                    fontsize=10, color=color, fontweight='bold')
    ax.set_xlabel('Inter Top-K Admission Gap (Steps)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} — Useful Gap Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'visualizations/calibration_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
