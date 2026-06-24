#!/usr/bin/env python3
"""Build overhead at matched α: trial-range box (left) + extras % bars (right).

Both engines build at the dataset's PA-aligned `scfg.alpha`; every build
param (R / L_build / α / num_threads / metric) is identical and the only
delta is the `compute_candidate_sets` flag that drives the per-node
60/40 partition pass.

  * **Left** — per-dataset trial-range box at the propagated σ around
    the `Orion_total / DiskANN` ratio. No baseline bar — the box itself
    *is* the range. Rounded both ends, filled in the dataset's palette
    colour; a thin horizontal tick marks the mean. Reference line at
    y = 1.0 makes the parity claim land at a glance.
  * **Right** — `orion_overhead / DiskANN` ratio as a percentage,
    rounded bars in the same per-dataset palette colour. The headline
    is "sub-1 % on every dataset"; a y = 1 % reference line gives the
    eye the threshold for free.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(__file__))
from chart_style import *

datasets = [
    ("sift",    "SIFT"), ("glove25", "GloVe-25"),
    ("glove100","GloVe-100"), ("gist",    "GIST"),
]

def load_build(name):
    bp = f"visualizations/build_profile_{name}.json"
    if os.path.exists(bp):
        with open(bp) as f:
            d = json.load(f)
        return {
            'diskann_s': d['diskann_s'],
            'diskann_s_std': d.get('diskann_s_std', 0.0),
            'orion_graph_s': d['orion_graph_s'],
            'orion_graph_s_std': d.get('orion_graph_s_std', 0.0),
            'orion_overhead_s': d['orion_overhead_s'],
            'orion_overhead_s_std': d.get('orion_overhead_s_std', 0.0),
            'alpha': d.get('alpha'),
        }
    return None

diags = {}
for name, _ in datasets:
    b = load_build(name)
    if b is not None:
        diags[name] = {'build': b}

present = [(n, t) for n, t in datasets if n in diags]
short_labels = [
    f"{t}\nα={diags[n]['build'].get('alpha', '?'):.2f}" if isinstance(diags[n]['build'].get('alpha'), (int, float)) else t
    for n, t in present
]

# Paper figure: single panel — the left "Builds Match Within Trial
# Noise" ratio plot. The right "Extras Pass Is Sub-1%" subplot was
# useful in earlier iterations but the same conclusion is implicit
# in the left panel's near-1.0 ratio (Orion/DiskANN build ≈ 1×),
# so we drop it here to keep the paper figure dense.
fig, ax_left = plt.subplots(1, 1, figsize=(7, 5.5))
axes = [ax_left, None]
style_fig(fig)

x = np.arange(len(present))
ds_colors = [DATASET_COLORS.get(n, PALETTE['grey']) for n, _ in present]

da_mean = np.array([diags[n]['build']['diskann_s'] for n, _ in present])
da_std  = np.array([diags[n]['build']['diskann_s_std'] for n, _ in present])
sg_mean = np.array([diags[n]['build']['orion_graph_s'] for n, _ in present])
sg_std  = np.array([diags[n]['build']['orion_graph_s_std'] for n, _ in present])
so_mean = np.array([diags[n]['build']['orion_overhead_s'] for n, _ in present])


class TrialRangeBox(PathPatch):
    """Rectangle with rounded corners on BOTH top and bottom — visual
    twin of `chart_style.RoundedTopBar`, just extended to round all four
    corners. Recomputes its path at draw time so the corner radius in
    pixels matches the right-panel rounded-top bars (`radius_frac=0.3`
    of the box's pixel width, clamped to 14 px). Quarter-ellipses in
    data coords land as quarter-circles on screen because we project rx
    and ry independently through `transData.inverted()`."""

    def __init__(self, ax, x_center, y_lo, y_hi, *, width=0.5,
                 facecolor, alpha=0.85, radius_frac=0.3, zorder=3):
        self._b_ax = ax
        self._b_x = x_center
        self._b_w = width
        self._b_lo = y_lo
        self._b_hi = y_hi
        self._radius_frac = radius_frac
        left, right = x_center - width / 2, x_center + width / 2
        placeholder = Path(
            [(left, y_lo), (right, y_lo), (right, y_hi), (left, y_hi), (left, y_lo)],
            [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
        )
        super().__init__(
            placeholder, facecolor=facecolor, edgecolor='none',
            alpha=alpha, zorder=zorder, linewidth=0,
        )

    def draw(self, renderer):
        self._recompute_path()
        super().draw(renderer)

    def _recompute_path(self):
        ax = self._b_ax
        x = self._b_x
        w = self._b_w
        bottom, top = self._b_lo, self._b_hi
        if top - bottom <= 0:
            return
        trans = ax.transData
        inv = trans.inverted()

        # Pixel-space radius (same convention as RoundedTopBar) so the
        # corner curvature visually matches the right-panel bars.
        p_left  = trans.transform((x - w / 2, bottom))
        p_right = trans.transform((x + w / 2, bottom))
        bar_width_px = abs(p_right[0] - p_left[0])
        radius_px = min(bar_width_px * self._radius_frac, 14.0)

        # Pull radius_px in each axis direction back to data coords.
        p_origin = trans.transform((x, bottom))
        px_data = inv.transform((p_origin[0] + radius_px, p_origin[1]))
        py_data = inv.transform((p_origin[0], p_origin[1] + radius_px))
        rx = min(abs(px_data[0] - x), w / 2)
        ry = min(abs(py_data[1] - bottom), (top - bottom) / 2)

        left, right = x - w / 2, x + w / 2
        verts = [
            (left,       bottom + ry),                  # start mid-left edge
            (left,       bottom), (left + rx, bottom),  # bottom-left corner
            (right - rx, bottom),                       # bottom edge
            (right,      bottom), (right,     bottom + ry),  # bottom-right corner
            (right,      top - ry),                     # right edge
            (right,      top),    (right - rx, top),    # top-right corner
            (left + rx,  top),                          # top edge
            (left,       top),    (left,       top - ry),  # top-left corner
            (left,       bottom + ry),                  # close on mid-left edge
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE3, Path.CURVE3,
            Path.LINETO,
            Path.CURVE3, Path.CURVE3,
            Path.LINETO,
            Path.CURVE3, Path.CURVE3,
            Path.LINETO,
            Path.CURVE3, Path.CURVE3,
            Path.CLOSEPOLY,
        ]
        self.set_path(Path(verts, codes))


def trial_range_box(ax, x_center, y_lo, y_hi, color, *, width=0.5,
                    alpha=0.85, mean_tick=None):
    """Render a rounded-corner σ-range box (matches right-panel rounded
    bars in pixel-space curvature)."""
    patch = TrialRangeBox(ax, x_center, y_lo, y_hi,
                          width=width, facecolor=color, alpha=alpha)
    ax.add_patch(patch)
    if mean_tick is not None:
        left = x_center - width / 2
        ax.plot([left + width * 0.22, left + width * 0.78],
                [mean_tick, mean_tick],
                color='white', linewidth=2.0, solid_capstyle='round',
                zorder=4)


# ── Left: trial-range boxes around the Orion/DiskANN ratio (no bars). ──
ax = axes[0]
st_total      = sg_mean + so_mean
ratio         = st_total / da_mean
ratio_std     = ratio * np.sqrt((sg_std / sg_mean) ** 2 + (da_std / da_mean) ** 2)

for xi, r, s, c in zip(x, ratio, ratio_std, ds_colors):
    trial_range_box(ax, xi, r - s, r + s, c, mean_tick=r)

ax.axhline(y=1.0, color='#888', linestyle='--', alpha=0.7, linewidth=1.2)
ax.annotate('parity', xy=(x[-1] + 0.4, 1.0), va='center', fontsize=9,
            color=PALETTE['annot'])

for i, (r, s) in enumerate(zip(ratio, ratio_std)):
    ax.annotate(f'{r:.3f}×\n±{s:.3f}', xy=(x[i], r + s + 0.005),
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=PALETTE['annot'])

# Per-dataset legend handles (rounded patches in matching palette).
handles_left = [rounded_patch(c, alpha=0.85, label=t)
                for (_, t), c in zip(present, ds_colors)]

ax.set_xticks(x); ax.set_xticklabels(short_labels)
ax.set_ylabel('Build-time ratio (Orion / DiskANN)\nat matched α', fontsize=11)
ax.set_title('Builds Match Within Trial Noise', fontweight='bold')
make_legend(ax, handles_left, fontsize=10)
style_ax(ax)
lo = min(0.94, float(np.min(ratio - ratio_std) - 0.015))
hi = max(1.08, float(np.max(ratio + ratio_std) + 0.04))
ax.set_ylim(lo, hi)
ax.set_xlim(x[0] - 0.6, x[-1] + 0.95)

# ── Right subplot dropped for the paper figure (see header comment). ──

plt.tight_layout()
out = 'visualizations/build_analysis.png'
save_png_and_pdf(fig, out)
print(f'Saved {out}')
