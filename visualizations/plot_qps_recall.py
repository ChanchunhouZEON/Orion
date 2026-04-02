"""
QPS vs Recall curve for StagedDiskANN epsilon sweep.

Each curve = one window_size; each point = one epsilon value.
The epsilon=0 point (pure greedy, equivalent to DiskANN Vamana) is
highlighted with a dashed reference line so the speed-up is visible.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data from benchmark run (SIFT-10k, search_list_size=48, k=10) ──────────
# Format: (epsilon, qps, r@10)
DATA = {
    "ws=5": [
        (0.000,  27928.7, 0.9980),
        (0.001,  26613.2, 0.9980),
        (0.005,  32773.3, 0.9955),
        (0.010,  36258.8, 0.9900),
        (0.050,  40999.2, 0.9533),
        (0.100,  39460.1, 0.9204),
        (0.500,  49161.1, 0.8352),
        (1.000,  51299.1, 0.8021),
    ],
    "ws=10": [
        (0.000,  27713.9, 0.9980),
        (0.001,  27781.9, 0.9980),
        (0.005,  31027.0, 0.9980),
        (0.010,  29883.7, 0.9980),
        (0.050,  35178.1, 0.9895),
        (0.100,  39440.0, 0.9781),
        (0.500,  43579.2, 0.9377),
        (1.000,  46663.8, 0.9266),
    ],
    "ws=20": [
        (0.000,  30862.3, 0.9980),
        (0.001,  29085.2, 0.9980),
        (0.005,  31028.2, 0.9980),
        (0.010,  31301.2, 0.9980),
        (0.050,  31868.6, 0.9977),
        (0.100,  34362.2, 0.9951),
        (0.500,  38430.9, 0.9832),
        (1.000,  38289.6, 0.9810),
    ],
    "ws=50": [
        (0.000,  30507.0, 0.9980),
        (0.001,  28775.9, 0.9980),
        (0.005,  29602.1, 0.9980),
        (0.010,  29722.4, 0.9980),
        (0.050,  29985.8, 0.9980),
        (0.100,  29849.8, 0.9980),
        (0.500,  28743.6, 0.9978),
        (1.000,  29553.7, 0.9978),
    ],
}

COLORS   = ["#e05263", "#f5a623", "#4a90d9", "#7bc67e"]
MARKERS  = ["o", "s", "^", "D"]
EPSILONS = [0.000, 0.001, 0.005, 0.010, 0.050, 0.100, 0.500, 1.000]

# DiskANN Vamana baseline: the epsilon=0 points all cluster around 28-31k QPS
BASELINE_QPS   = np.mean([27928.7, 27713.9, 30862.3, 30507.0])
BASELINE_R10   = 0.9980

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for (label, points), color, marker in zip(DATA.items(), COLORS, MARKERS):
    xs = [p[2] for p in points]   # R@10
    ys = [p[1] for p in points]   # QPS
    eps_vals = [p[0] for p in points]

    ax.plot(xs, ys, color=color, marker=marker,
            linewidth=2, markersize=7, label=label, zorder=3)

    # Annotate each point with its epsilon value (skip 0.0 — handled by baseline)
    for x, y, eps in zip(xs, ys, eps_vals):
        if eps == 0.0:
            continue
        ax.annotate(
            f"ε={eps}",
            xy=(x, y), xytext=(4, 4), textcoords="offset points",
            fontsize=7, color=color, alpha=0.85,
        )

# DiskANN baseline reference lines
ax.axhline(BASELINE_QPS, color="grey", linestyle="--", linewidth=1.2,
           label=f"DiskANN baseline (~{BASELINE_QPS/1000:.0f}k QPS, ε=0)")
ax.axvline(BASELINE_R10, color="grey", linestyle=":",  linewidth=1.0)

# ── Axes & labels ────────────────────────────────────────────────────────────
ax.set_xlabel("R@10  (recall at 10)", fontsize=12)
ax.set_ylabel("QPS  (queries / second)", fontsize=12)
ax.set_title(
    "QPS vs Recall  —  StagedDiskANN epsilon sweep\n"
    "SIFT-10k · search_list_size=48 · k=10  "
    "(higher-right = better)",
    fontsize=11,
)

ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

ax.set_xlim(0.78, 1.01)
ax.set_ylim(0, 58000)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=10, loc="upper left")

plt.tight_layout()
out = "visualizations/qps_vs_recall.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
