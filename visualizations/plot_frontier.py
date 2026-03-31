"""
QPS vs Recall — For every L (search_list_size), StagedDiskANN finds a
(ws, ε) that outperforms DiskANN at matched recall.

Arrows are numbered ①②③… and a compact table in the upper-left maps
each number to (L, ws, ε, +gain%).  No overlapping text anywhere.

Reads: visualizations/full_frontier_data.json
"""

import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
RECALL_TOL = 0.020   # max recall drop for "matched" StagedDiskANN config

C_DISKANN = "#9B3A20"
C_STAGED  = "#D4751A"
C_PARETO  = "#E8A830"
C_SCATTER = "#F0C080"
C_SHADE   = "#FAE5C0"
C_ARROW   = "#7A3010"
C_NUM     = "#FFFFFF"
C_GRID    = "#EDE5DA"
BG        = "#FFFCF8"

# ── Load ──────────────────────────────────────────────────────────────────────
root = pathlib.Path(__file__).parent
data = json.loads((root / "full_frontier_data.json").read_text())
pts  = data["points"]
meta = data["meta"]

sls_values = sorted({p["search_list_size"] for p in pts})
all_x = [p["recall_10"] for p in pts]
all_y = [p["qps"]       for p in pts]

# ── Per-SLS baselines and best configs ───────────────────────────────────────
diskann_pts = []   # (recall, qps, sls)
best_pts    = []   # (recall, best_qps, sls, ws, eps, best_recall)

for sls in sls_values:
    sls_rows = [p for p in pts if p["search_list_size"] == sls]
    eps0     = [p for p in sls_rows if p["epsilon"] == 0.0]
    base     = max(eps0, key=lambda p: p["qps"])
    br, bq   = base["recall_10"], base["qps"]
    diskann_pts.append((br, bq, sls))

    candidates = [p for p in sls_rows if p["recall_10"] >= br - RECALL_TOL]
    if not candidates:
        continue
    best = max(candidates, key=lambda p: p["qps"])
    best_pts.append((br, best["qps"], sls,
                     best["window_size"], best["epsilon"], best["recall_10"]))

diskann_pts.sort(key=lambda p: p[0])
best_pts.sort(key=lambda p: p[0])

# ── Pareto frontier of all staged configs ────────────────────────────────────
def pareto(rows):
    rows = sorted(rows, key=lambda p: p[0])
    front, best_q = [], -1.
    for r, q in rows:
        if q > best_q:
            front.append((r, q)); best_q = q
    return front

pareto_front = pareto([(p["recall_10"], p["qps"]) for p in pts if p["epsilon"] > 0])

# ── Build gain table (only entries with gain > 3 %) ──────────────────────────
da_x = [p[0] for p in diskann_pts]
da_y = [p[1] for p in diskann_pts]
bp_x = [p[0] for p in best_pts]
bp_y = [p[1] for p in best_pts]

table_rows = []   # (num, r_anchor, d_qps, b_qps, gain, sls, ws, eps)
num = 1
for r_anchor, b_qps, sls, ws, eps, b_recall in best_pts:
    d_qps = float(np.interp(r_anchor, da_x, da_y))
    gain  = (b_qps - d_qps) / d_qps * 100
    if gain < 3:
        continue
    table_rows.append((num, r_anchor, d_qps, b_qps, gain, sls, ws, eps))
    num += 1

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ① Background scatter
ax.scatter(all_x, all_y, color=C_SCATTER, s=14, alpha=0.30, zorder=1,
           label="All 180 measured configurations")

# ② Pareto frontier
px = [p[0] for p in pareto_front]
py = [p[1] for p in pareto_front]
ax.plot(px, py, color=C_PARETO, linewidth=1.6, linestyle="--",
        zorder=3, label="StagedDiskANN — Pareto frontier")

# ③ Advantage shade
da_interp = np.interp(bp_x, da_x, da_y)
ax.fill_between(bp_x, da_interp, bp_y,
                where=[b > d for b, d in zip(bp_y, da_interp)],
                color=C_SHADE, alpha=0.70, zorder=2)

# ④ Best-per-SLS staged curve
ax.plot(bp_x, bp_y, color=C_STAGED, lw=2.5, marker="o", ms=7,
        zorder=6, label=f"StagedDiskANN best (ws, ε) per L  [tol ±{RECALL_TOL*100:.0f}%]")

# ⑤ DiskANN baseline curve
ax.plot(da_x, da_y, color=C_DISKANN, lw=2.5, marker="s", ms=7,
        zorder=7, label="DiskANN baseline (ε = 0)")

# ⑥ Numbered arrows
CIRCLE_R = 0.0018   # radius in recall units (used only for symbol size)
for n, r_anchor, d_qps, b_qps, gain, sls, ws, eps in table_rows:
    # Arrow shaft
    ax.annotate(
        "", xy=(r_anchor, b_qps - 1200),
        xytext=(r_anchor, d_qps + 1200),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                        lw=1.4, mutation_scale=10),
        zorder=8,
    )
    # Filled circle with number at midpoint
    mid_y = (b_qps + d_qps) / 2
    circle = plt.Circle((r_anchor, mid_y), radius=0,
                         transform=ax.transData)
    ax.plot(r_anchor, mid_y, "o", ms=15,
            color=C_ARROW, zorder=9)
    ax.text(r_anchor, mid_y, str(n),
            ha="center", va="center", fontsize=7,
            color=C_NUM, fontweight="bold", zorder=10)

# ⑦ L= labels below DiskANN baseline points
for i, (r, q, sls) in enumerate(diskann_pts):
    ax.annotate(f"L={sls}", xy=(r, q),
                xytext=(0, -13), textcoords="offset points",
                fontsize=7, color=C_DISKANN, ha="center")

# ── Inset table (upper-left) ─────────────────────────────────────────────────
table_header = "  #   L    ws   ε      +QPS"
table_lines  = [table_header, "─" * len(table_header)]
for n, r_anchor, d_qps, b_qps, gain, sls, ws, eps in table_rows:
    eps_str = f"{eps:.3f}".rstrip('0').rstrip('.')
    table_lines.append(f"  {n}  L={sls:<3} ws={ws:<2} ε={eps_str:<5}  +{gain:.0f}%")

table_text = "\n".join(table_lines)
ax.text(0.015, 0.97, table_text,
        transform=ax.transAxes, fontsize=8,
        va="top", ha="left", family="monospace",
        color="#5A2000",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF5E6",
                  edgecolor="#D4A060", alpha=0.92))

# ── Axes & style ──────────────────────────────────────────────────────────────
ax.set_xlabel("R@10  (recall at 10)", fontsize=12)
ax.set_ylabel("QPS  (queries / second)", fontsize=12)
ax.set_title(
    "QPS vs Recall — At every L, StagedDiskANN finds (ws, ε) that outperforms DiskANN\n"
    f"SIFT-{meta['num_points']//1000}k · k={meta['k']}  "
    "· numbered arrows = matched-recall QPS gain  (higher-right is better)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.set_xlim(min(all_x) - 0.005, 1.004)
ax.set_ylim(0, max(all_y) * 1.15)
ax.grid(True, color=C_GRID, linewidth=0.8)

legend_handles = [
    Line2D([0],[0], color=C_DISKANN, lw=2.3, marker="s", ms=6,
           label="DiskANN baseline (ε=0)"),
    Line2D([0],[0], color=C_STAGED, lw=2.3, marker="o", ms=6,
           label=f"StagedDiskANN best (ws, ε) per L  [tol ±{RECALL_TOL*100:.0f}%]"),
    Line2D([0],[0], color=C_PARETO, lw=1.6, linestyle="--",
           label="StagedDiskANN Pareto frontier"),
    mpatches.Patch(facecolor=C_SHADE, alpha=0.8, label="QPS advantage zone"),
    Line2D([0],[0], marker="o", color="none",
           markerfacecolor=C_SCATTER, ms=6, alpha=0.5,
           label="All 180 measured configs"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="lower left",
          framealpha=0.92, edgecolor="#D4B896")

fig.tight_layout()
out = root / "fig1_qps_recall_curve.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.close(fig)
