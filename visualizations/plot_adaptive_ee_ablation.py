#!/usr/bin/env python3
"""Same-L k=10 NDC reductions. Actual recall is retained in companion JSON."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from chart_style import PALETTE, make_legend, rounded_bars, style_ax, style_fig, save_png_and_pdf

ARMS = ("full", "adaptive", "ee_only", "adaptive_ee")
LABELS = ("Adaptive", "EE only", "Adaptive + EE")
COLORS = tuple(PALETTE[c] for c in ("green", "purple", "orange"))


def select(path, widths):
    with path.open() as source:
        data = json.load(source)
    if data.get("experiment") != "adaptive_ee_ablation" or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected adaptive_ee_ablation schema 1")
    if data["config"]["sweep"]["k"] != 10:
        raise ValueError(f"{path}: only k=10 is supported")
    points = {}
    for point in data["points"]:
        if point["l"] in points:
            raise ValueError(f"{path}: duplicate L")
        arms = point["arms"]
        if len(arms) != 4 or {a["arm"] for a in arms} != set(ARMS):
            raise ValueError(f"{path}: expected four distinct arms")
        records = {}
        for arm in arms:
            ndc = arm["diagnostic_means"]["total_stage_ndc"]
            recall = arm["recall"]
            if not math.isfinite(ndc) or ndc <= 0 or not 0 <= recall <= 1:
                raise ValueError(f"{path}: invalid NDC or recall")
            records[arm["arm"]] = {"ndc": ndc, "recall": recall}
        for record in records.values():
            record["reduction_pct"] = 100 * (1 - record["ndc"] / records["full"]["ndc"])
        points[point["l"]] = {"l": point["l"], "arms": records}
    missing = set(widths) - points.keys()
    if missing:
        raise ValueError(f"{path}: missing L values {sorted(missing)}")
    return {"source": str(path.resolve()), "dataset": data["config"]["dataset"],
            "query_count": data["query_count"], "diagnostic_count": len(data["diagnostic_query_ids"]),
            "points": [points[l] for l in widths]}


def render(summaries, output):
    fig, axes = plt.subplots(1, len(summaries), squeeze=False,
                             figsize=(7.6 * len(summaries), 3.8), sharey=True)
    style_fig(fig)
    values = [r["reduction_pct"] for s in summaries for p in s["points"] for r in p["arms"].values()]
    lower, upper = min(0, min(values) * 1.25), max(5, max(values) * 1.30)
    for ax, summary in zip(axes[0], summaries):
        x = np.arange(len(summary["points"])) * 1.5
        width, gap = .26, .065
        handles = []
        for i, (arm, label, color) in enumerate(zip(ARMS[1:], LABELS, COLORS)):
            positions = x + (i - 1) * (width + gap)
            heights = [p["arms"][arm]["reduction_pct"] for p in summary["points"]]
            # Keep regressions below zero visible; RoundedTopBar is for positive bars.
            positive = [(p, h) for p, h in zip(positions, heights) if h >= 0]
            handles.append(rounded_bars(ax, [p for p, _ in positive],
                                        [h for _, h in positive], width, color, label=label))
            for position, height in zip(positions, heights):
                if height < 0:
                    ax.bar(position, height, width=width, color=color, alpha=.9, zorder=3)
                ax.annotate(f"{height:.1f}%", (position, height),
                            xytext=(0, 4 if height >= 0 else -4), textcoords="offset points",
                            ha="center", va="bottom" if height >= 0 else "top",
                            fontsize=8, color=PALETTE["annot"])
        style_ax(ax)
        ax.axhline(0, color=PALETTE["grey"], linewidth=.8)
        ax.set_xticks(x, [str(p["l"]) for p in summary["points"]])
        ax.set_xlim(x[0] - .8, x[-1] + .8)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Beam width L (same-L comparison)", fontsize=11)
        name = {"sift": "SIFT", "gist": "GIST"}.get(summary["dataset"], summary["dataset"])
        ax.set_title(f"{name} (k=10)", fontsize=13, fontweight="bold")
        make_legend(ax, handles, loc="upper left", ncol=3, fontsize=9,
                    handlelength=1.4, columnspacing=1.5)
    axes[0, 0].set_ylabel("NDC reduction vs Full\n(higher is better)", fontsize=11)
    fig.tight_layout(w_pad=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in save_png_and_pdf(fig, str(output), pdf_font_scale=1, dpi=200):
        print(f"Saved {path}")
    plt.close(fig)
    companion = output.with_suffix('.json')
    companion.write_text(json.dumps({
        "comparison": "Same L, not matched recall; actual recall retained per arm",
        "reduction": "100 * (1 - NDC / Full at the same L); negative means more NDC",
        "ndc": "Sum of stage distance evaluations, not equal-cost full-precision operations",
        "datasets": summaries,
    }, indent=2) + '\n')
    print(f"Saved {companion}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--ls", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "adaptive_ee_ndc_same_l_k10.png")
    args = parser.parse_args()
    if args.output.suffix != '.png':
        parser.error('--output must end in .png; PDF and selected-point JSON are also written')
    if len(set(args.ls)) != len(args.ls) or any(l <= 0 for l in args.ls):
        parser.error('--ls must contain distinct positive values')
    try:
        render([select(p, sorted(args.ls)) for p in args.inputs], args.output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
