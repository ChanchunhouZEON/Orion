#!/usr/bin/env python3
"""Plot k=10 adaptive-search NDC versus recall from existing results.

Only plotting is performed. Ratios compare mean summed stage evaluations at
the same L, not equal recall or equal-cost full-precision distance operations.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chart_style import (
    PALETTE, make_legend, rounded_bars, save_png_and_pdf, style_ax, style_fig,
)

MODES = ("full_neighbor", "local_only", "local_extra")
LABELS = ("Local + remote", "Local only", "Local + extra")
COLORS = (PALETTE["blue"], PALETTE["green"], PALETTE["orange"])


def load_summary(path, widths):
    with path.open() as source:
        data = json.load(source)
    if data.get("schema_version") != 1 or data.get("early_exit") is not False:
        raise ValueError("Expected schema_version=1 and early_exit=false")
    if data["config"]["sweep"]["k"] != 10:
        raise ValueError("This figure accepts only k=10 results")
    points = {}
    for point in data["points"]:
        if point["l"] in points:
            raise ValueError("Duplicate L in input")
        arms = point["arms"]
        if len(arms) != 3 or {a["mode"] for a in arms} != set(MODES):
            raise ValueError("Expected exactly three neighbor modes")
        points[point["l"]] = {
            a["mode"]: (a["diagnostic_means"]["total_stage_ndc"], a["recall"])
            for a in arms
        }
    if set(widths) - points.keys():
        raise ValueError(f"Missing requested L values: {set(widths) - points.keys()}")
    rows = [points[l] for l in widths]
    for row in points.values():
        for ndc, recall in row.values():
            if not np.isfinite(ndc) or ndc <= 0 or not 0 <= recall <= 1:
                raise ValueError("Invalid NDC or recall")
    return {
        "name": data["config"].get("dataset") or path.stem,
        "queries": data["query_count"],
        "diagnostics": len(data["diagnostic_query_ids"]),
        "rows": rows,
        "sweep": [points[l] for l in sorted(points)],
    }


def render(summaries, widths, output):
    fig, axes = plt.subplots(1, len(summaries), squeeze=False,
                             figsize=(6.3 * len(summaries), 5.5), sharey=True)
    style_fig(fig)
    x = np.arange(len(widths)) * 1.15
    width = 0.27
    maximum = 1.0
    for ax, summary in zip(axes[0], summaries):
        handles = []
        rows = summary["rows"]
        for i, (mode, label, color) in enumerate(zip(MODES, LABELS, COLORS)):
            ratios = [row[mode][0] / row[MODES[0]][0] for row in rows]
            maximum = max(maximum, max(ratios))
            positions = x + (i - 1) * width
            handles.append(rounded_bars(ax, positions, ratios, width * 0.88,
                                        color, label=label))
            for position, ratio in zip(positions, ratios):
                ax.annotate(f"{ratio:.2f}", (position, ratio),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9,
                            color=PALETTE["annot"])
        ax.axhline(1, color=PALETTE["grey"], linestyle="--", alpha=0.6,
                   linewidth=1, zorder=1)
        ax.set_xticks(x, [str(l) for l in widths])
        ax.set_xlabel("Beam width L", fontsize=11)
        name = {"sift": "SIFT", "gist": "GIST"}.get(summary["name"], summary["name"])
        ax.set_title(f"{name} (k=10)", fontweight="bold", fontsize=13)
        ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
        style_ax(ax)
        make_legend(ax, handles, fontsize=9, loc="upper center", ncol=3,
                    columnspacing=1.1, handlelength=1.4)
        print(f"{name}: {summary['diagnostics']}/{summary['queries']} diagnostic queries")
        for l, row in zip(widths, rows):
            print(f"  L={l}: " + "; ".join(
                f"{mode}: NDC={row[mode][0]:.2f}, recall={row[mode][1]:.5f}"
                for mode in MODES))
    axes[0, 0].set_ylabel("NDC ratio to local + remote (lower is better)", fontsize=11)
    axes[0, 0].set_ylim(0, maximum * 1.30)
    fig.tight_layout(w_pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Use the same type sizes in both formats to keep grouped labels legible.
    for path in save_png_and_pdf(fig, str(output), pdf_font_scale=1.0):
        print(f"Saved {path}")
    plt.close(fig)


def render_recall(summaries, output, minimum):
    fig, axes = plt.subplots(1, len(summaries), squeeze=False,
                             figsize=(7 * len(summaries), 3.9))
    style_fig(fig)
    for ax, summary in zip(axes[0], summaries):
        visible_ndc = []
        for mode, label, color, marker, linestyle in zip(
            MODES, LABELS, COLORS, ("o", "^", "s"), ("-", "--", "-.")
        ):
            values = [row[mode] for row in summary["sweep"]]
            ndc, recall = np.array(values).T
            # Preserve L order and actual coordinates, including recall reversals.
            ax.plot(recall, ndc, color=color, label=label, linewidth=2.2,
                    linestyle=linestyle, marker=marker, markersize=4.5,
                    markerfacecolor="white", markeredgewidth=1.1)
            visible_ndc.extend(ndc[recall >= minimum])
            # Retain the segment crossing the left edge when setting limits.
            for i in range(len(values) - 1):
                if min(recall[i:i+2]) <= minimum <= max(recall[i:i+2]):
                    visible_ndc.extend(ndc[i:i+2])
        if not visible_ndc:
            raise ValueError(f"No measurements at recall >= {minimum}: {summary['name']}")
        name = {"sift": "SIFT", "gist": "GIST"}.get(summary["name"], summary["name"])
        ax.set_title(f"{name} (k=10)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall@10", fontsize=11)
        ax.set_ylabel("Mean NDC / query (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.set_xlim(minimum, 1.0001)
        ax.set_ylim(min(visible_ndc) / 1.15, max(visible_ndc) * 1.2)
        ax.set_xticks(np.linspace(minimum, 1, 6))
        style_ax(ax)
        ax.legend(fontsize=9, loc="upper left", facecolor="white",
                  edgecolor="#E0E0E0", labelspacing=0.9, handlelength=3,
                  borderpad=0.8)
    fig.tight_layout(w_pad=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in save_png_and_pdf(fig, str(output), pdf_font_scale=1.0, dpi=200):
        print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--ls", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument("--kind", choices=("recall", "bar"), default="recall")
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--zoom-recall", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        name = "adaptive_ndc_vs_recall_k10.png" if args.kind == "recall" else "adaptive_ndc_k10.png"
        args.output = Path(__file__).parent / name
    if not 0 <= args.min_recall < args.zoom_recall < 1:
        parser.error("Expected 0 <= --min-recall < --zoom-recall < 1")
    if args.output.suffix != ".png":
        parser.error("--output must have a .png suffix; a PDF is also saved")
    if len(set(args.ls)) != len(args.ls) or any(l <= 0 for l in args.ls):
        parser.error("--ls must contain distinct positive values")
    try:
        summaries = [load_summary(path, args.ls) for path in args.inputs]
        if args.kind == "bar":
            render(summaries, args.ls, args.output)
        else:
            render_recall(summaries, args.output, args.min_recall)
            render_recall(summaries, args.output.with_stem(args.output.stem + "_zoom"),
                          args.zoom_recall)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
