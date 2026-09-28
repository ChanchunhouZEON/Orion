#!/usr/bin/env python3
"""Plot schema-v1 adaptive benchmark JSON without mixing runs or k values.

Usage:
  python visualizations/plot_adaptive.py visualizations/adaptive_sift_k10.json
  python visualizations/plot_adaptive.py /path/to/result.json --detail-l 256
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chart_style import PALETTE, save_png_and_pdf, style_ax

MODES = ("full_neighbor", "local_only", "local_extra")
LABELS = ("Local + remote", "Local only", "Local + extra")
COLORS = (PALETTE["blue"], PALETTE["green"], PALETTE["orange"])
MARKERS = ("o", "^", "s")
PAIRS = (
    ("full_vs_local", "Local only - (local + remote)", PALETTE["blue"]),
    ("local_vs_extra", "(Local + extra) - local only", PALETTE["green"]),
    ("full_vs_extra", "(Local + extra) - (local + remote)", PALETTE["yellow"]),
)
STAGES = (
    ("admission_ndc", "Admission", PALETTE["blue"]),
    ("prefilter_candidate_ndc", "Prefilter: candidates", PALETTE["yellow"]),
    ("prefilter_threshold_ndc", "Prefilter: threshold", PALETTE["purple"]),
    ("rerank_ndc", "Rerank", PALETTE["green"]),
)


def load_result(path):
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("early_exit") is not False:
        raise ValueError("Expected schema_version=1 and early_exit=false")
    points = sorted(data["points"], key=lambda p: p["l"])
    if not points or len({p["l"] for p in points}) != len(points):
        raise ValueError("Expected nonempty points with unique L values")
    for point in points:
        arms = {a["mode"]: a for a in point["arms"]}
        if len(point["arms"]) != 3 or set(arms) != set(MODES):
            raise ValueError(f"L={point['l']}: expected exactly three neighbor modes")
        point["by_mode"] = arms
        for arm in arms.values():
            if not np.isfinite(arm["qps_median"]) or arm["qps_median"] <= 0:
                raise ValueError("QPS must be finite and positive")
            if not 0 <= arm["recall"] <= 1:
                raise ValueError("Recall must be in [0, 1]")
            for row in arm["queries"]:
                if len(row["first_discovery"]) != len(row["target_ids"]):
                    raise ValueError("Discovery times must align with target IDs")
    return data, points


def finish(fig, title, footnote, output):
    fig.suptitle(title, fontsize=13)
    fig.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=8,
             color=PALETTE["annot"])
    fig.tight_layout(rect=(0, 0.065, 1, 0.94), h_pad=2, w_pad=2)
    # Avoid the default PDF font enlargement colliding in multi-panel figures.
    png, pdf = save_png_and_pdf(fig, str(output), pdf_font_scale=1.0)
    plt.close(fig)
    print(f"Saved {png}\n      {pdf}")


def line(ax, x, y, arm, label=None):
    ax.plot(x, y, color=COLORS[arm], marker=MARKERS[arm], markersize=4,
            linewidth=1.8, label=label or LABELS[arm])


def format_ax(ax, title, xlabel, ylabel, *, fraction=False):
    style_ax(ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if fraction:
        ax.set_ylim(0, 1.03)


def overview(points, k, title, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axs = axes.ravel()
    ls = [p["l"] for p in points]
    for arm, mode in enumerate(MODES):
        rows = [p["by_mode"][mode] for p in points]
        # Keep L order: show measured sweep, not an invented monotone frontier.
        line(axs[0], [r["recall"] for r in rows], [r["qps_median"] for r in rows], arm)
        line(axs[1], ls, [r["recall"] for r in rows], arm)
        for ax, key in zip(axs[2:5], ("total_stage_ndc", "expansions", "discovery_fraction")):
            line(ax, ls, [r["diagnostic_means"][key] for r in rows], arm)
        switched = [
            np.mean([q["first_switch_step"] is not None for q in r["queries"]])
            if r["queries"] else np.nan for r in rows
        ]
        line(axs[5], ls, switched, arm)
    format_ax(axs[0], "Measured QPS-recall sweep", f"Recall@{k}", "QPS (log scale)")
    axs[0].set_yscale("log")
    axs[0].legend(fontsize=8)
    format_ax(axs[1], "Final recall", "Beam width L", f"Recall@{k}", fraction=True)
    format_ax(axs[2], "Stage distance evaluations", "Beam width L", "Mean evaluations / query")
    format_ax(axs[3], "Expanded vertices", "Beam width L", "Mean expansions / query")
    format_ax(axs[4], "Ground-truth targets encountered", "Beam width L", "Discovered fraction", fraction=True)
    format_ax(axs[5], "Queries entering converged mode", "Beam width L", "Query fraction", fraction=True)
    finish(fig, title,
           "QPS and final recall: all queries. Work and discovery: diagnostic sample. EE disabled.",
           out)


def detail(point, k, title, out):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    rows = [point["by_mode"][mode] for mode in MODES]
    x = np.arange(3)
    bottom = np.zeros(3)
    for key, label, color in STAGES:
        values = np.array([r["diagnostic_means"][key] for r in rows])
        if np.any(values):
            axs[0].bar(x, values, bottom=bottom, width=0.6, color=color, label=label)
        bottom += values
    axs[0].legend(fontsize=7)
    bottom = np.zeros(3)
    for key, label, color in (
        ("pre_expansions", "Navigation", PALETTE["blue"]),
        ("post_expansions", "Converged", PALETTE["yellow"]),
    ):
        values = np.array([r["diagnostic_means"][key] for r in rows])
        axs[1].bar(x, values, bottom=bottom, width=0.6, color=color, label=label)
        bottom += values
    axs[1].legend(fontsize=8)
    end = max((q["expansions"] for r in rows for q in r["queries"]), default=1)
    for arm, r in enumerate(rows):
        times = [t for q in r["queries"] for t in q["first_discovery"] if t is not None]
        denominator = sum(len(q["target_ids"]) for q in r["queries"])
        if denominator:
            steps, counts = np.unique(times, return_counts=True)
            xs = np.r_[0, steps, end]
            ys = np.r_[0, np.cumsum(counts) / denominator, len(times) / denominator]
            axs[2].step(xs, ys, where="post", color=COLORS[arm], label=LABELS[arm], linewidth=1.8)
    axs[2].legend(fontsize=8)
    for ax in axs[:2]:
        ax.set_xticks(x, ("Local +\nremote", "Local\nonly", "Local +\nextra"))
    format_ax(axs[0], "Stage NDC breakdown", "", "Mean evaluations / query")
    format_ax(axs[1], "Expansion phases", "", "Mean expansions / query")
    format_ax(axs[2], f"Discovery of ground-truth top-{k}", "Expansion step (entry = 0)",
              "Fraction of all query-target pairs", fraction=True)
    finish(fig, title + f" | L={point['l']}",
           "Diagnostic sample. Unfound targets remain in the discovery denominator; discovery is before prefilter.",
           out)


def paired(points, title, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    ls = [p["l"] for p in points]
    for key, label, color in PAIRS:
        pairs = [p["paired_discovery"][key] for p in points]
        delta = [
            p["right_mean_step"] - p["left_mean_step"]
            if p["common_targets"] and p["right_mean_step"] is not None
            and p["left_mean_step"] is not None else np.nan
            for p in pairs
        ]
        axs[0].plot(ls, delta, marker="o", color=color, label=label)
        axs[1].plot(ls, [p["common_targets"] for p in pairs], marker="o", color=color, label=label)
    axs[0].axhline(0, color=PALETTE["grey"], linestyle="--", linewidth=1)
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    format_ax(axs[0], "Paired first-discovery step difference", "Beam width L",
              "Step difference (negative = first arm earlier)")
    format_ax(axs[1], "Commonly discovered query-target pairs", "Beam width L", "Pair count")
    finish(fig, title,
           "Each comparison uses its own common-target intersection. Missing comparisons are gaps, not zero.",
           out)


def matched(data, title, out):
    targets = data.get("matched_recall", [])
    if not targets:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 2.2), 5.2))
    width = 0.23
    for arm, mode in enumerate(MODES):
        for i, target in enumerate(targets):
            records = {a["mode"]: a["best_measured"] for a in target["arms"]}
            best = records.get(mode)
            x = i + (arm - 1) * width
            if best is None:
                ax.text(x, 0.03, "Not reached", rotation=90, ha="center", va="bottom",
                        transform=ax.get_xaxis_transform(), fontsize=7, color=COLORS[arm])
            else:
                ax.bar(x, best["qps"], width=width * 0.9, color=COLORS[arm])
                ax.annotate(f"L={best['l']}\nR={best['recall']:.4f}",
                            (x, best["qps"]), xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)], fontsize=8)
    ax.set_xticks(range(len(targets)), [f"{t['target']:.3f}" for t in targets])
    ax.set_xlim(-0.6, len(targets) - 0.4)
    values = [a["best_measured"]["qps"] for t in targets for a in t["arms"]
              if a["best_measured"] is not None]
    ax.set_ylim(0, max(values, default=1) * 1.35)
    if not values:
        ax.set_yticks([])
        ax.text(0.5, 0.55, "No arm reached the requested recall targets",
                transform=ax.transAxes, ha="center", fontsize=10, color=PALETTE["annot"])
    format_ax(ax, "Best measured QPS meeting each recall target", "Minimum recall target", "QPS")
    finish(fig, title,
           "Actual achieved recall and L are labeled. No interpolation; an unmet target is not a zero-QPS result.",
           out)


def render(path, out_dir, detail_l):
    data, points = load_result(path)
    config = data["config"]
    sweep = config["sweep"]
    k = sweep["k"]
    name = config.get("dataset") or path.stem
    title = (f"{name} | N={data['num_points']:,}, k={k}, {sweep['threads']} threads"
             f" | diagnostic n={len(data['diagnostic_query_ids'])}")
    chosen = points[-1] if detail_l is None else next((p for p in points if p["l"] == detail_l), None)
    if chosen is None:
        raise ValueError(f"L={detail_l} not present; available: {[p['l'] for p in points]}")
    out_dir = out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, draw in (
        ("overview", lambda out: overview(points, k, title, out)),
        (f"detail_L{chosen['l']}", lambda out: detail(chosen, k, title, out)),
        ("paired_discovery", lambda out: paired(points, title, out)),
        ("matched_recall", lambda out: matched(data, title, out)),
    ):
        draw(out_dir / f"{path.stem}_{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more adaptive JSON results")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--detail-l", type=int, help="Detailed panel L (default: largest measured L)")
    args = parser.parse_args()
    if args.out_dir and len({p.stem for p in args.inputs}) != len(args.inputs):
        parser.error("Inputs sharing a filename would overwrite outputs in --out-dir")
    for path in args.inputs:
        try:
            render(path, args.out_dir, args.detail_l)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(2, f"{path}: {error}\n")


if __name__ == "__main__":
    main()
