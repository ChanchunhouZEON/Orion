# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Curate and plot request latency without rerunning an index or a search.

Input: frozen per-job result JSONs plus the verified USearch control analysis.
Output: portable public JSON, threshold tables, and chart_style PNG/PDF panels.
Native control points remain a distinct series, never an extension of USearch's
Python curve. Only sampling-qualified, ID-consistent measurements are plotted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

DATASETS = {
    "sift": "SIFT1M", "deep10m": "Deep10M", "gist": "GIST",
    "glove25": "GloVe-25", "glove100": "GloVe-100",
    "msmarco_bert_1M": "MS-MARCO", "wiki_ada_1M": "Wiki-Ada-1M",
}
ENGINES = {
    "orion": "Orion", "parlayann": "ParlayANN", "diskann": "DiskANN / Vamana",
    "hnswlib": "hnswlib", "usearch": "USearch Python (GIL)",
    "usearch-native": "USearch native (no GIL)",
    "faiss-hnsw-sq8": "FAISS HNSW-SQ8", "faiss-ivf-flat": "FAISS IVF-Flat",
    "faiss-ivf-pq": "FAISS IVF-PQ", "lancedb": "LanceDB IVF-HNSW-SQ", "annoy": "Annoy",
}
QUANTILES = ("p50_us", "p95_us", "p99_us")
TARGETS = (.90, .95, .98, .99)


def read_json(path):
    return json.loads(Path(path).read_text())


def qualified(point, protocol):
    rounds = point.get("rounds", [])
    good = (len(rounds) == protocol["rounds"] and all(
        r["minimum_reached"] and r["latency"]["requests"] >= protocol["min_requests"]
        and r["wall_seconds"] >= protocol["min_seconds"] for r in rounds))
    if good != point["complete"]:
        raise ValueError("Sampling status disagrees with recorded rounds")
    if not point["concurrent_ids_match"]:
        raise ValueError("Concurrent result ID verification failed")
    return good


def validate_latencies(row):
    values = [row[q] for q in QUANTILES]
    if not all(math.isfinite(v) and v > 0 for v in values) or values != sorted(values):
        raise ValueError("Invalid pooled latency percentiles")
    if not 0 <= row["recall"] <= 1:
        raise ValueError("Invalid recall")


def select(points, dataset, engine, concurrency, target):
    """One measured setting supplies all three percentiles; never interpolate."""
    candidates = [p for p in points if p["dataset"] == dataset and p["engine"] == engine
                  and p["concurrency"] == concurrency and p["recall"] >= target]
    return min(candidates, key=lambda p: (p["p50_us"], p["effort"])) if candidates else None


def aggregate(points, target, concurrency, engine):
    pairs = []
    for dataset in DATASETS:
        ours = select(points, dataset, "orion", concurrency, target)
        other = select(points, dataset, engine, concurrency, target)
        if ours and other:
            pairs.append({"dataset": dataset, "orion": ours, "baseline": other})
    ratios = {q: math.exp(sum(math.log(p["baseline"][q] / p["orion"][q])
                             for p in pairs) / len(pairs)) for q in QUANTILES} if pairs else None
    return {"target": target, "concurrency": concurrency, "engine": engine,
            "datasets": [p["dataset"] for p in pairs], "count": len(pairs), "ratios": ratios}


def curate(inputs, control):
    jobs = [read_json(p) for p in sorted(inputs.glob("*.json"))]
    jobs = [j for j in jobs if "engine" in j and "dataset" in j and "status" in j]
    if not jobs:
        raise ValueError("No per-job result JSONs in --inputs")
    valid = [j for j in jobs if j["status"] in ("complete", "incomplete")]
    for dataset in DATASETS:
        group = [j for j in valid if j["dataset"] == dataset]
        for field in ("base_identity", "query_sha256", "groundtruth_sha256", "protocol", "timing_scope"):
            if len({json.dumps(j[field], sort_keys=True) for j in group}) > 1:
                raise ValueError(f"Inconsistent {field} for {dataset}")
        if any(j["config"]["k"] != 10 for j in group):
            raise ValueError("This report requires k=10")
    points, excluded, coverage = [], [], []
    for job in jobs:
        coverage.append({"dataset": job["dataset"], "engine": job["engine"], "status": job["status"]})
        if job not in valid:
            continue
        for p in job["points"]:
            good = qualified(p, job["protocol"])
            if not p.get("pooled"):
                continue
            row = {"dataset": job["dataset"], "engine": job["engine"], "source": "main",
                   "effort": p["effort"], "concurrency": p["concurrency"],
                   "recall": p["evaluation"]["recall"],
                   **{q: p["pooled"][q] for q in QUANTILES},
                   "requests": p["pooled"]["requests"],
                   "round_requests": [r["latency"]["requests"] for r in p["rounds"]]}
            validate_latencies(row)
            (points if good else excluded).append(row)
    native = read_json(control)
    if any(c["status"] != "complete" or c["isa"] != "serial" for c in native["checks"]):
        raise ValueError("Revisit control protocol: expected completed serial-kernel control")
    control_rows = []
    for p in native["rows"]:
        if not p["complete"] or len(p["round_requests"]) != 3 or min(p["round_requests"]) < 10000:
            raise ValueError("Incomplete native control")
        row = {k: p[k] for k in ("dataset", "variant", "concurrency", "effort", "recall",
                                 "round_requests", "requests", "qps", *QUANTILES)}
        validate_latencies(row)
        control_rows.append(row)
        if p["variant"] == "native-nogil":
            points.append({**row, "engine": "usearch-native", "source": "native-control"})
    aggregates = [aggregate(points, t, c, e) for t in TARGETS for c in (1, 8)
                  for e in ENGINES if e != "orion"]
    # Whitelist fields: no local file paths, process IDs, or filesystem layout
    # escape into the public artifact. Hashes identify the exact source JSONs.
    sources = [{"id": p.stem, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(inputs.glob("*.json"))]
    sources.append({"id": "usearch-native-control", "sha256": hashlib.sha256(control.read_bytes()).hexdigest()})
    return {"schema": 1, "datasets": DATASETS, "engines": ENGINES, "targets": TARGETS,
            "protocol": {"hardware": "Apple M4 Max / macOS", "k": 10,
                         **valid[0]["protocol"],
                         "timing": "Python single-request adapter boundary, including binding and result assembly; not HTTP or pure-kernel latency",
                         "selection": "Minimum qualified pooled P50 at a common recall threshold; P95/P99 from the same measured setting; no interpolation",
                         "aggregation": "Geometric mean of baseline / Orion latency over listed common datasets; above 1 favors Orion",
                         "control": "Separate run, one effort per dataset, serial SIMD kernel; native no-GIL is not a complete recall curve",
                         "limitations": "Different processes and times; descriptive results, not significance-tested. Actual recall and coverage vary."},
            "job_counts": dict(Counter(j["status"] for j in jobs)),
            "point_counts": {"qualified": sum(p["source"] == "main" for p in points), "undersampled": len(excluded)},
            "coverage": coverage, "points": points, "excluded": excluded,
            "control": control_rows, "aggregates": aggregates, "sources": sources}


def write_table(data, output):
    lines = ["# Request latency", "", data["protocol"]["selection"] + ".",
             data["protocol"]["aggregation"] + ".", "",
             "| Recall ≥ | Clients | Baseline | Datasets | P50 | P95 | P99 |",
             "|---:|---:|---|---:|---:|---:|---:|"]
    for a in data["aggregates"]:
        if a["ratios"]:
            lines.append(f'| {a["target"]:.2f} | {a["concurrency"]} | {ENGINES[a["engine"]]} | {a["count"]} | '
                         + " | ".join(f'{a["ratios"][q]:.2f}×' for q in QUANTILES) + " |")
    (output / "latency-summary.md").write_text("\n".join(lines) + "\n")


def plot(data, output, engines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "visualizations"))
    from chart_style import PALETTE, PALETTE_VIVID, style_ax, style_fig, save_png_and_pdf
    # Reuse the QPS–recall panel's engine colors and marker identities.
    from plot_baseline_comparison import SERIES
    aliases = {"hnswlib": "hnsw", "usearch": "usearch_hnsw", "usearch-native": "usearch_hnsw",
               "faiss-hnsw-sq8": "faiss_hnsw_sq", "faiss-ivf-flat": "faiss_ivf_flat",
               "faiss-ivf-pq": "faiss_ivf_pq", "lancedb": "lancedb_hnsw"}
    styles = {row[0]: row for row in SERIES}
    colors = {e: styles[aliases.get(e, e)][2] for e in ENGINES}
    markers = {e: styles[aliases.get(e, e)][3] for e in ENGINES}
    data["colors"] = colors
    for concurrency in (1, 8):
        for quantile in QUANTILES:
            fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
            style_fig(fig)
            for ax, (ds, name) in zip(axes.flat, DATASETS.items()):
                style_ax(ax)
                for engine in engines:
                    rows = sorted([p for p in data["points"] if p["dataset"] == ds
                                   and p["concurrency"] == concurrency and p["engine"] == engine
                                   and p["recall"] >= .9], key=lambda p: (p["recall"], p["effort"]))
                    if rows:
                        ax.plot([p["recall"] for p in rows], [p[quantile] / 1000 for p in rows],
                                color=colors[engine], marker="D" if engine == "usearch-native" else markers[engine],
                                linestyle="none" if engine == "usearch-native" else "-",
                                markersize=5 if engine == "usearch-native" else 3,
                                zorder=5 if engine == "orion" else 3, markeredgecolor="white", markeredgewidth=.4,
                                linewidth=2.2 if engine == "orion" else 1.1)
                ax.set(title=name, xlim=(.899, 1.002), xlabel="Recall@10",
                       ylabel=f'{quantile[:3].upper()} latency (ms) ↓', yscale="log")
                ax.set_xticks([.9, .95, 1.0])
            axes.flat[-1].axis("off")
            axes.flat[-1].legend(handles=[Line2D([], [], color=colors[e], label=ENGINES[e],
                                    marker="D" if e == "usearch-native" else None,
                                    linestyle="none" if e == "usearch-native" else "-") for e in engines],
                                    loc="center", frameon=False, fontsize=9)
            fig.suptitle(f'Request latency · {concurrency} concurrent client{"s" if concurrency > 1 else ""} · {quantile[:3].upper()}', fontsize=16)
            fig.text(.05, .025, "M4 Max · k=10 · 3 rounds · qualified samples only · lines join measured points; no interpolation in comparisons.\nUSearch native: separate no-GIL control, three datasets / one effort each, serial kernel. Lower is better.", fontsize=9, color=PALETTE["annot"])
            fig.tight_layout(rect=(0, .08, 1, .96))
            save_png_and_pdf(fig, str(output / f'latency-{quantile[:3]}-c{concurrency}.png'), pdf_font_scale=1.0)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Frozen per-job result JSON directory")
    parser.add_argument("--usearch-control", type=Path, required=True, help="Verified control analysis.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engines", default="orion,parlayann,diskann,hnswlib,usearch-native", help="Comma-separated engines to plot")
    args = parser.parse_args()
    engines = args.engines.split(",")
    if set(engines) - ENGINES.keys():
        parser.error("Unknown engine")
    data = curate(args.inputs, args.usearch_control)
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(data, args.output)
    plot(data, args.output, engines)
    (args.output / "request-latency.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "points": data["point_counts"]}))


if __name__ == "__main__":
    main()
