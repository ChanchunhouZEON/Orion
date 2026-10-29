#!/usr/bin/env python3
"""Print README speedups from the existing k=10 panel data; never rewrite results."""
import argparse
import json
from pathlib import Path

DATASETS = [
    ("sift", "SIFT"),
    ("glove25", "GloVe-25"),
    ("glove100", "GloVe-100"),
    ("gist", "GIST"),
    ("deep10m", "DEEP"),
    ("msmarco_bert_1M", "MS-MARCO"),
]


def frontier_qps(points, target):
    # At each recall retain the fastest point, then remove points dominated by
    # a faster run with higher recall. Interpolate only inside that frontier.
    by_recall = {}
    for recall, qps in points:
        by_recall[recall] = max(qps, by_recall.get(recall, 0.0))
    frontier = []
    best = 0.0
    for recall, qps in sorted(by_recall.items(), reverse=True):
        if qps > best:
            frontier.append((recall, qps))
            best = qps
    frontier.reverse()
    for recall, qps in frontier:
        if recall == target:
            return qps
    for (r0, q0), (r1, q1) in zip(frontier, frontier[1:]):
        if r0 < target < r1:
            return q0 + (q1 - q0) * (target - r0) / (r1 - r0)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recalls", nargs="+", type=float, default=[0.95, 0.99])
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / "visualizations"
    columns = [f"vs {name} @ {r:.2f}" for r in args.recalls
               for name in ("Vamana", "ParlayANN")]
    print("| Dataset | " + " | ".join(columns) + " |")
    print("|---|" + "---:|" * len(columns))
    for dataset, name in DATASETS:
        data = json.loads((source / f"sweep_orion_vs_parlayann_{dataset}.json").read_text())
        values = []
        for recall in args.recalls:
            orion = frontier_qps(data["orion"], recall)
            for engine in ("diskann", "parlayann"):
                baseline = frontier_qps(data[engine], recall)
                values.append(f"{orion / baseline:.2f}×" if orion and baseline else "—")
        print(f"| {name} | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
