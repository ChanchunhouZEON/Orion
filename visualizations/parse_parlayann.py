#!/usr/bin/env python3
"""Parse ParlayANN raw run CSVs into median-aggregated JSON for plotting.

For each dataset subfolder under `visualizations/parlayann/`, reads all
`run*.csv` files, takes the per-L median recall/QPS across runs, and writes
`visualizations/parlayann_{slug}.json`. The plot scripts
(`plot_qps_recall_full.py` etc.) then pick up these JSONs as overlays.

ParlayANN CSV layout (emitted by `algorithms/vamana/neighbors`):
    "GRAPH","Parameters","Size","Build time",...
    "Vamana","R = 64, L = 100",1000000,57.6,...
    (blank)
    "Num queries","Target recall","Actual recall","QPS",...
    10000,0.5,0.52901,71054,...
    ...
"""

import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PARLAYANN_DIR = os.path.join(ROOT, "parlayann")

# Directory slug → output JSON slug. Kept explicit so plot scripts know what
# to look for; add new datasets here after dropping their runs/ folder.
DATASETS = [
    ("sift1m", "sift1m"),
    ("glove25full", "glove25full"),
    ("glove100full", "glove100full"),
    ("gistfull", "gistfull"),
]


def parse_csv(path):
    """Return (build_time_s, [(recall, qps), ...]) from a ParlayANN CSV."""
    with open(path) as f:
        rows = list(csv.reader(f))
    build_time = float(rows[1][3]) if len(rows) >= 2 and len(rows[1]) >= 4 else 0.0
    start = None
    for i, r in enumerate(rows):
        if r and r[0].strip() == "Num queries":
            start = i + 1
            break
    pts = []
    if start is None:
        return build_time, pts
    for r in rows[start:]:
        if len(r) < 4 or not r[0].strip():
            continue
        try:
            pts.append((float(r[2]), float(r[3])))
        except ValueError:
            continue
    return build_time, pts


def aggregate(slug_in, slug_out):
    folder = os.path.join(PARLAYANN_DIR, slug_in)
    csv_paths = sorted(glob.glob(os.path.join(folder, "run*.csv")))
    if not csv_paths:
        print(f"Skipped {slug_in}: no runs under {folder}")
        return None

    build_times, all_pts = [], []
    for p in csv_paths:
        bt, pts = parse_csv(p)
        if not pts:
            print(f"  {p}: no data rows — skipping")
            continue
        build_times.append(bt)
        all_pts.append(pts)

    if not all_pts:
        return None

    n = min(len(pts) for pts in all_pts)
    arr = np.array([[[pts[i][0], pts[i][1]] for i in range(n)] for pts in all_pts])
    med_recall = np.median(arr[:, :, 0], axis=0)
    med_qps = np.median(arr[:, :, 1], axis=0)
    out = {
        "dataset": slug_out,
        "algorithm": "ParlayANN Vamana",
        "runs": len(all_pts),
        "build_time_s_first_run": build_times[0] if build_times else 0.0,
        "parlayann_vamana": [
            [round(float(r), 4), int(q)] for r, q in zip(med_recall, med_qps)
        ],
    }
    out_path = os.path.join(ROOT, f"parlayann_{slug_out}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{slug_in}: {len(all_pts)} runs × {n} L points → {out_path}")
    return out


def main():
    for slug_in, slug_out in DATASETS:
        aggregate(slug_in, slug_out)


if __name__ == "__main__":
    main()
