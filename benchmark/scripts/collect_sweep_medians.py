#!/usr/bin/env python3
"""Parse ParlayANN `neighbors` stdout and StagedDiskANN sweep stdout from
a back-to-back run, compute per-recall medians across NUM_RUNS, and emit
a single JSON consumable by the comparison plot script.

Input files (inside `--tmpdir`):
  staged_{1..N}.out  — full stdout from `cargo run … staged-parlayann-sweep`
                        (lines like `  L=  16  R@10=0.9095  QPS=100291`)
  pa_{1..N}.out      — full stdout from `./neighbors …`
                        (lines like `For 10@10 recall = 0.529, QPS = 1.969e+05, …`)

Output JSON:
  {
    "dataset": "sift",
    "num_points": 1000000,
    "num_runs": N,
    "staged_runs": [[[recall, qps], …], …],   // raw per-run
    "staged":       [[recall, qps], …],       // per-L median
    "parlayann_runs": […],                    // raw per-run (recall, qps)
    "parlayann":      […],                    // per-slot median
  }

Median is taken per L index for staged (L values are fixed across runs)
and per row index for ParlayANN (it sweeps a fixed Q schedule).
"""

import argparse
import json
import re
import statistics
from pathlib import Path


STAGED_RE = re.compile(
    r"^\s*L=\s*(\d+)\s+R@10=([0-9.]+)\s+QPS=(\d+)"
)
PA_RE = re.compile(
    r"^For\s+10@10\s+recall\s*=\s*([0-9.]+),\s*QPS\s*=\s*([0-9.eE+-]+)"
)


def parse_staged(path: Path) -> list[tuple[float, float]]:
    out = []
    for line in path.read_text().splitlines():
        m = STAGED_RE.match(line)
        if m:
            _L, r, q = m.group(1), float(m.group(2)), float(m.group(3))
            out.append((r, q))
    return out


def parse_pa(path: Path) -> list[tuple[float, float]]:
    out = []
    for line in path.read_text().splitlines():
        m = PA_RE.match(line)
        if m:
            r, q = float(m.group(1)), float(m.group(2))
            out.append((r, q))
    return out


def median_by_slot(runs: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    if not runs:
        return []
    # Align by row index — each run is expected to produce the same sweep
    # schedule, so slot i has the same recall target.
    n = min(len(r) for r in runs)
    out = []
    for i in range(n):
        recalls = [r[i][0] for r in runs]
        qpss = [r[i][1] for r in runs]
        recall = statistics.median(recalls)
        qps = statistics.median(qpss)
        out.append((recall, qps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmpdir", required=True, help="Directory with *.out files")
    ap.add_argument("--num-runs", type=int, required=True)
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--num-points", type=int, default=1_000_000)
    args = ap.parse_args()

    tmpdir = Path(args.tmpdir)

    staged_runs = [parse_staged(tmpdir / f"staged_{i}.out") for i in range(1, args.num_runs + 1)]
    pa_runs = [parse_pa(tmpdir / f"pa_{i}.out") for i in range(1, args.num_runs + 1)]

    staged_med = median_by_slot(staged_runs)
    pa_med = median_by_slot(pa_runs)

    out = {
        "dataset": args.dataset,
        "num_points": args.num_points,
        "num_runs": args.num_runs,
        "staged_runs": staged_runs,
        "staged": staged_med,
        "parlayann_runs": pa_runs,
        "parlayann": pa_med,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved {args.out}")
    print(f"  staged: {len(staged_med)} points, QPS range [{min(p[1] for p in staged_med):.0f}, {max(p[1] for p in staged_med):.0f}]")
    print(f"  parlay: {len(pa_med)} points, QPS range [{min(p[1] for p in pa_med):.0f}, {max(p[1] for p in pa_med):.0f}]")


if __name__ == "__main__":
    main()
