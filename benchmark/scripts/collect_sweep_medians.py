#!/usr/bin/env python3
"""Parse ParlayANN `neighbors` stdout + Orion sweep stdout +
DiskANN sweep stdout from a back-to-back three-engine run, compute
per-recall medians across NUM_RUNS, and emit a single JSON consumable
by the comparison plot script.

Input files (inside `--tmpdir`):
  orion_{1..N}.out   — full stdout from `cargo run … orion-sweep`
                          (lines like `  L=  16  R@10=0.9095  QPS=100291`)
  diskann_{1..N}.out  — full stdout from `cargo run … diskann_sweep`
                          (same headline format as orion)
  pa_{1..N}.out       — full stdout from `./neighbors …`
                          (lines like `For 10@10 recall = 0.529, QPS = 1.969e+05, …`)

Output JSON:
  {
    "dataset": "sift",
    "num_points": 1000000,
    "num_runs": N,
    "orion_runs":    [[[recall, qps], …], …],
    "orion":         [[recall, qps], …],
    "diskann_runs":   [[[recall, qps], …], …],
    "diskann":        [[recall, qps], …],
    "parlayann_runs": […],
    "parlayann":      […],
  }

Median is taken per L index for orion/diskann (fixed L schedule
across runs) and per row index for ParlayANN (fixed Q schedule).
DiskANN files are optional — if absent, the diskann series is omitted
so the old two-engine consumers still work.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


ORION_RE = re.compile(
    r"\bL=\s*(\d+)\s+R@10=([0-9.]+)\s+QPS=(\d+)"
)
PA_RE = re.compile(
    r"^For\s+10@10\s+recall\s*=\s*([0-9.]+),\s*QPS\s*=\s*([0-9.eE+-]+)"
)


def parse_orion(path: Path) -> list[tuple[float, float]]:
    out = []
    for line in path.read_text().splitlines():
        m = ORION_RE.search(line)
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
    if not runs[0] or len({len(r) for r in runs}) != 1:
        raise ValueError("Empty or incomplete sweep: runs must have identical row counts")
    n = len(runs[0])
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

    if args.num_runs <= 0 or args.num_points <= 0:
        ap.error("num-runs and num-points must be positive")
    tmpdir = Path(args.tmpdir)

    orion_runs = [parse_orion(tmpdir / f"orion_{i}.out") for i in range(1, args.num_runs + 1)]
    pa_runs = [parse_pa(tmpdir / f"pa_{i}.out") for i in range(1, args.num_runs + 1)]

    # DiskANN files are optional — older two-engine sweeps don't write them.
    diskann_paths = [tmpdir / f"diskann_{i}.out" for i in range(1, args.num_runs + 1)]
    have_diskann = all(p.exists() for p in diskann_paths)
    if not have_diskann and any(p.exists() for p in diskann_paths):
        ap.error("Incomplete DiskANN runs")
    diskann_runs = (
        [parse_orion(p) for p in diskann_paths] if have_diskann else []
    )

    orion_med = median_by_slot(orion_runs)
    pa_med = median_by_slot(pa_runs)
    diskann_med = median_by_slot(diskann_runs) if have_diskann else []

    out = {
        "dataset": args.dataset,
        "num_points": args.num_points,
        "num_runs": args.num_runs,
        "k": 10,
        "threads": 8,
        "orion_runs": orion_runs,
        "orion": orion_med,
        "parlayann_runs": pa_runs,
        "parlayann": pa_med,
    }
    if have_diskann:
        out["diskann_runs"] = diskann_runs
        out["diskann"] = diskann_med

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved {args.out}")
    print(f"  orion: {len(orion_med)} points, QPS range [{min(p[1] for p in orion_med):.0f}, {max(p[1] for p in orion_med):.0f}]")
    if have_diskann:
        print(f"  diskann: {len(diskann_med)} points, QPS range [{min(p[1] for p in diskann_med):.0f}, {max(p[1] for p in diskann_med):.0f}]")
    print(f"  parlay: {len(pa_med)} points, QPS range [{min(p[1] for p in pa_med):.0f}, {max(p[1] for p in pa_med):.0f}]")


if __name__ == "__main__":
    main()
