#!/usr/bin/env python3
"""Warm-batch f32 Python smoke benchmark using an Index.save() snapshot.

This measures the Python API including query copies and result conversion. It
is not an estimate of FFI overhead against historical, differently timed runs.
An explicit new output path keeps published experiment data untouched.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from benchmark_support import read_fvecs, read_ivecs, recall_at_k
from orion_py import Index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Index.save() directory")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True, help="fvecs warmup/search queries")
    parser.add_argument("--groundtruth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--search-list-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = parser.parse_args()
    if args.trials < 1 or args.output.exists():
        parser.error("trials must be positive and output must be a new path")
    index = Index.open(args.cache, base=args.base, threads=args.threads)
    if index.info["storage"] != "f32":
        parser.error("this fvecs smoke benchmark expects an f32 snapshot")
    queries, count, _ = read_fvecs(str(args.queries))
    truth = read_ivecs(str(args.groundtruth)).tolist() if args.groundtruth else None
    calibration = index.calibrate(queries[:200], k=args.k)
    rows = []
    for beam in args.search_list_sizes:
        index.search(queries, k=args.k, search_list_size=beam)
        samples = []
        for _ in range(args.trials):
            start = time.perf_counter()
            result = index.search(queries, k=args.k, search_list_size=beam)
            samples.append(count / (time.perf_counter() - start))
        recall = recall_at_k(result.ids.tolist(), truth, args.k) if truth else None
        row = dict(search_list_size=beam, recall=recall, qps=statistics.median(samples), qps_samples=samples)
        rows.append(row)
        print(json.dumps(row))
    report = dict(protocol="warm Python batches; copies and result conversion included",
                  index=index.info, calibration=vars(calibration), k=args.k, trials=args.trials, rows=rows)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
