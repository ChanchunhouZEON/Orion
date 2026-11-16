#!/usr/bin/env python3
# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Measure all baseline request latencies using explicit, reproducible recipes.

Example: python -m benchmark.latency_baselines --config comparison.json \
    --output ../experiment/2026-11-12/baseline-latency
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

from .adapters import ENGINES
from .data import write_json
from .protocol import Protocol
from .report import report

PATH_KEYS = {
    "base",
    "query",
    "groundtruth",
    "library",
    "base_fbin",
    "graph",
    "index",
    "snapshot",
    "cache",
    "experiment_config",
}


def resolve_paths(value, root):
    if isinstance(value, list):
        return [resolve_paths(v, root) for v in value]
    if isinstance(value, dict):
        return {
            k: (
                str((root / v).resolve())
                if k in PATH_KEYS and isinstance(v, str)
                else resolve_paths(v, root)
            )
            for k, v in value.items()
        }
    return value


def validate(config):
    protocol = Protocol(**config.get("protocol", {})).validate()
    if (
        type(config.get("threads", 8)) is not int
        or not 1 <= config.get("threads", 8) <= 64
    ):
        raise ValueError("threads must be in 1..64")
    if type(config.get("k", 10)) is not int or config.get("k", 10) <= 0:
        raise ValueError("k must be positive")
    names = set()
    if not config.get("datasets"):
        raise ValueError("datasets must be nonempty")
    for ds in config["datasets"]:
        name = ds["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("dataset names must be unique safe path components")
        names.add(name)
        if ds["metric"] not in ("squared-l2", "cosine", "inner-product"):
            raise ValueError("metric must be squared-l2, cosine or inner-product")
        for engine, cfg in ds["engines"].items():
            if engine not in ENGINES:
                raise ValueError(f"unknown engine {engine}")
            efforts = cfg["efforts"]
            if (
                not efforts
                or any(type(e) is not int or e <= 0 for e in efforts)
                or len(set(efforts)) != len(efforts)
            ):
                raise ValueError("efforts must be distinct positive integers")
    targets = config.get("recall_targets", [0.90, 0.95, 0.98, 0.99])
    if not targets or any(not 0 <= t <= 1 for t in targets):
        raise ValueError("recall targets must be in [0,1]")
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new result directory")
    p.add_argument(
        "--engines",
        default="all",
        help="all existing engines or comma-separated subset",
    )
    p.add_argument(
        "--datasets", help="comma-separated subset of configured dataset names"
    )
    p.add_argument(
        "--prepare-missing",
        action="store_true",
        help="explicitly build missing Python indices once; never rebuild native graphs",
    )
    p.add_argument(
        "--ephemeral-caches",
        action="store_true",
        help="build missing Python indices in owned temporary directories and remove them after each job",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="allow missing/unsupported rows, still report them",
    )
    p.add_argument(
        "--job-timeout",
        type=float,
        default=14400,
        help="wall-time bound for one engine, including load/build",
    )
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    config = resolve_paths(
        json.loads(args.config.read_text()), args.config.resolve().parent
    )
    protocol = validate(config)
    if args.report_only:
        report(args.output, config.get("recall_targets", [0.9, 0.95, 0.98, 0.99]))
        return 0
    engines = list(ENGINES) if args.engines == "all" else args.engines.split(",")
    if len(set(engines)) != len(engines) or set(engines) - set(ENGINES):
        p.error("unknown or duplicate engine")
    datasets = config["datasets"]
    if args.datasets:
        wanted = set(args.datasets.split(","))
        if wanted - {d["name"] for d in datasets}:
            p.error("unknown dataset")
        datasets = [d for d in datasets if d["name"] in wanted]
    if not args.job_timeout > 0:
        p.error("job-timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    write_json(args.output / "config.json", config)
    jobs = [(ds, engine) for ds in datasets for engine in engines]
    random.Random(protocol.seed).shuffle(jobs)  # reproducible engine/dataset order
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PARLAY_NUM_THREADS=str(config.get("threads", 8)),
        RAYON_NUM_THREADS=str(config.get("threads", 8)),
        LANCE_CPU_THREADS=str(config.get("threads", 8)),
    )
    for ds, engine in jobs:
        dest = args.output / (ds["name"] + "--" + engine)
        dest.mkdir()
        engine_config = ds["engines"].get(engine)
        temporary_cache = None
        if (
            args.ephemeral_caches
            and args.prepare_missing
            and engine_config
            and engine_config.get("cache")
            and not Path(engine_config["cache"]).exists()
        ):
            temporary_cache = tempfile.TemporaryDirectory(
                prefix="temporary-index-", dir=dest
            )
            engine_config = dict(
                engine_config, cache=str(Path(temporary_cache.name) / "index")
            )
        job = dict(
            dataset=ds,
            engine=engine,
            engine_config=engine_config,
            protocol=protocol.json(),
            k=config.get("k", 10),
            threads=config.get("threads", 8),
            prepare_missing=args.prepare_missing,
        )
        write_json(dest / "job.json", job)
        if job["engine_config"] is None:
            write_json(
                dest / "result.json",
                dict(
                    dataset=ds["name"],
                    engine=engine,
                    protocol=protocol.json(),
                    status="missing",
                    error="no configured cache/recipe",
                    points=[],
                ),
            )
            continue
        command = [
            sys.executable,
            "-m",
            "latency_baselines.worker",
            str(dest / "job.json"),
        ]
        print(f"Starting {ds['name']}/{engine}; log: {dest / 'worker.log'}", flush=True)
        with (dest / "worker.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.job_timeout,
                )
                error = (
                    f"worker exited {result.returncode}" if result.returncode else None
                )
            except subprocess.TimeoutExpired:
                error = f"worker exceeded {args.job_timeout}s; process terminated"
        result_path = dest / "result.json"
        record = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else dict(
                dataset=ds["name"],
                engine=engine,
                protocol=protocol.json(),
                points=[],
                status="running",
            )
        )
        if error and record["status"] in ("running", "complete"):
            record.update(status="failed", error=error)
            write_json(result_path, record)
        if temporary_cache is not None:
            temporary_cache.cleanup()
            record["temporary_index_removed"] = True
            write_json(result_path, record)
        report(args.output, config.get("recall_targets", [0.9, 0.95, 0.98, 0.99]))
    records = report(args.output, config.get("recall_targets", [0.9, 0.95, 0.98, 0.99]))
    allowed = (
        {"complete", "missing", "unsupported"} if args.allow_missing else {"complete"}
    )
    return 0 if all(r["status"] in allowed for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
