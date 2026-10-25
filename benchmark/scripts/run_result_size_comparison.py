#!/usr/bin/env python3
"""Compare k=10/100 with the full calibrated Orion pipeline at identical L."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["sift", "glove25", "glove100", "gist", "deep10m", "msmarco_bert_1M"]
POINT = re.compile(r"\bL=\s*(\d+)\s+R@(\d+)=([0-9.]+)\s+QPS=(\d+)")


def parse_points(text, k, schedule):
    points = [dict(L=int(l), recall=float(r), qps=int(q))
              for l, found_k, r, q in POINT.findall(text) if int(found_k) == k]
    if [p["L"] for p in points] != schedule:
        raise ValueError("Missing, duplicated, or unexpected beam widths")
    if any(not 0 <= p["recall"] <= 1 or p["qps"] <= 0 for p in points):
        raise ValueError("Invalid measured point")
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=1)
    args = ap.parse_args()
    if args.trials < 1 or args.cooldown < 0 or len(set(args.datasets)) != len(args.datasets):
        ap.error("Invalid trials, cooldown, or duplicate datasets")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "benchmark/configs/sweep.yaml").read_text())
    sweep = config["defaults"]["sweep"]
    schedule = sweep["schedules_by_k"][100]["sweep"]
    sweep["calibration"]["base_l"] = 200
    snapshot = out / "config.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(os.environ, ORION_RERANK_FACTOR="2")
    binary = ROOT / "target/release/orion-sweep"
    plans = {}
    for dataset in args.datasets:
        plans[dataset] = {}
        for k in (10, 100):
            cmd = [str(binary), dataset, "--config", str(snapshot), "--k", str(k),
                   "--threads", "8", "--trials", str(args.trials),
                   "--search-list-sizes", ",".join(map(str, schedule))]
            output = subprocess.check_output(cmd + ["--print-config"], cwd=ROOT,
                                             env=env, stderr=subprocess.STDOUT, text=True)
            resolved = json.loads(output[output.index("{"):])
            assert resolved["sweep"]["calibration_l"] == 200
            assert resolved["sweep"]["search_list_sizes"] == schedule
            gt = ROOT / resolved["groundtruth"]
            with gt.open("rb") as f:
                width = struct.unpack("<I", f.read(4))[0]
            assert width >= 100 and gt.stat().st_size == resolved["num_queries"] * (width + 1) * 4
            (out / f"{dataset}_k{k}.config.json").write_text(json.dumps(resolved, indent=2) + "\n")
            plans[dataset][k] = (cmd, resolved)
        for key in ("base", "query", "groundtruth", "staged_file", "cascade", "graph_degree", "alpha", "build_l"):
            assert plans[dataset][10][1][key] == plans[dataset][100][1][key]
    manifest = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    early_exit=True, adaptive=True, calibration_l=200, calibration_samples=200,
                    rerank_factor=2, threads=8, trials=args.trials, cooldown_s=args.cooldown,
                    search_list_sizes=schedule, status={}, commands={})
    results = {}
    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    save()
    for index, dataset in enumerate(args.datasets):
        results[dataset] = {}
        for k in ((10, 100) if index % 2 == 0 else (100, 10)):
            name = f"{dataset}_k{k}"
            cmd, resolved = plans[dataset][k]
            manifest["commands"][name] = cmd
            manifest["status"][name] = "running"
            save()
            print(f"Running {name}", flush=True)
            time.sleep(args.cooldown)
            log = out / f"{name}.log"
            with log.open("w") as f:
                run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if run.returncode:
                manifest["status"][name] = "failed"
                save()
                raise RuntimeError(f"{name} failed; see {log}")
            text = log.read_text()
            calibration = re.search(r"Calibrated .*threshold=([0-9.]+), early_exit_limit=(\d+)", text)
            if not calibration:
                raise ValueError(f"Missing calibration in {name}")
            results[dataset][str(k)] = dict(query_count=resolved["num_queries"],
                threshold_logged=float(calibration[1]), early_exit_limit=int(calibration[2]),
                points=parse_points(text, k, schedule))
            manifest["status"][name] = "completed"
            save()
        a, b = results[dataset]["10"]["points"], results[dataset]["100"]["points"]
        results[dataset]["paired"] = [dict(L=x["L"], recall10=x["recall"], recall100=y["recall"],
            qps10=x["qps"], qps100=y["qps"], throughput_retention=y["qps"] / x["qps"])
            for x, y in zip(a, b)]
        save()
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
