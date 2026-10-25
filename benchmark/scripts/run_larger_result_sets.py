#!/usr/bin/env python3
"""Run full-query k=100 sweeps, retaining raw results and resolved configs."""
import argparse
import datetime
import json
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["sift", "glove25", "glove100", "gist", "deep10m", "msmarco_bert_1M"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/adaptive")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cooldown", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cooldown < 0 or len(args.datasets) != len(set(args.datasets)):
        parser.error("cooldown must be nonnegative; datasets must be unique")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_dir or ROOT / "visualizations/larger_result_sets" / stamp).resolve()
    binary = args.binary.resolve()
    commands = {}
    configs = {}
    # Resolve and validate every input before beginning any timing.
    for dataset in args.datasets:
        if not dataset.replace("_", "").replace("-", "").isalnum():
            parser.error("invalid dataset shortcut")
        cmd = [str(binary), dataset, "--k", "100", "--threads", "8",
               "--trials", "5", "--diagnostic-queries", "1000",
               "--recall-targets", "0.9,0.95,0.97,0.99"]
        config = json.loads(subprocess.check_output(cmd + ["--print-config"], cwd=ROOT))
        for key in ("base", "query", "groundtruth"):
            path = ROOT / config[key]
            if not path.is_file():
                raise FileNotFoundError(path)
        gt = ROOT / config["groundtruth"]
        with gt.open("rb") as handle:
            width = struct.unpack("<I", handle.read(4))[0]
        if width < 100 or gt.stat().st_size != config["num_queries"] * (width + 1) * 4:
            raise ValueError(f"{dataset}: ground-truth width/count incompatible with Recall@100")
        commands[dataset] = cmd + ["--output", str(out / f"{dataset}_k100.json")]
        configs[dataset] = config
    if args.dry_run:
        print(json.dumps({"output_dir": str(out), "commands": commands}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol": "All queries, k=100, 8 threads, 5 trials, YAML L schedule; EE disabled. Use local-plus-extra for workload results; other arms are controls. Diagnostics are separately sampled at up to 1000 queries.",
        "commands": commands, "status": {},
    }
    failures = []
    for dataset, cmd in commands.items():
        (out / f"{dataset}.config.json").write_text(json.dumps(configs[dataset], indent=2) + "\n")
        manifest["status"][dataset] = "running"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Running {dataset}: {out}", flush=True)
        time.sleep(args.cooldown)
        with (out / f"{dataset}.log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        manifest["status"][dataset] = "completed" if result.returncode == 0 else "failed"
        if result.returncode:
            failures.append(dataset)
        else:
            data = json.loads((out / f"{dataset}_k100.json").read_text())
            if data["query_count"] != configs[dataset]["num_queries"]:
                raise ValueError(f"{dataset}: timed query count mismatch")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{dataset}: {manifest['status'][dataset]}", flush=True)
    print(f"Results: {out}", flush=True)
    if failures:
        raise SystemExit(f"Failed datasets: {', '.join(failures)}")


if __name__ == "__main__":
    main()
