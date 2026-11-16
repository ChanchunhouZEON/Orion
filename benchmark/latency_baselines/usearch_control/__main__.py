# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Deferred USearch control; never start alongside the preceding experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ..data import write_json

TERMINAL = {"complete", "incomplete", "missing", "unsupported"}


def process_identity(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("cannot inspect preceding process: " + result.stderr)
    return result.stdout.strip()


def gate(config, identity):
    if identity == config["wait_identity"]:
        return False, "preceding experiment still running"
    root = Path(config["wait_results"])
    for name in config["wait_jobs"]:
        path = root / name / "result.json"
        if not path.exists():
            raise RuntimeError(
                f"preceding experiment exited with missing result: {name}"
            )
        status = json.loads(path.read_text()).get("status")
        if status not in TERMINAL:
            raise RuntimeError(
                f"preceding experiment not safely finished: {name}: {status}"
            )
    return True, "preceding process ended; all expected jobs terminal"


def summarize(output):
    lines = [
        "| Dataset | Variant | Clients | Recall | QPS | P50 ms | P95 ms | P99 ms | Complete |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for path in sorted(output.glob("*/result.json")):
        record = json.loads(path.read_text())
        for point in record.get("points", []):
            if "pooled" not in point:
                continue
            d = point["pooled"]
            lines.append(
                "| {} | {} | {} | {} | {:.1f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                    record["dataset"],
                    point["variant"],
                    point["concurrency"],
                    point["evaluation"]["recall"],
                    point["median_instrumented_qps"],
                    d["p50_us"] / 1000,
                    d["p95_us"] / 1000,
                    d["p99_us"] / 1000,
                    point["complete"],
                )
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.plan.read_text())
    output = args.plan.resolve().parent
    state = {"status": "waiting", "pid": os.getpid(), "wait_pid": config["wait_pid"]}
    try:
        while True:
            ready, reason = gate(config, process_identity(config["wait_pid"]))
            state["reason"] = reason
            write_json(output / "state.json", state)
            if ready:
                break
            time.sleep(30)
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        for job in config["jobs"]:
            dest = output / job["dataset"]["name"]
            dest.mkdir(exist_ok=False)
            write_json(dest / "job.json", job)
            state.update(status="running", dataset=job["dataset"]["name"])
            write_json(output / "state.json", state)
            with (dest / "worker.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "latency_baselines.usearch_control.worker",
                        str(dest / "job.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=28800,
                )
            summarize(output)
            if result.returncode:
                raise RuntimeError(
                    f"{job['dataset']['name']} failed/incomplete; inspect worker.log"
                )
        state.update(status="complete")
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "state.json", state)


if __name__ == "__main__":
    main()
