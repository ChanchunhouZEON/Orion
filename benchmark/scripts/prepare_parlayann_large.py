#!/usr/bin/env python3
"""Convert (if needed), preflight, and build a reproducible large ParlayANN run.

Requires a prebuilt neighbors executable. Every invocation uses a new directory;
full logs and failure state remain there if a stage fails. This builds the graph,
extras checkpoint and STAG export, without running a query benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from parlayann_convert import atomic_output, convert_vectors, inspect_vectors


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"root": str(root), "revision": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain", "--untracked-files=normal")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-format", choices=["bvecs", "fvecs", "u8bin", "fbin"])
    parser.add_argument("--neighbors", type=Path, required=True)
    parser.add_argument("--parlay-root", type=Path, default=Path(__file__).resolve().parents[3] / "ParlayANN")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--build-l", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.15)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--max-extra", type=int, default=16)
    parser.add_argument("--local-pct", type=int, default=60)
    parser.add_argument("--build-batch-size", type=int, default=1000000)
    parser.add_argument("--export-batch-size", type=int, default=65536)
    parser.add_argument("--memory-budget-gib", type=float, required=True)
    parser.add_argument("--no-resource-time", action="store_true",
                        help="Skip /usr/bin/time for small portable checks; peak RSS is then unavailable")
    args = parser.parse_args()
    if args.threads <= 0 or args.max_points < 0:
        parser.error("threads must be positive; max-points nonnegative")
    base_format = args.base_format or args.base.suffix.lstrip(".")
    if base_format not in ("bvecs", "fvecs", "u8bin", "fbin"):
        parser.error("unknown input format")
    count, dimension = inspect_vectors(args.base, base_format)
    count = min(count, args.max_points) if args.max_points else count
    binary = args.neighbors.resolve(strict=True)
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running", "source": file_identity(args.base),
                "num_points": count, "dimension": dimension,
                "executable": {**file_identity(binary), "sha256": sha256(binary)},
                "parlay": git_state(args.parlay_root.resolve()),
                "driver": git_state(Path(__file__).resolve().parents[2]),
                "threads": args.threads, "stages": [],
                "peak_rss_available": not args.no_resource_time}

    def publish_manifest():
        with atomic_output(root / "run.json") as output:
            output.write((json.dumps(manifest, indent=2) + "\n").encode())

    def run_stage(name, command):
        measured = list(map(str, command))
        if not args.no_resource_time:
            option = "-l" if platform.system() == "Darwin" else "-v"
            measured = ["/usr/bin/time", option, "-o", str(root / (name + ".time.txt")), *measured]
        stage = {"name": name, "command": measured, "status": "running"}
        manifest["stages"].append(stage)
        publish_manifest()
        started = time.monotonic()
        with (root / (name + ".log")).open("w") as log:
            result = subprocess.run(measured, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PARLAY_NUM_THREADS": str(args.threads)})
        stage.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started,
                     status="complete" if result.returncode == 0 else "failed")
        publish_manifest()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {root / (name + '.log')}")

    publish_manifest()
    try:
        suffix = "u8bin" if base_format in ("bvecs", "u8bin") else "fbin"
        base = args.base.resolve()
        if base_format != suffix or args.max_points:
            # Use a subprocess so conversion has its own resource measurement.
            base = root / ("base." + suffix)
            run_stage("convert", [os.sys.executable, Path(__file__).resolve(), "--convert-base",
                                  args.base.resolve(), base, base_format, args.max_points])
        common = [binary, "-base_path", base, "-data_type", "uint8" if suffix == "u8bin" else "float",
                  "-dist_func", "Euclidian", "-R", args.degree, "-L", args.build_l,
                  "-alpha", args.alpha, "-num_passes", args.passes,
                  "-max_extra", args.max_extra, "-local_pct", args.local_pct,
                  "-build_batch_size", args.build_batch_size, "-export_batch_size", args.export_batch_size,
                  "-memory_budget_gib", args.memory_budget_gib, "-build_only", "-light_stats"]
        run_stage("preflight", [*common, "-preflight"])
        run_stage("build", [*common, "-graph_outfile", root / "graph.bin",
                            "-extras_outfile", root / "extras.bin", "-staged_outfile", root / "graph.staged"])
        manifest["artifacts"] = [file_identity(root / name) for name in ("graph.bin", "extras.bin", "graph.staged")]
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        publish_manifest()
    print(root / "run.json")


if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--convert-base":
        _, _, source, destination, vector_format, limit = os.sys.argv
        convert_vectors(source, destination, vector_format, int(limit))
    else:
        main()
