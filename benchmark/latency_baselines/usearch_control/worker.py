# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""One graph, one kernel, three bindings; no mutation during timed trials."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
from ..adapters import PythonIndex
from ..data import rows, recall, write_json
from ..protocol import Protocol, evaluate, measure, distribution
from ..worker import versions
from .native import NativeIndex, library


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(job_path):
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    out = job_path.parent
    ds = job["dataset"]
    protocol = Protocol(**job["protocol"]).validate()
    k = job.get("k", 10)
    record = dict(
        status="preparing",
        dataset=ds["name"],
        job=job,
        versions=versions(),
        points=[],
        timing_scope="Python single-row adapter wall time; includes binding and result allocation",
        variants=["python", "native-gil", "native-nogil"],
        comparison="All variants share serialized graph, coordinates, effort and SimSIMD library. Native variants differ only in PyDLL vs CDLL.",
    )
    write_json(out / "result.json", record)
    native = None
    python = None
    try:
        base = rows(ds["base"])
        queries = np.array(rows(ds["query"]), dtype="f4", order="C", copy=True)
        gt = rows(ds["groundtruth"], groundtruth=True)
        if (
            len(gt) != len(queries)
            or queries.shape[1] != base.shape[1]
            or not np.isfinite(queries).all()
        ):
            raise ValueError("invalid query/GT shape or values")
        for offset in range(0, len(base), 8192):
            if not np.isfinite(base[offset : offset + 8192]).all():
                raise ValueError("nonfinite base")
        # These are owned temporary indices, deleted only after every trial and
        # raw sample is saved. No old experiment cache or config is overwritten.
        with tempfile.TemporaryDirectory(prefix="shared-index-", dir=out) as temp:
            cfg = dict(ds["engines"]["usearch"], cache=str(Path(temp) / "index"))
            start = time.perf_counter()
            python = PythonIndex("usearch", cfg, ds, base, 8, True)
            index = python.index
            native = NativeIndex(
                library(job["library"]),
                cfg["cache"],
                dimension=base.shape[1],
                count=len(base),
                metric=index.metric_kind,
                slots=8,
            )
            if native.isa != index.specs["hardware_acceleration"]:
                raise ValueError(
                    "distance backend differs; refusing performance comparison"
                )
            record.update(
                graph_sha256=sha_file(cfg["cache"]),
                graph_bytes=Path(cfg["cache"]).stat().st_size,
                native_library_sha256=sha_file(job["library"]),
                prepare_seconds=time.perf_counter() - start,
                native_isa=native.isa,
                python_specs={key: str(value) for key, value in index.specs.items()},
                switch_interval=sys.getswitchinterval(),
                scratch_slots=8,
                status="validating",
            )
            order = (
                np.random.default_rng(protocol.seed).permutation(len(queries)).tolist()
            )
            for effort in job["efforts"]:
                python.configure(effort, k)
                native.configure(effort)
                # IDs must agree exactly; distances permit only f32 roundoff.
                control = evaluate(python.search, queries, 1)
                for q, expected in zip(queries, control):
                    ids, dist = native.search(q, k)
                    py_result = index.search(q, count=k, threads=1)
                    if tuple(map(int, ids)) != expected:
                        raise ValueError("native/stock IDs differ")
                    np.testing.assert_allclose(
                        dist, py_result.distances, rtol=2e-6, atol=2e-6
                    )
                functions = {
                    "python": python.search,
                    "native-gil": lambda q: native.search(q, k, release_gil=False)[0],
                    "native-nogil": lambda q: native.search(q, k)[0],
                }
                for name, search in functions.items():
                    for clients in protocol.concurrency:
                        if evaluate(search, queries, clients) != control:
                            raise ValueError(f"{name}/{clients}: concurrent IDs differ")
                evaluation = recall(control, gt, k, len(base))
                record["parity"] = (
                    "all queries: IDs, f32 distances, serial/concurrent results verified"
                )
                record["status"] = "running"
                points = {}
                samples = {}
                for name in functions:
                    for clients in protocol.concurrency:
                        key = (name, clients)
                        points[key] = dict(
                            variant=name,
                            effort=effort,
                            concurrency=clients,
                            evaluation=evaluation,
                            rounds=[],
                            complete=False,
                        )
                        samples[key] = []
                        record["points"].append(points[key])
                # Rotate variants and reverse client order to avoid always
                # measuring the no-GIL implementation last.
                names = list(functions)
                for round_id in range(protocol.rounds):
                    names_round = names[round_id % 3 :] + names[: round_id % 3]
                    clients_round = (
                        protocol.concurrency
                        if round_id % 2 == 0
                        else protocol.concurrency[::-1]
                    )
                    for clients in clients_round:
                        for name in names_round:
                            time.sleep(protocol.cooldown_seconds)
                            trial, raw = measure(
                                functions[name], queries, order, clients, protocol
                            )
                            path = (
                                out / f"{name}-l{effort}-c{clients}-r{round_id+1}.npy"
                            )
                            np.save(path, raw)
                            trial.update(
                                round=round_id + 1,
                                samples=path.name,
                                samples_sha256=sha_file(path),
                            )
                            point = points[name, clients]
                            point["rounds"].append(trial)
                            samples[name, clients].append(raw)
                            point["pooled"] = distribution(
                                np.concatenate(samples[name, clients])
                            )
                            point["median_instrumented_qps"] = float(
                                np.median(
                                    [t["instrumented_qps"] for t in point["rounds"]]
                                )
                            )
                            point["complete"] = len(
                                point["rounds"]
                            ) == protocol.rounds and all(
                                t["minimum_reached"] for t in point["rounds"]
                            )
                            write_json(out / "result.json", record)
                            print(
                                ds["name"],
                                name,
                                clients,
                                round_id + 1,
                                point["pooled"],
                                flush=True,
                            )
            functions.clear()
            search = None
            native.close()
            native = None
            python = None
            index = None
            gc.collect()
        record["temporary_index_removed"] = True
        record["status"] = (
            "complete" if all(p["complete"] for p in record["points"]) else "incomplete"
        )
    except Exception as e:
        record.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        if native is not None:
            native.close()
        write_json(out / "result.json", record)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))
