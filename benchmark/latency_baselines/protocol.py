# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Closed-loop request timing, separate from input/loading/recall evaluation.

All adapters expose a Python single-request boundary. This intentionally includes
Python binding/GIL cost; it must not be merged with the Rust Index::search clock.
"""
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import itertools
import math
import threading
import time

import numpy as np


@dataclass(frozen=True)
class Protocol:
    concurrency: tuple = (1, 8)
    rounds: int = 3
    min_requests: int = 10000
    min_seconds: float = 1.0
    max_seconds: float = 30.0
    warmup_requests: int = 1000
    warmup_seconds: float = 0.2
    cooldown_seconds: float = 1.0
    seed: int = 20261112

    def validate(self):
        if (
            not self.concurrency
            or len(set(self.concurrency)) != len(self.concurrency)
            or any(type(c) is not int or not 1 <= c <= 64 for c in self.concurrency)
        ):
            raise ValueError("concurrency must be distinct integers in 1..64")
        for name in ("rounds", "min_requests", "warmup_requests"):
            value = getattr(self, name)
            if type(value) is not int or value < (
                0 if name == "warmup_requests" else 1
            ):
                raise ValueError(f"invalid {name}")
        for name in (
            "min_seconds",
            "max_seconds",
            "warmup_seconds",
            "cooldown_seconds",
        ):
            v = getattr(self, name)
            if not math.isfinite(v) or not 0 <= v <= 3600:
                raise ValueError(f"invalid {name}")
        if self.max_seconds <= 0 or self.max_seconds < self.min_seconds:
            raise ValueError("max_seconds must be positive and >= min_seconds")
        return self

    def json(self):
        return asdict(self)


def distribution(ns):
    """Exact nearest-rank quantiles over all samples, not means of percentiles."""
    x = np.sort(np.asarray(ns, dtype=np.uint64))
    if not len(x):
        raise ValueError("no latency samples")
    return {
        **{
            f"p{int(q*100)}_us": int(x[math.ceil(len(x) * q) - 1]) / 1000
            for q in (0.5, 0.95, 0.99)
        },
        "mean_us": float(x.mean()) / 1000,
        "max_us": int(x[-1]) / 1000,
        "requests": len(x),
    }


def measure(search, queries, order, concurrency, protocol, initialize=lambda: None):
    protocol.validate()
    if sorted(order) != list(range(len(queries))) or not len(queries):
        raise ValueError("order must be a complete query permutation")
    if concurrency not in protocol.concurrency:
        raise ValueError("unconfigured concurrency")
    # No coarse lock around search. CPython next(count) distributes query tickets
    # under the GIL; ctypes and supported native extensions release it in search.
    ticket = itertools.count()
    barrier = threading.Barrier(concurrency + 1)
    stop = threading.Event()
    start = [0.0]
    quota = math.ceil(max(protocol.min_requests, len(queries)) / concurrency)
    warm_quota = math.ceil(protocol.warmup_requests / concurrency)

    def worker(worker_id):
        samples = array("Q")
        try:
            initialize()  # e.g. thread-local OpenMP limits, outside timing
            began = time.perf_counter()
            n = 0
            while (
                n < warm_quota or time.perf_counter() - began < protocol.warmup_seconds
            ):
                result = search(
                    queries[order[(worker_id + n * concurrency) % len(order)]]
                )
                del result
                n += 1
            barrier.wait()
            n = 0
            while not stop.is_set():
                elapsed = time.perf_counter() - start[0]
                if elapsed >= protocol.max_seconds or (
                    n >= quota and elapsed >= protocol.min_seconds
                ):
                    break
                query = queries[order[next(ticket) % len(order)]]
                before = time.perf_counter_ns()
                result = search(query)
                duration = time.perf_counter_ns() - before
                samples.append(duration)
                del result  # destruction and recording are outside request timing
                n += 1
            ended = time.perf_counter()
            return (
                samples,
                ended,
                n >= quota and ended - start[0] >= protocol.min_seconds,
            )
        except BaseException:
            stop.set()
            barrier.abort()  # includes warmup failures; never strand peers
            raise

    def publish_start():
        start[0] = time.perf_counter()

    barrier = threading.Barrier(concurrency + 1, action=publish_start)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(concurrency)]
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass  # retrieve the originating error from the futures
        values, errors = [], []
        for future in futures:
            try:
                values.append(future.result())
            except BaseException as e:
                errors.append(e)
        if errors:
            raise next(
                (e for e in errors if not isinstance(e, threading.BrokenBarrierError)),
                errors[0],
            )
    merged = np.concatenate([np.frombuffer(v[0], dtype=np.uint64) for v in values])
    wall = max(v[1] for v in values) - start[0]
    result = dict(
        latency=distribution(merged),
        wall_seconds=wall,
        instrumented_qps=len(merged) / wall,
        minimum_reached=all(v[2] for v in values),
    )
    return result, merged


def evaluate(search, queries, concurrency, initialize=lambda: None):
    """One complete query pass outside timing, including concurrent result checks."""

    def one(q):
        return tuple(map(int, search(q)))

    with ThreadPoolExecutor(
        max_workers=concurrency, initializer=initialize
    ) as executor:
        return list(executor.map(one, queries))
