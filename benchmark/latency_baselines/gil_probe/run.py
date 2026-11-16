#!/usr/bin/env python3
# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""GIL attribution using existing snapshots, with same-interpreter controls.

No graph building. The raw traces belong to a diagnostic interpreter and are
not replacements for production latency measurements. Neither perf counters
nor kernel scheduler probes are available: cond_wait includes wake scheduling
and reacquiring its pthread mutex, not just time asleep.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['RAYON_NUM_THREADS'] = '8'
import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import threading
import time
import _gilprobe as probe
import numpy as np


def measure(index, requests, order, concurrency, effort, mode):
    start = [0.0]
    gate = threading.Barrier(concurrency + 1, action=lambda: start.__setitem__(0, time.perf_counter()))
    done = threading.Barrier(concurrency)
    tickets = itertools.count()
    stop = threading.Event()
    quota = (10000 + concurrency - 1) // concurrency

    def worker(worker_id):
        search = index.search
        try:
            began = time.perf_counter()
            n = 0
            while n < 1000 // concurrency or time.perf_counter() - began < .2:
                result = search(requests[order[(worker_id+n*concurrency)%len(order)]],
                                k=10, search_list_size=effort)
                del result
                n += 1
            index._handle._start_search_timing(200000)
            identity = probe.start(1000000) if mode == 'traced' else None
            samples = array('Q')
            gate.wait(timeout=60)
            n = 0
            while not stop.is_set():
                elapsed = time.perf_counter() - start[0]
                if elapsed >= 30 or (elapsed >= 1 and n >= quota): break
                query = requests[order[next(tickets) % len(order)]]
                before = time.perf_counter_ns()
                if mode != 'bare': probe.begin(n + 1)
                result = search(query, k=10, search_list_size=effort)
                if mode != 'bare': probe.end()
                samples.append(time.perf_counter_ns() - before)
                del result
                n += 1
            ended = time.perf_counter()
            # Drain only after every worker exits the timed region. Converting
            # trace bytes / Python timing tuples must not delay live queries.
            done.wait(timeout=60)
            raw, dropped = probe.finish()
            native = np.asarray(index._handle._finish_search_timing(), dtype=np.uint64)
            latency = np.frombuffer(samples, dtype=np.uint64).copy()
            if dropped: raise RuntimeError(f'trace overflow: {dropped}')
            if n < quota or ended - start[0] < 1: raise RuntimeError('sample minimum not reached')
            assert native.shape == (n, 6)
            assert np.all(latency >= native[:, 5])
            events = np.frombuffer(raw, dtype=np.uint64).reshape(-1, 4).copy()
            return dict(worker=worker_id, tstate=identity, ended=ended), np.column_stack((latency, native)), events
        except BaseException:
            stop.set()
            gate.abort()
            done.abort()
            raise
        finally:
            probe.finish()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(concurrency)]
        try: gate.wait(timeout=60)
        except threading.BrokenBarrierError: pass
        values = [future.result() for future in futures]
    wall = max(item[0]['ended'] for item in values) - start[0]
    return values, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='Historical experiment date directory')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    sys.path[:0] = [str(args.package), str(args.repo)]
    from orion_ann import Index
    from benchmark.latency_baselines.data import rows, recall, write_json
    from benchmark.latency_baselines.protocol import distribution
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'metadata.json', dict(
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        switch_interval_seconds=sys.getswitchinterval(), rounds=args.rounds,
        source='CPython 3.9.19 with bounded TLS tracing; no PGO, not Conda Python',
        native_sha256=hashlib.sha256((args.package/'orion_ann/_native.abi3.so').read_bytes()).hexdigest(),
        modes=['bare', 'marked', 'traced'], native_feature='search-timing',
        clocks='C CLOCK_MONOTONIC ns shared across threads; Rust relative Instant durations; never subtract clocks from different domains',
        event_columns=['monotonic_ns', 'request_id_per_worker', 'event', 'argument'],
        sample_columns=['end_to_end_ns','input_ns','release_ns','native_ns','gil_return_ns','wrap_ns','ffi_total_ns'],
        sampling='closed-loop C=1/8, >=1s and >=10000 requests per point, warmup >=.2s, cooldown1s',
        caveats=['Conditional wait includes pthread mutex reacquisition and OS scheduling.',
                 'A notify event does not identify its recipient.',
                 'GIL ownership wall time includes time the owner was descheduled.',
                 'Trace clocks inside gil_mutex perturb contention; inspect overhead controls.']))
    records = []
    for dataset, efforts in [('sift', [25,48,76]), ('glove25', [37,68,106])]:
        config = json.loads((args.inputs/'search-result-wrapper-ab'/f'r1-{dataset}-optimized/job.json').read_text())['dataset']
        queries = np.array(rows(config['query']), dtype=np.float32, order='C', copy=True)
        truth = rows(config['groundtruth'], groundtruth=True)
        requests = [queries[i:i+1] for i in range(len(queries))]
        index = Index.open(args.inputs/'single-query-dispatch-ab'/f'{dataset}-snapshot',
                           base=config['base'], threads=8, auto_pin=True)
        order = np.random.default_rng(20261116).permutation(len(queries)).tolist()
        recalls = {}
        for effort in efforts:
            batch = index.search(queries, k=10, search_list_size=effort)
            ids = [list(batch.ids[i,:int(batch.counts[i])]) for i in range(len(queries))]
            recalls[effort] = recall(ids, truth, 10, index.info['num_points'])['recall']
            # Verify all queries against the shared batch reference, with GIL
            # tracing on, outside timed work. No calibration or rebuild occurs.
            def verify(i):
                probe.start(1000)
                try:
                    probe.begin(i+1)
                    result = index.search(requests[i], k=10, search_list_size=effort)
                    probe.end()
                    assert list(result.ids[0,:int(result.counts[0])]) == ids[i]
                finally:
                    _, lost = probe.finish()
                    assert lost == 0
            with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(verify, range(len(queries))))
        print(dataset, 'all query IDs/counts verified', recalls, flush=True)
        for rnd in range(1, args.rounds+1):
            for j, effort in enumerate(efforts if rnd%2 else efforts[::-1]):
                for concurrency in ([1,8] if rnd%2 else [8,1]):
                    modes = ['bare','marked','traced']
                    offset = (rnd+j) % 3
                    modes = modes[offset:] + modes[:offset]
                    for mode in modes:
                        time.sleep(1)
                        values, wall = measure(index, requests, order, concurrency, effort, mode)
                        name = f'{dataset}-L{effort}-c{concurrency}-r{rnd}-{mode}'
                        folder = args.output/name
                        folder.mkdir()
                        samples = np.concatenate([item[1] for item in values])
                        workers = []
                        for info, sample, events in values:
                            np.save(folder/f"worker{info['worker']}-samples.npy", sample)
                            np.save(folder/f"worker{info['worker']}-events.npy", events)
                            workers.append(info)
                        write_json(folder/'workers.json', workers)
                        stat = dict(name=name, dataset=dataset, effort=effort, concurrency=concurrency,
                                    round=rnd, mode=mode, recall=recalls[effort], requests=len(samples),
                                    wall_seconds=wall, qps=len(samples)/wall,
                                    latency=distribution(samples[:,0]), gil_return=distribution(samples[:,4]))
                        records.append(stat)
                        write_json(args.output/'results.json', records)
                        print(name, round(stat['qps']), 'P99',stat['latency']['p99_us'], flush=True)
        del index
    write_json(args.output/'status.json', dict(status='complete', trials=len(records)))


if __name__ == '__main__': main()
