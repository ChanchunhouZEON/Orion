# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""One process owns one dataset/engine; one load serves every effort and round."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .adapters import Unsupported, open_adapter
from .data import identity, recall, rows, write_json
from .protocol import Protocol, distribution, evaluate, measure


def versions():
    result = {}
    for name in ('numpy', 'faiss-cpu', 'hnswlib', 'usearch', 'lancedb', 'annoy', 'orion-ann'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def run(job_path):
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    output = job_path.parent
    ds, cfg, engine = job['dataset'], job['engine_config'], job['engine']
    protocol = Protocol(**job.get('protocol', {})).validate()
    record = dict(dataset=ds['name'], engine=engine, status='running', points=[],
                  protocol=protocol.json(), timing_scope='python-single-request-api',
                  config=job, versions=versions(), machine=platform.platform(),
                  python=sys.version, process_id=os.getpid(),
                  thread_environment={k: v for k, v in os.environ.items()
                    if k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                             'PARLAY_NUM_THREADS', 'RAYON_NUM_THREADS', 'LANCE_CPU_THREADS')})
    write_json(output / 'result.json', record)
    adapter = None
    try:
        base = rows(ds['base'], ds.get('base_format'))
        queries = np.array(rows(ds['query'], ds.get('query_format')), dtype=np.float32, order='C', copy=True)
        gt = rows(ds['groundtruth'], ds.get('groundtruth_format'), groundtruth=True)
        k = job['k']
        if (queries.shape[1] != base.shape[1] or not np.isfinite(queries).all()
                or len(gt) != len(queries) or not 1 <= k <= len(base)):
            raise ValueError('invalid query/GT shape, k or values')
        # Validate base/GT once before native code; never silently truncate data.
        for start in range(0, len(base), 8192):
            block = np.asarray(base[start:start+8192], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError('base contains nonfinite values')
            if ds['metric'] == 'cosine' and np.any(np.linalg.norm(block, axis=1) == 0):
                raise ValueError('cosine base contains zero vectors')
        recall([[] for _ in queries], gt, k, len(base))
        if ds['metric'] == 'cosine' and np.any(np.linalg.norm(queries, axis=1) == 0):
            raise ValueError('cosine query contains zero vectors')
        record['base_identity'] = identity(ds, base)
        record['query_sha256'] = hashlib.sha256(queries.tobytes()).hexdigest()
        record['groundtruth_sha256'] = hashlib.sha256(np.ascontiguousarray(gt[:, :k]).tobytes()).hexdigest()
        load_start = time.perf_counter()
        adapter = open_adapter(engine, cfg, ds, base, queries, job['threads'], job['prepare_missing'])
        record['load_or_prepare_seconds'] = time.perf_counter() - load_start
        if hasattr(adapter, 'info'):
            if adapter.info['num_points'] != len(base) or adapter.info['dimension'] != base.shape[1]:
                raise ValueError('Orion index shape differs from comparison inputs')
            record['effective_index'] = adapter.info
        if cfg.get('library'):
            record['library_sha256'] = hashlib.sha256(Path(cfg['library']).read_bytes()).hexdigest()
        order = np.random.default_rng(protocol.seed).permutation(len(queries)).tolist()
        for effort in cfg['efforts']:
            adapter.configure(effort, k)
            if engine == 'orion':
                record['effective_index'] = adapter.index.info
            control = evaluate(adapter.search, queries, 1, adapter.initialize_thread)
            evaluation = recall(control, gt, k, len(base))
            points, samples = {}, {}
            for c in protocol.concurrency:
                concurrent = evaluate(adapter.search, queries, c, adapter.initialize_thread)
                if concurrent != control:
                    raise ValueError(f'{c}-client result IDs differ from serial search at effort={effort}')
                points[c] = dict(effort=effort, concurrency=c, evaluation=evaluation,
                    rounds=[], complete=False, concurrent_ids_match=True)
                samples[c] = []
                record['points'].append(points[c])
            for r in range(protocol.rounds):
                clients = protocol.concurrency if r % 2 == 0 else protocol.concurrency[::-1]
                for c in clients:
                    time.sleep(protocol.cooldown_seconds)
                    trial, raw = measure(adapter.search, queries, order, c, protocol, adapter.initialize_thread)
                    raw_path = output / f'effort-{effort}-c{c}-r{r+1}.npy'
                    np.save(raw_path, raw, allow_pickle=False)
                    trial.update(round=r+1, samples=raw_path.name,
                                 samples_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
                    point = points[c]
                    point['rounds'].append(trial)
                    samples[c].append(raw)
                    point['pooled'] = distribution(np.concatenate(samples[c]))
                    point['median_instrumented_qps'] = float(np.median([t['instrumented_qps'] for t in point['rounds']]))
                    p99 = [t['latency']['p99_us'] for t in point['rounds']]
                    point['p99_round_range_us'] = [min(p99), max(p99)]
                    point['complete'] = len(point['rounds']) == protocol.rounds and all(t['minimum_reached'] for t in point['rounds'])
                    write_json(output / 'result.json', record)
                    print(f"{ds['name']}/{engine} effort={effort} clients={c} round={r+1} "
                          f"recall={evaluation['recall']:.6f} P99={trial['latency']['p99_us']:.2f}us", flush=True)
        record['status'] = 'complete' if all(p['complete'] for p in record['points']) else 'incomplete'
    except (ImportError, FileNotFoundError) as e:
        record.update(status='missing', error=str(e))
    except Unsupported as e:
        record.update(status='unsupported', error=str(e))
    except Exception as e:
        record.update(status='failed', error=f'{type(e).__name__}: {e}')
    finally:
        if adapter is not None:
            adapter.close()
        write_json(output / 'result.json', record)
    return 0 if record['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
