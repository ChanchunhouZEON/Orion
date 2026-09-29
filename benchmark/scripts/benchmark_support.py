"""Shared paths, vector I/O and evaluation for standalone benchmark scripts.

Preset resolution belongs to Rust; Python consumes its JSON contract instead of
maintaining another copy of sweep.yaml. Importing this module performs no builds.
"""
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DATASETS = ('sift', 'glove25', 'glove100', 'gist', 'deep10m',
                     'msmarco_bert_1M', 'wiki_ada_1M')


def binary_path(name):
    target = Path(os.environ.get('BUILD_DIR', os.environ.get('CARGO_TARGET_DIR', 'target')))
    return ROOT / target / 'release' / name


def dataset_paths(name):
    binary = Path(os.environ.get('ORION_CONFIG_BIN', binary_path('orion')))
    binary = ROOT / binary
    if not binary.is_file():
        raise FileNotFoundError(f'{binary}: build with cargo build --release --bin orion '
                                'or set ORION_CONFIG_BIN')
    result = subprocess.run([str(binary), name, '--print-config'], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    config = json.loads(result.stdout)
    paths = {}
    for key, field in (('base', 'base'), ('query', 'query'), ('gt', 'groundtruth')):
        if not config.get(field):
            raise ValueError(f'{name}: baseline evaluation requires {field}')
        paths[key] = str(ROOT / config[field])
    # Cosine and raw inner product must stay distinct for non-unit-norm vectors.
    metric = {'l2': 'l2', 'cosine': 'cos', 'inner-product': 'ip', 'ip': 'ip'}[config['metric']]
    paths.update(metric=metric, space={'l2': 'l2', 'cos': 'cosine', 'ip': 'ip'}[metric])
    return paths


def _read_vecs(path, dtype, max_n):
    if max_n < 0:
        raise ValueError('max_n must be nonnegative')
    with open(path, 'rb') as stream:
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError(f'{path}: missing dimension header')
        dim, = struct.unpack('<i', header)
        size = os.fstat(stream.fileno()).st_size
        if dim <= 0 or size % (4 * (dim + 1)):
            raise ValueError(f'{path}: invalid dimension or incomplete vector record')
        total = size // (4 * (dim + 1))
        count = min(total, max_n) if max_n else total
        stream.seek(0)
        # count is applied to I/O, not just the returned slice (critical at 1B).
        records = np.fromfile(stream, dtype='<i4', count=count * (dim + 1))
        if records.size != count * (dim + 1):
            raise ValueError(f'{path}: file changed while reading')
        records = records.reshape(count, dim + 1)
        if np.any(records[:, 0] != dim):
            raise ValueError(f'{path}: inconsistent per-record dimensions')
        return records[:, 1:].view(dtype).copy(), count, dim


def read_fvecs(path, max_n=0):
    return _read_vecs(path, '<f4', max_n)


def read_ivecs(path, max_n=0):
    return _read_vecs(path, '<i4', max_n)[0]


def recall_at_k(results, ground_truth, k):
    if k <= 0 or not len(results) or len(results) != len(ground_truth):
        raise ValueError('Recall requires positive k and matching nonempty query counts')
    hits = 0
    for result, truth in zip(results, ground_truth):
        if len(truth) < k:
            raise ValueError('Ground truth has fewer than k neighbors')
        hits += len(set(map(int, result[:k])) & set(map(int, truth[:k])))
    return hits / (len(results) * k)


def exact_top_k(base, queries, k, metric='l2', chunk=8192, query_batch=64):
    """Bound both dimensions of temporary distance matrices; return IDs/distances.

    Negative dot product is the distance for IP. Cosine normalizes one base chunk
    at a time, keeping the original data unchanged and avoiding a second full base.
    """
    if metric not in ('l2', 'ip', 'cos') or min(k, chunk, query_batch) <= 0:
        raise ValueError('Invalid metric, k or batch size')
    if base.ndim != 2 or queries.ndim != 2 or base.shape[1] != queries.shape[1] or k > len(base):
        raise ValueError('Dimension mismatch or k exceeds base count')
    ids = np.empty((len(queries), k), dtype=np.uint32)
    distances = np.empty((len(queries), k), dtype=np.float32)
    for start in range(0, len(queries), query_batch):
        q = queries[start:start + query_batch].astype(np.float64)
        if metric == 'cos':
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError('Cosine requires nonzero vectors')
            q /= norms
        best_d = np.empty((len(q), 0))
        best_ids = np.empty((len(q), 0), dtype=np.uint32)
        for offset in range(0, len(base), chunk):
            b = base[offset:offset + chunk].astype(np.float64)
            if metric == 'cos':
                norms = np.linalg.norm(b, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError('Cosine requires nonzero vectors')
                b /= norms
            d = -(q @ b.T)
            if metric == 'l2':
                d = np.maximum(0, 2 * d + (q * q).sum(1)[:, None] + (b * b).sum(1)[None, :])
            elif metric == 'cos':
                d += 1
            if not np.isfinite(d).all():
                raise ValueError('Distances must be finite')
            candidates = np.concatenate((best_d, d), axis=1)
            candidate_ids = np.concatenate((best_ids, np.broadcast_to(
                np.arange(offset, offset + len(b), dtype=np.uint32), d.shape)), axis=1)
            keep = min(k, candidates.shape[1])
            order = np.argpartition(candidates, keep - 1, axis=1)[:, :keep]
            best_d = np.take_along_axis(candidates, order, axis=1)
            best_ids = np.take_along_axis(candidate_ids, order, axis=1)
        order = np.argsort(best_d, axis=1)
        ids[start:start + len(q)] = np.take_along_axis(best_ids, order, axis=1)
        distances[start:start + len(q)] = np.take_along_axis(best_d, order, axis=1)
    return ids, distances


def evaluation_ground_truth(base, queries, gt, k, metric):
    if (gt.shape[0] == len(queries) and gt.shape[1] >= k
            and np.all(gt >= 0) and np.all(gt < len(base))):
        return gt
    print(f'Recomputing {metric} ground truth against {len(base)} base vectors...')
    return exact_top_k(base, queries, k, metric)[0]
