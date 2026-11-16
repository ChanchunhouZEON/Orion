# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Validated, read-only benchmark inputs; decoding is outside request timing."""
import json
from pathlib import Path
import struct

import numpy as np


def rows(path, kind=None, *, groundtruth=False):
    path = Path(path)
    kind = kind or path.suffix.lstrip('.')
    if kind == 'npy':
        a = np.load(path, mmap_mode='r', allow_pickle=False)
    elif kind in ('fvecs', 'ivecs', 'bvecs'):
        dtype = np.dtype({'fvecs': '<f4', 'ivecs': '<i4', 'bvecs': 'u1'}[kind])
        with path.open('rb') as f:
            header = f.read(4)
        if len(header) != 4:
            raise ValueError(f'{path}: truncated header')
        d, = struct.unpack('<I', header)
        width = 4 + d * dtype.itemsize
        if d == 0 or path.stat().st_size % width:
            raise ValueError(f'{path}: incomplete vector rows')
        raw = np.memmap(path, mode='r', dtype='u1').reshape(-1, width)
        if not np.all(raw[:, :4].copy().view('<u4') == d):
            raise ValueError(f'{path}: inconsistent dimension headers')
        a = raw[:, 4:].view(dtype)
    elif kind in ('fbin', 'u8bin', 'ibin', 'bin'):
        dtype = np.dtype('u1' if kind == 'u8bin' else '<u4' if groundtruth else '<f4')
        with path.open('rb') as f:
            header = f.read(8)
        if len(header) != 8:
            raise ValueError(f'{path}: truncated header')
        n, d = struct.unpack('<II', header)
        end = 8 + n * d * dtype.itemsize
        # Ground-truth .bin may append its distance matrix. Never use it for recall.
        allowed = (end, end + n * d * 4) if groundtruth else (end,)
        if not n or not d or path.stat().st_size not in allowed:
            raise ValueError(f'{path}: size/header mismatch')
        a = np.memmap(path, mode='r', dtype=dtype, offset=8, shape=(n, d))
    else:
        raise ValueError(f'unsupported input format: {kind}')
    if a.ndim != 2 or min(a.shape) <= 0:
        raise ValueError('expected a nonempty 2-D matrix')
    if groundtruth and a.dtype.kind not in 'iu':
        raise ValueError('ground truth must contain integer IDs')
    return a


def normalized(a):
    a = np.array(a, dtype=np.float32, order='C', copy=True)
    norm = np.sqrt(np.einsum('ij,ij->i', a, a, dtype=np.float64))
    if not np.all(np.isfinite(norm)) or np.any(norm == 0):
        raise ValueError('cosine requires finite nonzero vectors')
    a /= norm[:, None]
    return a


def identity(dataset, base):
    p = Path(dataset['base']).resolve()
    s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                shape=list(base.shape), dtype=base.dtype.str, metric=dataset['metric'])


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def recall(results, truth, k, count):
    if len(results) != len(truth) or not len(truth) or truth.shape[1] < k:
        raise ValueError('ground truth/query count mismatch or GT width < k')
    hits = partial = 0
    for found, expected in zip(results, truth):
        ids = [int(v) for v in found]
        if len(ids) > k or len(set(ids)) != len(ids) or any(v < 0 or v >= count for v in ids):
            raise ValueError('engine returned duplicate/out-of-range IDs or too many results')
        targets = [int(v) for v in expected[:k]]
        if len(set(targets)) != k or any(v < 0 or v >= count for v in targets):
            raise ValueError('GT contains duplicate/out-of-range IDs; prefix datasets need new GT')
        hits += len(set(ids) & set(targets))
        partial += len(ids) < k
    return dict(recall=hits / (len(truth) * k), partial_queries=partial, queries=len(truth))
