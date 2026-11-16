# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Explicit tiny-fixture integration test for all adapters, then cache-only reload.

Requires the optional baseline packages, current Orion wheel, native adapters and
existing diskann_sweep executable. Builds only generated 512-vector test indices.
"""
import argparse
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

import numpy as np

from .data import normalized, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--native-dir', required=True, type=Path)
    p.add_argument('--diskann-sweep', required=True, type=Path)
    p.add_argument('--metric', choices=['squared-l2', 'cosine', 'inner-product'], default='squared-l2')
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    suffix = '.dylib' if platform.system() == 'Darwin' else '.so'
    rng = np.random.default_rng(20261112)
    base = rng.uniform(0, 200, (512, 128)).astype('f4')
    query = base[:12] + rng.uniform(-.1, .1, (12, 128)).astype('f4')
    if args.metric == 'squared-l2':
        distance = ((query[:, None, :] - base[None, :, :])**2).sum(axis=2)
    elif args.metric == 'cosine':
        distance = -(normalized(query) @ normalized(base).T)
    else:
        distance = -(query @ base.T)
    gt = np.argsort(distance, axis=1)[:, :10].astype('<u4')
    for name, matrix in [('base.fbin', base), ('query.fbin', query), ('gt.ibin', gt)]:
        (root/name).write_bytes(struct.pack('<II', *matrix.shape) + matrix.tobytes())
    with (root/'pa.graph').open('wb') as f:
        f.write(struct.pack('<II', len(base), len(base)-1))
        f.write(np.full(len(base), len(base)-1, dtype='<u4').tobytes())
        for i in range(len(base)):
            f.write(np.delete(np.arange(len(base), dtype='<u4'), i).tobytes())
    from orion_ann import Index, GraphConfig
    index = Index.from_file(root/'base.fbin', metric=args.metric, threads=8,
                           graph=GraphConfig(degree=16, build_l=32), cache_dir=root/'orion-cache')
    index.calibrate(query, k=5, search_list_size=32)
    index.save(root/'snapshot')
    del index
    with (root/'diskann-build.log').open('w') as log:
        subprocess.run([str(args.diskann_sweep.resolve()), 'glove100' if args.metric == 'cosine' else 'sift',
            '--base', str(root/'base.fbin'), '--query', str(root/'query.fbin'),
            '--ground-truth', str(root/'gt.ibin'), '--cache-dir', str(root/'diskann-cache'),
            '--max-points', '512', '--threads', '8', '--prepare-only'], check=True,
            stdout=log, stderr=subprocess.STDOUT)
    engines = {}
    for name in ('hnswlib', 'usearch', 'lancedb', 'faiss-hnsw-sq8', 'faiss-ivf-flat', 'faiss-ivf-pq', 'annoy'):
        engines[name] = dict(cache=str(root/'cache'/name), efforts=[2, 4] if name.startswith('faiss-ivf') else [16, 32])
        if name.startswith('faiss-ivf'):
            engines[name]['build'] = dict(nlist=4)
        if name == 'faiss-ivf-pq':
            engines[name]['build'].update(pq_m=4, pq_bits=4)
        if name == 'annoy':
            engines[name]['build'] = dict(trees=5)
    native = args.native_dir.resolve()
    engines['orion'] = dict(snapshot=str(root/'snapshot'), efforts=[16, 32])
    engines['diskann'] = dict(index=str(next((root/'diskann-cache').glob('*.bin'))), efforts=[16, 32],
        library=str(native/'cargo/release'/('libbaseline_latency_diskann'+suffix)))
    engines['parlayann'] = dict(base_fbin=str(root/'base.fbin'), graph=str(root/'pa.graph'),
        library=str(native/('liblatency_parlay'+suffix)), quantize_bits=8 if args.metric == 'squared-l2' else 16,
        quantize_mode=0 if args.metric == 'squared-l2' else 1, rerank_factor=2, efforts=[16, 32])
    config = dict(k=5, threads=8, protocol=dict(concurrency=[1, 8], rounds=3, min_requests=80,
        min_seconds=.02, max_seconds=3, warmup_requests=16, warmup_seconds=.005, cooldown_seconds=.01),
        datasets=[dict(name='fixture', metric=args.metric, base=str(root/'base.fbin'),
            query=str(root/'query.fbin'), groundtruth=str(root/'gt.ibin'), engines=engines)])
    write_json(root/'config.json', config)
    for name, flags in [('prepare', ['--prepare-missing']), ('reload', [])]:
        # Unsupported raw-IP DiskANN is expected; all other missing rows fail below.
        command = [sys.executable, '-m', 'latency_baselines', '--config', str(root/'config.json'),
                   '--output', str(root/name), '--allow-missing', *flags]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
        records = json.loads((root/name/'comparison.json').read_text())['jobs']
        if len(records) != 10:
            raise AssertionError('not every engine was tested')
        for result in records:
            expected = 'unsupported' if args.metric == 'inner-product' and result['engine'] == 'diskann' else 'complete'
            if result['status'] != expected:
                raise AssertionError((result['engine'], result['status'], result.get('error')))
            if expected == 'complete':
                assert len(result['points']) == 4
                assert all(len(p['rounds']) == 3 and p['concurrent_ids_match'] for p in result['points'])
    print('All configured adapters passed prepare, concurrent search and cache-only reload.')


if __name__ == '__main__':
    main()
