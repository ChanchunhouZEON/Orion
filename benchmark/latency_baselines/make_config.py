# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Reuse explicit Orion latency configs and existing ParlayANN benchmark recipes.

Writes configuration only. Does not load indices, build graphs or run benchmarks.
Missing/ambiguous DiskANN caches remain explicit missing rows in the runner.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import sys

from .data import rows, write_json
from .protocol import Protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orion-configs', type=Path, required=True)
    parser.add_argument('--parlay-root', type=Path, required=True)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--diskann-cache', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reuse the recorded build/query recipes; no second hard-coded recipe table.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_k100_baselines import recipe
    suffix = '.dylib' if platform.system() == 'Darwin' else '.so'
    datasets = []
    k_values = set()
    for file in sorted(args.orion_configs.glob('*.json')):
        src = json.loads(file.read_text())
        if 'index' not in src or 'dataset' not in src:
            continue
        k = src['k']
        k_values.add(k)
        ds = dict(name=src['dataset'], base=src['base'], query=src['query'], groundtruth=src['groundtruth'],
                  metric={'l2': 'squared-l2', 'ip': 'inner-product'}.get(src['index']['metric'], src['index']['metric']))
        base = rows(ds['base'])
        nlist = min(256, max(1, len(base)//40))
        beams = sorted({k, *[v for v in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536) if v >= k]})
        engines = {'orion': dict(experiment_config=str(file.resolve()), efforts=src['beams'],
                                 calibration_l=src['calibration_l'], calibration_samples=src['calibration_samples'])}
        pa = recipe(ds['name'], args.parlay_root.resolve())
        options = pa['query_args']
        def option(key, default):
            return int(options[options.index(key)+1]) if key in options else default
        engines['parlayann'] = dict(library=str(args.native_dir.resolve()/('liblatency_parlay'+suffix)),
            base_fbin=str(pa['base']), graph=str(pa['graph']), quantize_bits=option('-quantize_bits', 0),
            quantize_mode=option('-quantize_mode', 0), rerank_factor=option('-rerank_factor', 10),
            normalize=bool(pa['normalize']), efforts=beams)
        matches = [p for root in args.diskann_cache for p in root.glob(f"**/{ds['name']}_n{len(base)}_*_l2.bin")]
        if len(matches) == 1:
            index = matches[0]
            match = re.search(r'_r(\d+)_l(\d+)_a(\d+)_(\d+)_l2\.bin$', index.name)
            if not match:
                raise ValueError(f'cannot infer explicit legacy recipe: {index}')
            degree, build_l, major, minor = match.groups()
            engines['diskann'] = dict(index=str(index.resolve()), efforts=beams,
                build=dict(degree=int(degree), build_l=int(build_l), alpha=float(major+'.'+minor)),
                unit_norm_ip=ds['metric'] == 'inner-product',
                library=str(args.native_dir.resolve()/'cargo/release'/('libbaseline_latency_diskann'+suffix)))
        else:
            print(f"{ds['name']}: {len(matches)} matching DiskANN caches; configure index explicitly", file=sys.stderr)
        for engine in ('hnswlib', 'usearch', 'lancedb', 'faiss-hnsw-sq8', 'faiss-ivf-flat', 'faiss-ivf-pq', 'annoy'):
            efforts = ([v for v in (1, 2, 4, 8, 16, 32, 64, 128, 256) if v <= nlist]
                       if engine.startswith('faiss-ivf') else
                       [100, 500, 1000, 5000, 10000, 50000, 100000] if engine == 'annoy' else beams)
            engines[engine] = dict(cache=str(args.cache_root.resolve()/ds['name']/engine), efforts=efforts)
        ds['engines'] = engines
        datasets.append(ds)
    if not datasets or len(k_values) != 1:
        parser.error('expected nonempty Orion latency configs with a common k')
    if args.output.exists():
        parser.error('output already exists; edit explicitly rather than overwrite')
    write_json(args.output, dict(k=k_values.pop(), threads=8, protocol=Protocol().json(),
                                recall_targets=[.9, .95, .98, .99], datasets=datasets))


if __name__ == '__main__':
    main()
