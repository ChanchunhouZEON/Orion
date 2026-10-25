#!/usr/bin/env python3
"""Collect full-query k=100 baselines without modifying existing experiment artifacts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np
from run_result_size_comparison import parse_points

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["sift", "glove25", "glove100", "gist", "deep10m", "msmarco_bert_1M"]
SCHEDULE = [100, 112, 128, 144, 160, 180, 200, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048]
PA_ROW = re.compile(r"search: Q=(\d+), k=(\d+), limit=(\d+), recall=([0-9.eE+-]+), visited=.*?, comparisons=.*?, QPS=([0-9.eE+-]+)")


def parse_pa_configurations(text, k):
    rows = {}
    for q, found_k, limit, recall, qps in PA_ROW.findall(text):
        if int(found_k) != k:
            continue
        key = (int(q), int(limit))
        r, speed = float(recall), float(qps)
        if not 0 <= r <= 1 or speed <= 0:
            raise ValueError("Invalid ParlayANN point")
        rows.setdefault(key, []).append((r, speed))
    if not rows:
        raise ValueError("No ParlayANN configurations parsed")
    return {key: (statistics.median(p[0] for p in values),
                  statistics.median(p[1] for p in values)) for key, values in rows.items()}


def recipe(dataset, pa_root):
    command = ['bash', '-c',
               'source benchmark/scripts/_parlay.sh; load_parlay_recipe "$1"; '
               'printf "%s\\0" "$PA_DIR_NAME" "$PA_BUILD_R" "$PA_BUILD_ALPHA" '
               '"$PA_DIST_FUNC" "$PA_NORMALIZE_FLAG" "${PA_QUERY_ARGS[@]}"',
               'recipe', dataset]
    fields = subprocess.check_output(command, cwd=ROOT).decode().rstrip('\0').split('\0')
    directory, degree, alpha, metric, normalize, *query_args = fields
    data = pa_root / 'data' / directory
    graph = data / 'graphs' / f'graph_{degree}_{float(alpha):g}'
    inputs = {}
    for key, filename in [('base', 'base.fbin'), ('query', 'query.fbin'), ('gt', 'gt.bin')]:
        path = data / filename
        if not path.is_file() and path.is_symlink():
            relocated = ROOT / 'data' / dataset / path.readlink().name
            if relocated.is_file():
                path = relocated
        inputs[key] = path
    return dict(data=data, graph=graph, metric=metric, normalize=normalize, query_args=query_args, **inputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--orion-source', type=Path, required=True)
    ap.add_argument('--pa-root', type=Path, default=ROOT.parent / 'ParlayANN')
    ap.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    ap.add_argument('--engines', nargs='+', choices=['diskann', 'parlayann', 'faiss'], default=['diskann', 'parlayann', 'faiss'])
    ap.add_argument('--extend-parlay', action='store_true', help='Append native five-trial fixed-beam runs at 1536 and 2048')
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = args.orion_source.resolve()
    pa_root = args.pa_root.resolve()
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        created_at=datetime.datetime.now().astimezone().isoformat(), k=100, threads=8,
        trials=5, cooldown_s=1, orion_source=str(src), commands={}, status={},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        binaries={})
    assert manifest['orion_source'] == str(src)
    env = dict(os.environ, PARLAY_NUM_THREADS='8', ORION_CONFIG_BIN=str(ROOT / 'target/release/orion'),
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8',
               PYTHONUNBUFFERED='1')
    recipes = {}
    for dataset in args.datasets:
        config = json.loads((src / f'{dataset}_k100.config.json').read_text())
        pa = recipe(dataset, pa_root)
        for path in [pa['graph'], pa['base'], pa['query'], pa['gt']]:
            if not path.is_file():
                raise FileNotFoundError(path)
        nq, width = np.fromfile(pa['gt'], dtype='<u4', count=2)
        gt = np.memmap(ROOT / config['groundtruth'], dtype='<u4', mode='r')
        gt_width = int(gt[0])
        gt = gt.reshape(-1, gt_width + 1)
        pa_gt = np.memmap(pa['gt'], dtype='<u4', mode='r', offset=8, shape=(int(nq), int(width)))
        assert int(nq) == config['num_queries'] == len(gt)
        assert width >= 100 and gt_width >= 100
        assert np.array_equal(pa_gt[:, :100], gt[:, 1:101]), f'{dataset}: PA GT mismatch'
        for key in ['base', 'query']:
            n, dim = map(int, np.fromfile(pa[key], dtype='<i4', count=2))
            vectors = np.memmap(pa[key], dtype='<f4', mode='r', offset=8, shape=(n, dim))
            rust_path = ROOT / config[key]
            rust_dim = int(np.fromfile(rust_path, dtype='<i4', count=1)[0])
            rust_vectors = np.memmap(rust_path, dtype='<f4', mode='r').reshape(-1, rust_dim + 1)[:, 1:]
            assert n == len(rust_vectors) and dim <= rust_dim
            ids = np.linspace(0, n - 1, 1000, dtype=int)
            a, b = np.array(vectors[ids]), np.array(rust_vectors[ids])
            if pa['normalize']:
                a /= np.linalg.norm(a, axis=1, keepdims=True)
            assert np.allclose(a, b[:, :dim], rtol=2e-5, atol=2e-6)
            assert np.all(b[:, dim:] == 0), 'Extra Rust dimensions must be zero padding'
        recipes[dataset] = pa
    for binary in [ROOT / 'target/release/diskann_sweep', pa_root / 'algorithms/vamana/neighbors']:
        manifest['binaries'][str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    def run(name, cmd, cwd=ROOT):
        if manifest['status'].get(name) == 'completed':
            return (out / f'{name}.log').read_text()
        manifest['commands'][name] = cmd
        manifest['status'][name] = 'running'
        save()
        print(f'Running {name}', flush=True)
        time.sleep(1)
        with (out / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest['status'][name] = 'completed' if result.returncode == 0 else 'failed'
        save()
        if result.returncode:
            raise RuntimeError(f'{name}: see log')
        print(f'Completed {name}', flush=True)
        return (out / f'{name}.log').read_text()

    if args.extend_parlay:
        for dataset in args.datasets:
            path = out / f'{dataset}_parlayann.json'
            payload = json.loads(path.read_text())
            payload.setdefault('extended_runs', {})
            def measure_beam(beam):
                cmd = manifest['commands'][f'{dataset}_parlayann_1'] + ['-Q', str(beam)]
                text = run(f'{dataset}_parlayann_Q{beam}', cmd)
                measured = parse_pa_configurations(text, 100)
                assert len(PA_ROW.findall(text)) == 5 and len(measured) == 1
                (q, limit), (recall, qps) = next(iter(measured.items()))
                assert q == beam
                payload['points'] = [p for p in payload['points'] if (p['L'], p['limit']) != (q, limit)]
                payload['points'].append(dict(L=q, limit=limit, recall=recall, qps=qps))
                payload['extended_runs'][str(beam)] = PA_ROW.findall(text)
                path.write_text(json.dumps(payload, indent=2) + '\n')
            for beam in [1536, 2048]:
                measure_beam(beam)
            if 'refinement_beams' not in payload:
                max_limit = max(p['limit'] for p in payload['points'])
                full = sorted((p for p in payload['points'] if p['limit'] == max_limit), key=lambda p: p['L'])
                beams = set()
                for target in [.95, .97, .99]:
                    crossing = next((i for i, p in enumerate(full) if p['recall'] >= target), None)
                    if crossing is not None and crossing > 0:
                        lo, hi = full[crossing - 1]['L'], full[crossing]['L']
                        beams.update(lo + (hi - lo) * j // 4 for j in [1, 2, 3])
                payload['refinement_beams'] = sorted(beams - {p['L'] for p in full})
                path.write_text(json.dumps(payload, indent=2) + '\n')
            for beam in payload['refinement_beams']:
                measure_beam(beam)
        print('Completed extended ParlayANN beams', flush=True)
        return

    # Timing jobs run serially. FAISS builds occur after the cached graph sweeps.
    for engine in args.engines:
        for dataset in args.datasets:
            if engine == 'diskann':
                if dataset == 'msmarco_bert_1M':
                    manifest['status'][f'{dataset}_diskann'] = 'excluded_metric_mismatch'
                    save()
                    continue
                cmd = [str(ROOT / 'target/release/diskann_sweep'), dataset, '--k', '100', '--trials', '5',
                       '--search-list-sizes', ','.join(map(str, SCHEDULE))]
                text = run(f'{dataset}_diskann', cmd)
                points = parse_points(text, 100, SCHEDULE)
                payload = dict(k=100, threads=8, trials=5, points=points)
            elif engine == 'parlayann':
                pa = recipes[dataset]
                cmd = [str(pa_root / 'algorithms/vamana/neighbors'), *pa['query_args'], '-k', '100',
                       '-data_type', 'float', '-dist_func', pa['metric'], '-file_type', 'bin',
                       '-base_path', str(pa['base']), '-query_path', str(pa['query']),
                       '-gt_path', str(pa['gt']), '-graph_path', str(pa['graph'])]
                if pa['normalize']:
                    cmd.append(pa['normalize'])
                runs = [parse_pa_configurations(run(f'{dataset}_parlayann_{i}', cmd), 100) for i in range(1, 6)]
                if any(set(r) != set(runs[0]) for r in runs):
                    raise ValueError(f'{dataset}: PA configuration sets differ')
                points = [dict(L=q, limit=limit, recall=statistics.median(r[(q, limit)][0] for r in runs),
                               qps=statistics.median(r[(q, limit)][1] for r in runs))
                          for q, limit in runs[0]]
                payload = dict(k=100, threads=8, trials=5, points=points,
                               raw_runs=[{str(key): value for key, value in r.items()} for r in runs])
            else:
                path = out / f'{dataset}_faiss.json'
                cmd = [sys.executable, '-B', str(ROOT / 'benchmark/scripts/dbms_baselines.py'),
                       '--dataset', dataset, '--engines', 'faiss', '--k', '100', '--threads', '8',
                       '--trials', '5', '--max-points', '0', '--output', str(path),
                       '--ef-list', *map(str, SCHEDULE)]
                run(f'{dataset}_faiss', cmd)
                payload = json.loads(path.read_text())
                assert payload['k'] == 100 and payload['threads'] == 8
                continue
            (out / f'{dataset}_{engine}.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Completed: {out}', flush=True)


if __name__ == '__main__':
    main()
