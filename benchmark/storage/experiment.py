# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.


"""Run controlled storage trials inside the allocated Slurm job step."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import cgroup

ARMS = ['mmap', 'mmap-driver', 'ssd']


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * cgroup.MIB), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(root):
    expected = json.loads((root / 'checksums.json').read_text())
    for name, record in expected.items():
        path = root / name
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Checksum mismatch: {name}')
    print(f'Verified {len(expected)} deployment files', flush=True)


def limits(estimated):
    if estimated <= 0:
        raise ValueError('expected a positive resident estimate')
    return {divisor: math.ceil(estimated / divisor / cgroup.MIB) * cgroup.MIB
            for divisor in (2, 4)}


def command(root, dataset, calibration, arm, output, work, concurrency, capacity, seconds):
    return [os.environ.get("ORION_STORAGE_BENCH", str(root / "bin/orion-storage-bench")), 'run', '--dataset', str(dataset),
            '--calibration', str(calibration), '--arm', arm, '--output', str(output),
            '--work', str(work), '--concurrency', str(concurrency),
            '--page-cache-mib', str(capacity), '--seconds', str(seconds)]


def summaries(directory):
    rows = []
    for path in sorted(directory.glob('r*-*.jsonl')):
        # Telemetry shares case prefixes but has a different record schema.
        if path.name.endswith('.telemetry.jsonl'):
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row['file'] = path.name
                rows.append(row)
    if not rows:
        raise RuntimeError("No search result records found")
    groups = {}
    checks = {}
    for row in rows:
        key = (row['concurrency'], row['L'])
        checks.setdefault(key, set()).add(row['validation_sha256'])
        groups.setdefault((*key, row['arm']), []).append(row)
    mismatch = [key for key, values in checks.items() if len(values) != 1]
    if mismatch:
        raise RuntimeError(f'Ordered neighbor IDs differ between trials: {mismatch}')
    table = ['| concurrency | L | arm | rounds | recall | median QPS | P50 us | P95 us | P99 us | gain vs mmap |',
             '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for (c, beam, arm), values in sorted(groups.items()):
        median = lambda name: statistics.median(row[name] for row in values)
        ref = groups.get((c, beam, 'Mmap'), [])
        gain = f"{(median('qps')/statistics.median(x['qps'] for x in ref)-1)*100:+.2f}%" if ref else 'n/a'
        note = '*' if any(x['undersampled'] for x in values) else ''
        table.append(f"| {c} | {beam} | {arm}{note} | {len(values)} | {median('recall'):.6f} | {median('qps'):,.1f} | {median('p50_us'):.1f} | {median('p95_us'):.1f} | {median('p99_us'):.1f} | {gain} |")
    table += ['', '* Fewer than 10,000 timed requests in one or more rounds; tail estimates are exploratory.',
              'Quantiles are medians of per-round quantiles. Search latency excludes preparation and validation/warmup.',
              'SSD denotes buffered positional graph reads plus a bounded cache, not io_uring or O_DIRECT.']
    (directory / 'summary.md').write_text('\n'.join(table) + '\n')
    save(directory / 'summary.json', {'rows': rows, 'ordered_ids_match': True})


def verify_isa():
    """Reject an incompatible compute node before executing AVX-512 code."""
    flags = set(Path('/proc/cpuinfo').read_text().split())
    required = {'avx2', 'fma', 'f16c', 'avx512f', 'avx512bw', 'avx512dq', 'avx512vl', 'avx512_vpopcntdq'}
    missing = required - flags
    if missing:
        raise RuntimeError('Compute node lacks required ISA features: ' + ', '.join(sorted(missing)))


def prepare(root, directory):
    directory.mkdir(parents=True, exist_ok=True)
    verify_isa()
    verify_files(root)
    # Execute all arms over the tiny fixture before touching full experiment data.
    binary = root / 'bin/orion-storage-bench'
    fixture = directory / 'fixture'
    subprocess.run([str(binary), 'fixture', '--directory', str(fixture)], check=True)
    expected_estimates = json.loads((root/'estimates.json').read_text())
    for dataset in ('sift', 'gist'):
        actual = json.loads(subprocess.check_output([str(binary), 'estimate', '--dataset',
                            str(root/'data'/dataset/'dataset.json')], text=True))
        if actual != expected_estimates[dataset]:
            raise RuntimeError('Compute-node memory estimate differs from submitted plan: ' + dataset)
        check = directory / ('smoke-' + dataset)
        check.mkdir()
        for concurrency in (1, 8):
            for arm in ARMS:
                output = check / f'r1-c{concurrency}-{arm}.jsonl'
                subprocess.run(command(root, fixture / f'{dataset}.json', fixture / 'calibration.json',
                                       arm, output, check / 'work', concurrency, 1, 0.01), check=True)
        summaries(check)
        with (directory / f'{dataset}-calibration.log').open('w') as log:
            subprocess.run([str(binary), 'calibrate', '--dataset', str(root / 'data' / dataset / 'dataset.json'),
                            '--work', str(directory / 'work'), '--output', str(directory / f'{dataset}-calibration.json')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    save(directory / 'complete.json', {'status': 'complete'})


def trial(args):
    root, directory = args.root.resolve(), args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state = {'status': 'checking_limits', 'dataset': args.dataset, 'limit_bytes': args.limit_bytes,
             'rounds': args.rounds, 'started': time.time(), 'cases': []}
    save(directory / 'state.json', state)
    try:
        info = cgroup.verify(args.limit_bytes)
        verify_isa()
        os.environ['ORION_STORAGE_CGROUP'] = info['limiting_cgroup']
        state['cgroup'] = info
        state['host'] = subprocess.check_output(['uname', '-a'], text=True).strip()
        state['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        if len(state['cpu_affinity']) < 8:
            raise RuntimeError('Fewer than 8 CPUs allocated')
        # Restrict helper threads as well as search workers to the same 8 CPUs.
        os.sched_setaffinity(0, state['cpu_affinity'][:8])
        state['measured_cpu_affinity'] = sorted(os.sched_getaffinity(0))
        state['filesystem'] = subprocess.check_output(['stat', '-f', '-c', '%T', str(root)], text=True).strip()
        if not (args.prepare / 'complete.json').is_file():
            raise RuntimeError('Preparation/calibration job has not completed')
        capacity = max(1, min(128, args.limit_bytes // 8 // cgroup.MIB))
        state['page_cache_mib'] = capacity
        state['status'] = 'running'
        state['load_only'] = args.load_only
        # Rotate arm order over three rounds. No concurrent trials or host-wide cache drops.
        for round_index in range(args.rounds):
            for concurrency in args.concurrencies:
                offset = round_index % len(args.arms)
                order = args.arms[offset:] + args.arms[:offset]
                if concurrency == 8:
                    order = list(reversed(order))
                for arm in order:
                    name = f'r{round_index+1}-c{concurrency}-{arm}'
                    case = {'name': name, 'status': 'running', 'started': time.time(),
                            'before': cgroup.snapshot(info)}
                    state['cases'].append(case)
                    save(directory / 'state.json', state)
                    output = directory / (name + '.jsonl')
                    cmd = command(root, root / 'data' / args.dataset / 'dataset.json',
                                  args.prepare / f'{args.dataset}-calibration.json', arm, output,
                                  directory / 'work', concurrency, capacity, args.seconds)
                    if args.load_only:
                        cmd.append('--load-only')
                    case['command'] = cmd
                    save(directory / 'state.json', state)
                    with (directory / (name + '.log')).open('w') as log, (directory / (name + '.telemetry.jsonl')).open('w') as metrics:
                        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                        while p.poll() is None:
                            sample = {'time': time.time(), 'pid': p.pid, 'process': cgroup.process_rss(p.pid),
                                      'cgroup': cgroup.snapshot(info)}
                            metrics.write(json.dumps(sample) + '\n'); metrics.flush()
                            time.sleep(1)
                        case['returncode'] = p.returncode
                    case.update(status='complete' if p.returncode == 0 else 'failed',
                                seconds=time.time()-case['started'], after=cgroup.snapshot(info))
                    save(directory / 'state.json', state)
                    if p.returncode != 0:
                        raise RuntimeError(f'{name} exited {p.returncode}; inspect log and cgroup events')
                    if args.load_only:
                        records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
                        if len(records) != 1 or records[0].get('kind') != 'load_validation' or records[0].get('status') != 'complete':
                            raise RuntimeError('Invalid load-validation completion record')
                        case['load_result'] = records[0]
                    else:
                        summaries(directory)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state['updated'] = time.time()
        save(directory / 'state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--dataset', choices=['sift', 'gist'])
    parser.add_argument('--limit-bytes', type=int)
    parser.add_argument('--load-only', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--concurrencies', nargs='+', type=int, choices=range(1, 9), default=[1, 8])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=30)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root.resolve(), args.output.resolve())
    elif args.mode == 'summarize':
        summaries(args.output)
    else:
        if not args.prepare or not args.dataset or not args.limit_bytes or args.rounds < 1 or args.seconds <= 0:
            parser.error('run requires preparation, dataset, positive limit, rounds and duration')
        trial(args)

if __name__ == '__main__':
    main()
