#!/usr/bin/env python3
# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Submit isolated, sequential storage loads without running timed searches."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from experiment import limits, save
from cgroup import MIB


def build_jobs(root, runs, prepare, estimates, partition):
    """Isolate every arm so a cgroup OOM cannot suppress the other checks."""
    jobs = []
    for dataset in ('sift', 'gist'):
        for divisor, limit in limits(estimates[dataset]['estimated_bytes']).items():
            for arm in ('mmap', 'mmap-driver', 'ssd'):
                name = f'{dataset}-one-over-{divisor}-{arm}'
                cmd = ['sbatch', '--parsable', '--partition', partition, '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=8', f'--mem={limit//MIB}M', '--time=01:00:00',
                       '--job-name', name, '--output', str(runs/(name+'-%j.log')),
                       str(root/'storage.sbatch'), str(root), 'run', '--dataset', dataset,
                       '--prepare', str(prepare), '--output', str(runs/name), '--limit-bytes', str(limit),
                       '--rounds', '1', '--concurrencies', '1', '--arms', arm, '--load-only']
                jobs.append({'name':name, 'limit_bytes':limit, 'command':cmd})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--prepare', type=Path, required=True)
    parser.add_argument('--partition', default='MEM_2TB')
    parser.add_argument('--submit', action='store_true', help='default writes a dry-run plan only')
    args = parser.parse_args()
    root = args.root.resolve()
    prepare = args.prepare.resolve()
    if args.submit:
        for name in ('complete.json', 'sift-calibration.json', 'gist-calibration.json'):
            if not (prepare/name).is_file():
                parser.error(f'missing prior preparation file: {prepare/name}')
    runs = root/'runs'/datetime.now().strftime('%Y%m%d-%H%M%S-load-validation-%f')
    runs.mkdir(parents=True, exist_ok=False)
    jobs = build_jobs(root, runs, prepare, json.loads((root/'estimates.json').read_text()), args.partition)
    plan = {'runs':str(runs), 'purpose':'load-only; no searches or new calibration', 'jobs':jobs}
    save(runs/'plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.submit:
        previous = None
        for job in jobs:
            cmd = job['command'].copy()
            if previous:
                cmd.insert(1, '--dependency=afterany:'+previous)
            job_id = subprocess.check_output(cmd, text=True).strip().split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError('Unexpected sbatch response')
            job['job_id'] = previous = job_id
            save(runs/'plan.json', plan)
            print(job['name'], job_id, flush=True)

if __name__ == '__main__':
    main()
