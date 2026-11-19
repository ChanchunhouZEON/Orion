# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.


"""Plan or submit sequential SIFT/GIST storage experiments. No SSH required."""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from experiment import limits, save
from cgroup import MIB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit', action='store_true', help='actually call sbatch; default only writes the plan')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--partition', default='MEM_2TB')
    p.add_argument('--seconds', type=float, default=30)
    a = p.parse_args()
    root = a.root.resolve()
    runs = root / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-storage')
    runs.mkdir(parents=True, exist_ok=False)
    prep = runs / 'prepare'
    script = root / 'storage.sbatch'
    jobs = []
    common = ['sbatch', '--parsable', '--partition', a.partition, '--cpus-per-task=8', '--nodes=1', '--ntasks=1']
    jobs.append({'name': 'prepare', 'command': common + ['--mem=8192M', '--time=04:00:00',
                 '--output', str(runs/'prepare-%j.log'), str(script), str(root), 'prepare', '--output', str(prep)]})
    estimates = {}
    for dataset in ('sift', 'gist'):
        # Never execute an AVX-512 binary on an unknown login-node CPU.
        # Preparation rechecks these compiled-layout estimates on the job node.
        estimate = json.loads((root/'estimates.json').read_text())[dataset]
        estimates[dataset] = estimate
        for divisor, limit in limits(estimate['estimated_bytes']).items():
            name = f'{dataset}-one-over-{divisor}'
            jobs.append({'name': name, 'limit_bytes': limit, 'command': common + [f'--mem={limit//MIB}M',
                         '--time=08:00:00', '--output', str(runs/(name+'-%j.log')), str(script), str(root), 'run',
                         '--dataset', dataset, '--prepare', str(prep), '--output', str(runs/name),
                         '--limit-bytes', str(limit), '--seconds', str(a.seconds)]})
    plan = {'runs': str(runs), 'estimates': estimates, 'jobs': jobs}
    save(runs/'plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    if a.submit:
        previous = None
        prepare_id = None
        for job in jobs:
            cmd = list(job['command'])
            if previous:
                # An unsuccessful calibration must prevent every measurement.
                # Subsequent failed trials need not block independent ratios.
                dependency = f'afterok:{prepare_id}'
                if previous != prepare_id:
                    dependency += f',afterany:{previous}'
                cmd.insert(1, '--dependency=' + dependency)
            job_id = subprocess.check_output(cmd, text=True).strip().split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError('Unexpected sbatch response')
            job['job_id'] = job_id
            previous = job_id
            if prepare_id is None:
                prepare_id = job_id
            save(runs/'plan.json', plan)
            print(job['name'], job_id, flush=True)

if __name__ == '__main__':
    main()
