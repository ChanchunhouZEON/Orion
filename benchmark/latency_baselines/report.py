# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Keep complete curves and select measured points at equal recall targets."""
import csv
import json
from pathlib import Path


def selected(points, target):
    eligible = [p for p in points if p['evaluation']['recall'] >= target
                and p.get('complete') and p.get('pooled')]
    return min(eligible, key=lambda p: p['pooled']['p50_us']) if eligible else None


def report(output, targets):
    output = Path(output)
    jobs = [json.loads(p.read_text()) for p in sorted(output.glob('*/result.json'))]
    lines = ['# Baseline request latency', '',
        'Python single-request API boundary, including binding/GIL, query preparation and result assembly. '
        'Closed-loop callers; not HTTP latency or the native Rust benchmark. '
        'Quantiles pool raw samples; QPS is median instrumented wall throughput. '
        'Loading/build/calibration/recall are excluded. No interpolation.', '',
        '| Dataset | Engine | Clients | Recall target | Effort | Recall | P50 µs | P95 µs | P99 µs | QPS |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    curve = []
    for job in jobs:
        for point in job.get('points', []):
            if not point.get('pooled'):
                continue
            curve.append(dict(dataset=job['dataset'], engine=job['engine'],
                concurrency=point['concurrency'], effort=point['effort'],
                recall=point['evaluation']['recall'], complete=point.get('complete', False),
                **point['pooled'], instrumented_qps=point['median_instrumented_qps']))
        for c in job['protocol']['concurrency']:
            points = [p for p in job.get('points', []) if p['concurrency'] == c]
            for target in targets:
                point = selected(points, target) if job['status'] == 'complete' else None
                if point:
                    p = point['pooled']
                    lines.append(f"| {job['dataset']} | {job['engine']} | {c} | {target:.3f} | "
                        f"{point['effort']} | {point['evaluation']['recall']:.6f} | "
                        f"{p['p50_us']:.2f} | {p['p95_us']:.2f} | {p['p99_us']:.2f} | "
                        f"{point['median_instrumented_qps']:,.0f} |")
                else:
                    reason = job['status'] if job['status'] != 'complete' else 'target not reached'
                    lines.append(f"| {job['dataset']} | {job['engine']} | {c} | {target:.3f} | — | {reason} | — | — | — | — |")
    lines += ['', 'Each target selects the lowest measured pooled P50 among configurations reaching that recall. '
              'P95/P99 and QPS come from that same point. See curves.csv for every measured configuration.', '',
              '## Run status', '']
    lines.extend(f"- {j['dataset']}/{j['engine']}: {j['status']} {j.get('error', '')}" for j in jobs)
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    if curve:
        with (output / 'curves.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
    (output / 'comparison.json').write_text(json.dumps(dict(targets=targets, jobs=jobs), indent=2) + '\n')
    return jobs
