# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.


import json
from pathlib import Path
import tempfile
import unittest
from cgroup import inspect, MIB, mount_path
from experiment import limits, summaries

class Tests(unittest.TestCase):
    def test_loading_plan_isolates_every_arm_and_never_searches(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('loading_plan', Path(__file__).with_name('submit-load-validation.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root = Path('/deployment')
        estimates = {name: {'estimated_bytes':1024*MIB} for name in ('sift','gist')}
        jobs = module.build_jobs(root, root/'runs/test', root/'prepare', estimates, 'MEM_2TB')
        self.assertEqual(len(jobs),12)
        self.assertEqual(len({j['name'] for j in jobs}),12)
        for job in jobs:
            cmd = job['command']
            self.assertIn('--load-only', cmd)
            self.assertEqual(cmd[cmd.index('--rounds')+1], '1')
            self.assertEqual(cmd[cmd.index('--concurrencies')+1], '1')
            self.assertIn('--mem='+str(job['limit_bytes']//MIB)+'M', cmd)

    def test_limits_round_up_exact_fractions(self):
        self.assertEqual(limits(1024*MIB), {2:512*MIB,4:256*MIB})
        self.assertEqual(limits(1024*MIB+1), {2:513*MIB,4:257*MIB})
        with self.assertRaises(ValueError): limits(0)

    def test_subtree_mount_resolution(self):
        self.assertEqual(mount_path('/slurm/job/step','/slurm','/sys/fs/cgroup'),Path('/sys/fs/cgroup/job/step'))
        with self.assertRaises(RuntimeError): mount_path('/other','/slurm','/cgroup')

    def test_v2_ancestor_enforcement_and_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); child=root/'step';child.mkdir()
            (child/'memory.max').write_text('max')
            (root/'memory.max').write_text(str(512*MIB))
            (root/'memory.swap.max').write_text('0')
            result=inspect('v2',child,root,512*MIB,True)
            self.assertEqual(result['effective_limit_bytes'],512*MIB)
            with self.assertRaises(RuntimeError):inspect('v2',child,root,256*MIB,True)
            (root/'memory.swap.max').write_text('max')
            with self.assertRaises(RuntimeError):inspect('v2',child,root,512*MIB,True)
            self.assertTrue(inspect('v2',child,root,512*MIB,False)['swap_disabled'])
            (root/'memory.max').write_text('max')
            with self.assertRaises(RuntimeError):inspect('v2',child,root,512*MIB,False)

    def test_v1_unlimited_and_memsw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'memory.limit_in_bytes').write_text(str(2**63-4096))
            with self.assertRaises(RuntimeError):inspect('v1',root,root,512*MIB,False)
            (root/'memory.limit_in_bytes').write_text(str(512*MIB))
            (root/'memory.memsw.limit_in_bytes').write_text(str(512*MIB))
            with self.assertRaises(RuntimeError):inspect('v1',root,root,512*MIB,True)
            (root/'memory.memsw.limit_in_bytes').write_text(str(1024*MIB))
            with self.assertRaises(RuntimeError):inspect('v1',root,root,512*MIB,True)

    def test_summary_excludes_telemetry_but_keeps_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            row = dict(concurrency=1, L=128, arm='Mmap', validation_sha256='same',
                       recall=0.96, qps=1234, p50_us=10, p95_us=20, p99_us=30, undersampled=False)
            (path/'r1-c1-mmap.jsonl').write_text(json.dumps(row)+'\n')
            (path/'r1-c1-mmap.telemetry.jsonl').write_text(json.dumps({'time':1, 'process':{}, 'cgroup':{}})+'\n')
            summaries(path)
            summary = json.loads((path/'summary.json').read_text())
            self.assertEqual(len(summary['rows']), 1)
            self.assertEqual(summary['rows'][0]['qps'], 1234)
            self.assertTrue(summary['ordered_ids_match'])

    def test_telemetry_only_is_not_a_successful_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'r1-c1-mmap.telemetry.jsonl').write_text('{"time":1}\n')
            with self.assertRaises(RuntimeError): summaries(path)
            self.assertFalse((path/'summary.json').exists())

    def test_malformed_measurement_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'r1-c1-mmap.jsonl').write_text('{"unexpected":true}\n')
            with self.assertRaises(KeyError): summaries(path)
            self.assertFalse((path/'summary.json').exists())

    def test_parity_failure_is_not_summarized_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            rows=[{'concurrency':8,'L':40,'arm':'Mmap','validation_sha256':'a'},
                  {'concurrency':8,'L':40,'arm':'Ssd','validation_sha256':'b'}]
            (path/'r1-test.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(RuntimeError): summaries(path)
            self.assertFalse((path/'summary.json').exists())

if __name__=='__main__':unittest.main()
