"""Small fixtures and fake executables exercise wrappers without dataset runs."""
import ast
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / 'data'))
import numpy as np
import benchmark_support as support
from collect_sweep_medians import parse_orion, median_by_slot
from vector_io import write_fvecs, write_ivecs


class ScriptContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='orion scripts ')
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def executable(self, name, body):
        path = self.directory / name
        path.write_text('#!' + sys.executable + '\n' + body)
        path.chmod(0o755)
        return path

    def test_syntax_all_scripts(self):
        for path in SCRIPTS.glob('*.sh'):
            subprocess.run(['bash', '-n', str(path)], check=True)
        for folder in ('benchmark/scripts', 'data', 'visualizations'):
            for path in (ROOT / folder).rglob('*.py'):
                ast.parse(path.read_text(), filename=str(path))

    def test_readers_roundtrip_and_prefix_io(self):
        base = np.arange(35, dtype=np.float32).reshape(7, 5)
        file = self.directory / 'vectors.fvecs'
        write_fvecs(file, base)
        with patch.object(support.np, 'fromfile', wraps=np.fromfile) as reader:
            vectors, count, dim = support.read_fvecs(file, 2)
            self.assertEqual(reader.call_args.kwargs['count'], 12)
        np.testing.assert_array_equal(vectors, base[:2])
        self.assertEqual((count, dim), (2, 5))
        ids = self.directory / 'truth.ivecs'
        write_ivecs(ids, [[0, 1], [2, 3]])
        np.testing.assert_array_equal(support.read_ivecs(ids), [[0, 1], [2, 3]])

    def test_malformed_vecs_fail(self):
        file = self.directory / 'bad.fvecs'
        for payload in (b'', struct.pack('<i', 0), struct.pack('<if', 2, 1),
                        struct.pack('<ifif', 1, 2, 2, 3)):
            file.write_bytes(payload)
            with self.assertRaises(ValueError):
                support.read_fvecs(file)

    def test_exact_metrics_across_chunk_boundaries(self):
        base = np.array([[1, 0], [10, 3], [0, 2], [-1, 1]], dtype=np.float32)
        queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
        for metric in ('l2', 'ip', 'cos'):
            if metric == 'l2':
                expected = ((queries[:, None] - base) ** 2).sum(2)
            elif metric == 'ip':
                expected = -(queries @ base.T)
            else:
                expected = 1 - (queries @ base.T) / np.linalg.norm(base, axis=1)
            ids, distances = support.exact_top_k(base, queries, 2, metric, chunk=1, query_batch=1)
            np.testing.assert_allclose(distances, np.sort(expected, axis=1)[:, :2], atol=1e-6)
            np.testing.assert_allclose(np.take_along_axis(expected, ids, axis=1), distances, atol=1e-6)
        with self.assertRaises(ValueError):
            support.exact_top_k(base, queries, 5)

    def test_recall_rejects_mismatch_and_deduplicates(self):
        self.assertEqual(support.recall_at_k([[1, 1]], [[1, 2]], 2), .5)
        with self.assertRaises(ValueError):
            support.recall_at_k([[1]], [[1], [2]], 1)

    def test_config_resolution_handles_spaces_without_shell_evaluation(self):
        config = dict(base='data/a b.fvecs', query='data/q.fvecs', groundtruth='data/g.ivecs', metric='inner-product')
        binary = self.executable('config binary', 'print(' + repr(json.dumps(config)) + ')\n')
        env = dict(os.environ, ORION_CONFIG_BIN=str(binary), PY=sys.executable,
                   CARGO_TARGET_DIR=str(self.directory / 'cargo target'))
        with patch.dict(os.environ, env):
            self.assertEqual(support.dataset_paths('sift')['metric'], 'ip')
        result = subprocess.run(['bash', '-eu', '-c',
                                 'source "$1/_common.sh"; paths_for sift; printf "%s\\n" "$BASE"; binary_path orion',
                                 'test', str(SCRIPTS)], cwd=self.directory, env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), ['data/a b.fvecs', str(self.directory / 'cargo target/release/orion')])

    def test_large_dataset_preflight_uses_run_parameters(self):
        log = self.directory / 'calls.jsonl'
        binary = self.executable('sweep binary', 'import json, os, sys\nwith open(os.environ["CALL_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint("{}")\n')
        env = dict(os.environ, BIN=str(binary), RUN_DIR=str(self.directory / 'run'),
                   CALL_LOG=str(log), THREADS='7', TRIALS='2')
        subprocess.run(['bash', str(SCRIPTS / 'run_large_dataset.sh'), 'sift1b'],
                       cwd=self.directory, env=env, check=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(calls), 4)
        self.assertIn('--preflight', calls[0])
        self.assertEqual(calls[0][calls[0].index('--threads') + 1], '7')
        self.assertEqual(calls[0][calls[0].index('--k') + 1], '100')
        self.assertIn('--prepare-only', calls[1])

    def test_failed_preflight_stops_before_build(self):
        binary = self.executable('bad sweep', 'import sys\nsys.exit(9)\n')
        env = dict(os.environ, BIN=str(binary), RUN_DIR=str(self.directory / 'run'))
        result = subprocess.run(['bash', str(SCRIPTS / 'run_large_dataset.sh'), 'sift'], env=env)
        self.assertEqual(result.returncode, 9)
        self.assertFalse((self.directory / 'run/prepare.log').exists())

    def test_profile_child_failure_does_not_wait_forever(self):
        self.executable('xctrace', 'import sys\nsys.exit(0)\n')
        binary = self.executable('bad sweep', 'import sys\nsys.exit(7)\n')
        env = dict(os.environ, BIN=str(binary), RUN_DIR=str(self.directory / 'profile'),
                   PATH=str(self.directory) + os.pathsep + os.environ['PATH'])
        result = subprocess.run(['bash', str(SCRIPTS / 'run_search_profile.sh'), 'sift'],
                                env=env, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 7)

    def test_log_parser_accepts_prefixed_logs_and_rejects_partial_runs(self):
        log = self.directory / 'sweep.log'
        log.write_text('[2026-09-29 INFO orion]  L=  16  R@10=0.8  QPS=100\n  L=32 R@10=0.9 QPS=200\n')
        self.assertEqual(parse_orion(log), [(0.8, 100), (0.9, 200)])
        with self.assertRaises(ValueError):
            median_by_slot([[(0.8, 100)], []])

    def test_vectors_only_conversion_then_metric_groundtruth(self):
        base = self.directory / 'base.fvecs'
        query = self.directory / 'query.fvecs'
        out = self.directory / 'converted'
        write_fvecs(base, [[1, 0], [10, 2], [0, 1]])
        write_fvecs(query, [[1, 0]])
        subprocess.run([sys.executable, str(SCRIPTS / 'parlayann_convert.py'),
                        '--base', str(base), '--query', str(query), '--skip-groundtruth',
                        '--max-base-points', '2', '--out-dir', str(out)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(SCRIPTS / 'compute_gt_brute.py'),
                        '--base-fbin', str(out / 'base.fbin'), '--query-fbin', str(out / 'query.fbin'),
                        '--metric', 'ip', '-k', '1', '--out', str(out / 'gt.bin')], check=True, capture_output=True)
        self.assertEqual(struct.unpack('<III', (out / 'gt.bin').read_bytes()[:12]), (1, 1, 1))

    def test_baseline_refresh_uses_maintained_wrapper(self):
        import baseline_comparison
        with patch.object(baseline_comparison.subprocess, 'run') as run:
            baseline_comparison.run_rust_sweep('msmarco_bert_1M', 3)
        command = run.call_args.args[0]
        self.assertTrue(command[1].endswith('/sweep_orion_vs_diskann_vs_parlayann.sh'))
        self.assertEqual(run.call_args.kwargs['env']['DATASET'], 'msmarco_bert_1M')
        self.assertEqual(run.call_args.kwargs['env']['NUM_RUNS'], '3')
        self.assertTrue(run.call_args.kwargs['check'])

    def test_three_engine_wrapper_uses_staged_override_and_custom_target(self):
        log = self.directory / 'calls.jsonl'
        # Fake build and engines let us inspect exact argv, including space paths.
        self.executable('cargo', 'import sys\nsys.exit(0)\n')
        record = 'import json, os, sys\nwith open(os.environ["CALL_LOG"], "a") as f: f.write(json.dumps(sys.argv)+"\\n")\n'
        target = self.directory / 'cargo target'
        release = target / 'release'
        release.mkdir(parents=True)
        for name in ('orion-sweep', 'diskann_sweep'):
            binary = self.executable(name, record + 'print("  L=16 R@10=0.9 QPS=100")\n')
            binary.rename(release / name)
        pa = self.directory / 'Parlay ANN'
        vamana = pa / 'algorithms/vamana'
        vamana.mkdir(parents=True)
        neighbor = self.executable('neighbors', record + 'print("For 10@10 recall = 0.9, QPS = 100")\n')
        neighbor.rename(vamana / 'neighbors')
        # wiki has no build quantization flag; exercises Bash 3.2 empty-array handling.
        data = pa / 'data/WikiAda1M'
        data.mkdir(parents=True)
        for name in ('base.fbin', 'query.fbin', 'gt.bin'):
            (data / name).write_bytes(struct.pack('<II', 42, 1))
        staged = self.directory / 'explicit export.staged'
        staged.touch()
        output = self.directory / 'result.json'
        env = dict(os.environ, PATH=str(self.directory) + os.pathsep + os.environ['PATH'],
                   PY=sys.executable, PA_ROOT=str(pa), CARGO_TARGET_DIR=str(target),
                   CALL_LOG=str(log), ORION_FILE=str(staged), DATASET='wiki_ada_1M',
                   OUT_JSON=str(output), COOLDOWN_S='0', NUM_RUNS='1')
        result = subprocess.run(['bash', str(SCRIPTS / 'sweep_orion_vs_diskann_vs_parlayann.sh')],
                                cwd=self.directory, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        orion = next(call for call in calls if call[0].endswith('/orion-sweep'))
        self.assertEqual(orion[orion.index('--staged-file') + 1], str(staged))
        self.assertTrue(any(call[0].endswith('/diskann_sweep') for call in calls))
        self.assertEqual(json.loads(output.read_text())['num_points'], 42)

    def test_baseline_help_includes_wiki_without_loading_optional_engines(self):
        for name in ('baseline_comparison.py', 'additional_baselines.py', 'dbms_baselines.py'):
            result = subprocess.run([sys.executable, str(SCRIPTS / name), '--help'],
                                    cwd=self.directory, capture_output=True, text=True, check=True)
            self.assertIn('wiki_ada_1M', result.stdout)


if __name__ == '__main__':
    unittest.main()
