# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Error contracts are shared with Rust and CLI; no Rust panics escape as BaseException."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np
from orion_ann import Index, OrionError
import test_index as fixtures
from test_index import write_vectors


class ErrorTests(unittest.TestCase):
    setUp = fixtures.IndexTests.setUp
    arrays = fixtures.IndexTests.arrays
    make = fixtures.IndexTests.make

    def test_python_and_cli_missing_input_reports(self):
        missing = self.root / 'missing.fbin'
        with self.assertRaises(OrionError) as raised:
            Index.from_file(missing)
        error = raised.exception
        self.assertEqual(error.code, 'ORION_IO')
        self.assertEqual(error.details['path'], str(missing))
        self.assertEqual(error.details['kind'], 'NotFound')
        env = {k: v for k, v in os.environ.items() if not k.startswith('ORION_')}
        output = subprocess.run([os.environ['ORION_TEST_CLI'], '--metric', 'l2', '--prepare-only',
            '--no-groundtruth', '--base', str(missing), '--error-format', 'json'],
            capture_output=True, text=True, env=env)
        self.assertEqual(output.returncode, 2, output.stderr)
        self.assertEqual(json.loads(output.stderr.splitlines()[-1]), error.report)


    def test_configuration_errors_match_cli(self):
        base, _ = self.arrays()
        path = self.root / 'config-base.fbin'
        write_vectors(path, base)
        for kwargs, flags in [
            ({'threads': 0}, ['--threads', '0']),
            ({'threads': 1000000}, ['--threads', '1000000']),
            ({'window_size': 65}, ['--window-size', '65']),
            ({'memory_budget_gib': 0.0}, ['--memory-budget-gib', '0']),
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(OrionError) as raised:
                    Index.from_file(path, **kwargs)
                output = subprocess.run([os.environ['ORION_TEST_CLI'], '--base', str(path),
                    '--prepare-only', '--error-format', 'json'] + flags,
                    capture_output=True, text=True)
                self.assertEqual(output.returncode, 2, output.stderr)
                self.assertEqual(json.loads(output.stderr.splitlines()[-1]), raised.exception.report)

    def test_array_and_file_query_errors_match_and_index_stays_usable(self):
        base, queries = self.arrays()
        index = self.make(base)
        bad = np.ascontiguousarray(queries[:, :31])
        path = self.root / 'wrong-dimension.fbin'
        write_vectors(path, bad)
        reports = []
        for value in [bad, path]:
            with self.assertRaises(OrionError) as raised:
                index.search(value, k=3)
            reports.append(raised.exception.report)
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0]['code'], 'ORION_DIMENSION_MISMATCH')
        self.assertEqual(reports[0]['details'], {'field': 'queries.dimension', 'expected': 32, 'actual': 31})
        index.calibrate(queries, k=3)
        self.assertEqual(index.search(queries, k=3).ids.shape, (24, 3))


    def test_errors_have_actionable_categories(self):
        base, queries = self.arrays()
        with self.assertRaises(OrionError) as raised:
            self.make(base, memory_budget_gib=1e-12)
        self.assertEqual(raised.exception.code, 'ORION_RESOURCE_EXHAUSTED')
        self.assertGreater(raised.exception.details['required_bytes'], 0)
        self.assertTrue(raised.exception.details['components'])
        index = self.make(base)
        with self.assertRaises(OrionError) as raised:
            index.search(queries, k=3)
        self.assertEqual(raised.exception.code, 'ORION_INVALID_STATE')
        self.assertEqual(raised.exception.details['k'], 3)
        with self.assertRaises(TypeError):
            index.search(queries, k=3, adaptive=False)
        with self.assertRaises(OrionError) as raised:
            index.search(queries, k=0)
        self.assertEqual(raised.exception.code, 'ORION_INVALID_ARGUMENT')
        self.assertEqual(raised.exception.details['k'], 0)
        saved = self.root / 'snapshot'
        reference = index.save(saved)
        metadata = json.loads((saved / 'index.json').read_text())
        metadata['payloads']['base_sha256'] = '0' * 64
        (saved / 'index.json').write_text(json.dumps(metadata))
        with self.assertRaises(OrionError) as raised:
            Index.open(reference, base=base)
        self.assertEqual(raised.exception.code, 'ORION_CACHE_INCOMPATIBLE')


    def test_unexpected_panic_is_catchable_and_poisoned_handle_is_rejected(self):
        base, _ = self.arrays()
        path = self.root / 'base.fbin'
        write_vectors(path, base)
        program = '''
    import sys
    from orion_ann import Index, GraphConfig, OrionError
    index = Index.from_file(sys.argv[1], graph=GraphConfig(source='parlayann', staged_file=sys.argv[2], degree=8, build_l=32, max_extra=4), cache_dir=sys.argv[3], threads=2)
    try:
        index._handle._test_panic()
    except Exception as error:
        assert isinstance(error, OrionError), type(error)
        assert error.code == 'ORION_INTERNAL'
        assert error.details['requires_reopen']
    else:
        raise AssertionError('panic did not raise')
    try:
        index.info
    except OrionError as error:
        assert error.code == 'ORION_INVALID_STATE'
    else:
        raise AssertionError('poisoned index reused')
    print('interpreter survived')
    '''
        output = subprocess.run([sys.executable, '-c', textwrap.dedent(program), str(path), str(self.staged), str(self.root / 'panic-cache')], capture_output=True, text=True, timeout=60)
        self.assertEqual(output.returncode, 0, output.stderr)
        self.assertIn('interpreter survived', output.stdout)

    def test_read_panic_also_invalidates_the_handle(self):
        base, queries = self.arrays()
        index = self.make(base)
        index.calibrate(queries, k=3)
        with self.assertRaises(OrionError) as raised:
            index._handle._test_read_panic()
        self.assertEqual(raised.exception.code, "ORION_INTERNAL")
        self.assertTrue(raised.exception.details["requires_reopen"])
        for operation in [lambda: index.info, lambda: index.search(queries, k=3),
                          lambda: index.calibrate(queries, k=3),
                          lambda: index.save(self.root / "invalid")]:
            with self.assertRaises(OrionError) as rejected:
                operation()
            self.assertEqual(rejected.exception.code, "ORION_INVALID_STATE")
