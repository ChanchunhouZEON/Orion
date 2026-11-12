# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Cross-dtype input contracts, including CLI snapshot transport and wide u32."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np
from orion_ann import Index, GraphConfig, OrionError


def binary(path, values):
    with path.open('wb') as stream:
        stream.write(struct.pack('<II', *values.shape))
        stream.write(values.tobytes())


class QueryTypeTests(unittest.TestCase):
    def test_python_and_cli_preserve_query_dtype_for_each_base_and_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = ((np.arange(32 * 13) * 17) % 97 + 1).reshape(32, 13)
            queries = [
                (np.arange(26).reshape(2, 13).astype(np.float32) - 6.5, 'fbin'),
                (original[:2].astype(np.uint8), 'u8bin'),
                ((np.arange(26).reshape(2, 13) - 13).astype(np.int8), 'i8bin'),
                ((np.arange(26).reshape(2, 13) + 65500).astype(np.uint16), 'u16bin'),
                ((np.arange(26, dtype=np.uint32).reshape(2, 13) + 16_777_217), 'u32bin'),
            ]
            for dtype in [np.float32, np.uint8]:
                for metric in ['squared-l2', 'inner-product', 'cosine']:
                    folder = root / f'{dtype.__name__}-{metric}'
                    folder.mkdir()
                    base = original.astype(dtype)
                    base_file = folder / ('base.fbin' if dtype == np.float32 else 'base.u8bin')
                    binary(base_file, base)
                    index = Index.from_file(base_file, metric=metric, threads=2, auto_pin=False,
                                             graph=GraphConfig(degree=8, build_l=16, max_extra=4),
                                             cache_dir=folder / 'cache')
                    # Every dtype is accepted by calibration without narrowing.
                    for values, extension in queries:
                        before = values.copy()
                        index.calibrate_many(values, ks=[3, 5], search_list_size=24)
                        np.testing.assert_array_equal(values, before)
                    saved = index.save(folder / 'snapshot')
                    for values, extension in queries:
                        with self.subTest(base=dtype, metric=metric, query=extension):
                            path = folder / f'queries.{extension}'
                            binary(path, values)
                            expected = index.search(values, k=3, search_list_size=24, scores=True)
                            file_result = index.search(path, k=3, search_list_size=24, scores=True)
                            np.testing.assert_array_equal(expected.ids, file_result.ids)
                            np.testing.assert_array_equal(expected.scores, file_result.scores)
                            cli = subprocess.run([os.environ['ORION_TEST_CLI'], '--snapshot', str(folder / 'snapshot'),
                                '--base', str(base_file), '--query', str(path), '--no-groundtruth',
                                '--k', '3', '--search-l', '24', '--threads', '2', '--scores'],
                                text=True, capture_output=True)
                            self.assertEqual(cli.returncode, 0, cli.stderr)
                            rows = [json.loads(line) for line in cli.stdout.splitlines()]
                            rows = [row for row in rows if row['type'] == 'result']
                            np.testing.assert_array_equal([row['ids'] for row in rows], expected.ids)
                            np.testing.assert_allclose([row['scores'] for row in rows], expected.scores, rtol=1e-6, atol=1e-6)
                    with self.assertRaises(OrionError):
                        index.search(np.zeros((2,13), dtype=np.float64), k=3)
                    with self.assertRaises(OrionError):
                        index.search(np.ones((2,13), dtype='>u4'), k=3)
                    with self.assertRaises(OrionError):
                        index.search(np.ones((2,26), dtype=np.uint16)[:,::2], k=3)

    def test_fresh_cli_calibration_accepts_integer_query_streams(self):
        from test_index import write_staged
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staged = root / 'graph.staged'
            write_staged(staged, 64)
            graph = GraphConfig(source='parlayann', staged_file=staged,
                                degree=8, build_l=32, alpha=1.2, max_extra=4)
            for base_type in [np.float32, np.uint8]:
                base = ((np.arange(64 * 13) * 17) % 97 + 1).reshape(64, 13).astype(base_type)
                bp = root / ('base.fbin' if base_type == np.float32 else 'base.u8bin')
                binary(bp, base)
                for dtype, extension, offset in [(np.int8, 'i8bin', -20),
                                                 (np.uint16, 'u16bin', 1000),
                                                 (np.uint32, 'u32bin', 16_777_217)]:
                    for metric in ['squared-l2', 'inner-product', 'cosine']:
                        with self.subTest(base=base_type, query=dtype, metric=metric):
                            values = (np.arange(39).reshape(3, 13) + offset).astype(dtype)
                            qp = root / f'queries.{extension}'
                            binary(qp, values)
                            cache = root / f'cache-{base_type.__name__}-{metric}'
                            index = Index.from_file(bp, metric=metric, graph=graph,
                                                    cache_dir=cache, threads=2, auto_pin=False)
                            index.calibrate(values, k=3, search_list_size=48)
                            expected = index.search(values, k=3, search_list_size=24, scores=True)
                            run = subprocess.run([os.environ['ORION_TEST_CLI'], '--base', str(bp),
                                '--query', str(qp), '--metric', metric, '--staged-file', str(staged),
                                '--graph-source', 'parlayann', '--graph-degree', '8', '--build-l', '32',
                                '--alpha', '1.2', '--max-extra', '4', '--threads', '2',
                                '--cache-dir', str(cache), '--no-auto-pin', '--no-groundtruth',
                                '--k', '3', '--search-l', '24', '--scores', '--query-batch-size', '1'],
                                text=True, capture_output=True)
                            self.assertEqual(run.returncode, 0, run.stderr)
                            rows = [json.loads(line) for line in run.stdout.splitlines()]
                            rows = [row for row in rows if row['type'] == 'result']
                            np.testing.assert_array_equal([row['ids'] for row in rows], expected.ids)
                            np.testing.assert_allclose([row['scores'] for row in rows], expected.scores,
                                                       rtol=1e-6, atol=1e-6)

    def test_u32_low_bits_are_preserved_in_exact_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            base = np.array([[16_777_216, 0], [16_777_220, 0], [16_777_224, 0], [16_777_228, 0]], dtype=np.float32)
            index = Index.from_array(base, graph=GraphConfig(degree=3, build_l=4, max_extra=0),
                                     cache_dir=Path(temp) / 'cache', threads=1, auto_pin=False)
            query = np.array([[16_777_217, 0]], dtype=np.uint32)
            index.calibrate(query, k=4, search_list_size=4)
            result = index.search(query, k=4, search_list_size=4, scores=True)
            np.testing.assert_array_equal(result.ids, [[0,1,2,3]])
            np.testing.assert_array_equal(result.scores, [[1,9,49,121]])
