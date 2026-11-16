# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Small tests; no real data, baseline packages or native builds required."""
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import numpy as np
from .adapters import Native, Unsupported
from .data import normalized, recall, rows, write_json
from .protocol import Protocol, distribution, evaluate, measure
from .report import report, selected


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = Protocol(concurrency=(1, 8), rounds=3, min_requests=80,
                                 min_seconds=0, max_seconds=2, warmup_requests=0,
                                 warmup_seconds=0, cooldown_seconds=0)
        self.queries = list(range(10))

    def test_quantiles_merge_samples_not_percentiles(self):
        result = distribution([1000] + [10000]*99)
        self.assertEqual(result['p50_us'], 10)
        self.assertEqual(result['p99_us'], 10)
        self.assertEqual(result['mean_us'], 9.91)
        self.assertEqual(distribution([0, 1, 2])['p50_us'], .001)

    def test_complete_query_cycle_and_concurrency(self):
        seen, lock = [], threading.Lock()
        def search(q):
            with lock:
                seen.append(q)
            return [q]
        trial, samples = measure(search, self.queries, list(reversed(self.queries)), 8, self.protocol)
        self.assertTrue(trial['minimum_reached'])
        self.assertEqual(len(samples), 80)
        self.assertEqual(sorted(seen), sorted(self.queries*8))
        self.assertGreater(trial['instrumented_qps'], 0)

    def test_warmup_exception_does_not_deadlock(self):
        def fail(_):
            raise ValueError('warmup failed')
        with self.assertRaisesRegex(ValueError, 'warmup failed'):
            measure(fail, self.queries, self.queries, 8, replace(self.protocol, warmup_requests=8))

    def test_search_exception_is_not_omitted(self):
        def fail(_):
            raise ValueError('search failed')
        with self.assertRaisesRegex(ValueError, 'search failed'):
            measure(fail, self.queries, self.queries, 8, self.protocol)

    def test_initializer_failure_does_not_deadlock(self):
        def fail():
            raise RuntimeError('init failed')
        with self.assertRaisesRegex(RuntimeError, 'init failed'):
            measure(lambda q: [q], self.queries, self.queries, 8, self.protocol, fail)

    def test_incomplete_is_explicit(self):
        trial, _ = measure(lambda q: [q], self.queries, self.queries, 1,
                           replace(self.protocol, min_requests=10**9, max_seconds=.01))
        self.assertFalse(trial['minimum_reached'])

    def test_invalid_protocol_and_order(self):
        for config in (replace(self.protocol, concurrency=(0,)),
                       replace(self.protocol, concurrency=(1, 1)),
                       replace(self.protocol, min_seconds=float('nan')),
                       replace(self.protocol, rounds=0)):
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            measure(lambda q: [q], self.queries, [0]*10, 1, self.protocol)

    def test_evaluation_keeps_order(self):
        self.assertEqual(evaluate(lambda q: [q], self.queries, 8), [(q,) for q in self.queries])

    def test_recall_is_strict_and_partial_results_count(self):
        gt = np.array([[0, 1], [2, 3]])
        result = recall([[0], [3, 2]], gt, 2, 4)
        self.assertEqual(result['recall'], .75)
        self.assertEqual(result['partial_queries'], 1)
        for bad in ([[0, 0], [2, 3]], [[-1], [2, 3]], [[4], [2, 3]]):
            with self.assertRaises(ValueError):
                recall(bad, gt, 2, 4)
        with self.assertRaises(ValueError):
            recall([[0], [1]], np.array([[0, 0], [1, 2]]), 2, 4)

    def test_recall_target_selection(self):
        points = [dict(complete=True, evaluation={'recall': r}, pooled={'p50_us': v})
                  for r, v in ((.94, 1), (.95, 10), (.98, 20))]
        self.assertIs(selected(points, .95), points[1])
        self.assertIsNone(selected(points, .99))
        points[1]['complete'] = False
        self.assertIs(selected(points, .95), points[2])

    def test_input_formats_and_malformed_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data = np.array([[1, 2], [3, 4]], dtype='<f4')
            (p/'base.fbin').write_bytes(struct.pack('<II', 2, 2)+data.tobytes())
            np.testing.assert_array_equal(rows(p/'base.fbin'), data)
            (p/'base.fvecs').write_bytes(b''.join(struct.pack('<I', 2)+r.tobytes() for r in data))
            np.testing.assert_array_equal(rows(p/'base.fvecs'), data)
            (p/'base.u8bin').write_bytes(struct.pack('<II', 2, 2)+data.astype('u1').tobytes())
            np.testing.assert_array_equal(rows(p/'base.u8bin'), data)
            truth = data.astype('<u4')
            (p/'gt.bin').write_bytes(struct.pack('<II', 2, 2)+truth.tobytes()+data.tobytes())
            np.testing.assert_array_equal(rows(p/'gt.bin', groundtruth=True), truth)
            (p/'base.fbin').write_bytes(struct.pack('<II', 2, 2)+b'bad')
            with self.assertRaises(ValueError):
                rows(p/'base.fbin')

    def test_cosine_copy_and_zero_rejection(self):
        source = np.array([[3, 4]], dtype='f4')
        np.testing.assert_allclose(normalized(source), [[.6, .8]])
        np.testing.assert_array_equal(source, [[3, 4]])
        with self.assertRaises(ValueError):
            normalized(np.zeros((1, 3)))

    def test_invalid_metric_conversion_rejected_before_loading_native_library(self):
        with self.assertRaises(Unsupported):
            Native('diskann', {'unit_norm_ip': True}, {'metric': 'inner-product'},
                   np.array([[3, 4]], dtype='f4'), 8)
        with self.assertRaises(Unsupported):
            Native('parlayann', {'normalize': True}, {'metric': 'squared-l2'},
                   np.array([[3, 4]], dtype='f4'), 8)

    def test_failed_jobs_never_rank_and_missing_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'fixture--engine'
            dest.mkdir()
            write_json(dest/'result.json', dict(dataset='fixture', engine='engine', status='missing',
                                               protocol={'concurrency': [1, 8]}, error='no index', points=[]))
            report(tmp, [.95])
            text = (Path(tmp)/'comparison.md').read_text()
            self.assertIn('missing', text)
            self.assertNotIn('nan', text)
            self.assertFalse((dest/'result.json.partial').exists())


if __name__ == '__main__':
    unittest.main()
