# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Small shared-graph parity, scratch, GIL and ownership tests; no real datasets."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ctypes as ct
import os
import tempfile
import threading
import unittest
import numpy as np
from .native import NativeIndex, library


@unittest.skipUnless(
    os.getenv("USEARCH_CONTROL_LIBRARY"), "build private adapter first"
)
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = library(os.environ["USEARCH_CONTROL_LIBRARY"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def fixture(self, metric="l2sq", dimension=32, slots=8):
        from usearch.index import Index

        rng = np.random.default_rng(20261113)
        base = rng.normal(size=(512, dimension)).astype("f4")
        queries = rng.normal(size=(24, dimension)).astype("f4")
        py = Index(
            ndim=dimension,
            metric=metric,
            dtype="f32",
            connectivity=16,
            expansion_add=100,
        )
        py.add(np.arange(len(base), dtype="u8"), base, threads=1)
        path = Path(self.temp.name) / f"{metric}-{dimension}-{slots}.index"
        py.save(str(path))
        native = NativeIndex(
            self.lib,
            path,
            dimension=dimension,
            count=len(base),
            metric=py.metric_kind,
            slots=slots,
        )
        self.addCleanup(native.close)
        self.assertEqual(native.isa, py.specs["hardware_acceleration"])
        return py, native, queries

    def test_shared_graph_ids_and_distances(self):
        for metric in ["l2sq", "ip", "cos"]:
            for dimension in [25, 128]:
                py, native, queries = self.fixture(metric, dimension)
                for effort in [16, 64]:
                    py.expansion_search = effort
                    native.configure(effort)
                    for q in queries:
                        expected = py.search(q, count=10, threads=1)
                        ids, distances = native.search(q, 10)
                        np.testing.assert_array_equal(ids, expected.keys)
                        held_ids, held_distances = native.search(
                            q, 10, release_gil=False
                        )
                        np.testing.assert_array_equal(held_ids, ids)
                        np.testing.assert_array_equal(held_distances, distances)
                        np.testing.assert_allclose(
                            distances, expected.distances, rtol=2e-6, atol=2e-6
                        )

    def test_eight_callers_two_exclusive_scratch_slots(self):
        py, native, queries = self.fixture(slots=2)
        py.expansion_search = 64
        native.configure(64)
        expected = [py.search(q, count=10, threads=1) for q in queries]
        barrier = threading.Barrier(8)

        def worker(i):
            barrier.wait(timeout=5)
            for n in range(60):
                j = (i + n) % len(queries)
                ids, distances = native.search(queries[j], 10)
                np.testing.assert_array_equal(ids, expected[j].keys)
                np.testing.assert_allclose(
                    distances, expected[j].distances, rtol=2e-6, atol=2e-6
                )

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(worker, range(8)))

    def test_invalid_inputs_and_recovery(self):
        _, native, queries = self.fixture()
        native.configure(16)
        for q, k in [
            (queries[0, :3], 10),
            (queries[0].astype("f8"), 10),
            (queries[0], 0),
            (queries[0], 513),
        ]:
            with self.assertRaises(ValueError):
                native.search(q, k)
        native.configure(1)
        with self.assertRaisesRegex(ValueError, "expansion"):
            native.search(queries[0], 10)
        native.configure(32)
        self.assertEqual(len(native.search(queries[0], 10)[0]), 10)
        native.close()
        native.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            native.search(queries[0], 10)

    def test_open_rejects_missing_file_and_wrong_shape(self):
        from usearch.index import MetricKind

        with self.assertRaises(ValueError):
            NativeIndex(
                self.lib,
                Path(self.temp.name) / "missing",
                dimension=32,
                count=512,
                metric=MetricKind.L2sq,
            )
        self.fixture()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            NativeIndex(
                self.lib,
                Path(self.temp.name) / "l2sq-32-8.index",
                dimension=33,
                count=512,
                metric=MetricKind.L2sq,
            )

    def test_close_waits_for_active_borrow(self):
        _, native, _ = self.fixture()
        begun, done = threading.Event(), threading.Event()

        def close():
            begun.set()
            native.close()
            done.set()

        with native._borrow():
            thread = threading.Thread(target=close, daemon=True)
            thread.start()
            self.assertTrue(begun.wait(1))
            self.assertFalse(done.wait(0.02))
        thread.join(timeout=2)
        self.assertTrue(done.is_set())

    def test_cdll_releases_gil_during_native_wait(self):
        probe = self.lib.uc_probe_new()
        self.assertTrue(probe)
        try:
            # Native wait has a 2s watchdog. With PyDLL (GIL held), it times out
            # before this thread can call release, so the result would be -1.
            with ThreadPoolExecutor(1) as pool:
                waiting = pool.submit(self.lib.uc_probe_wait, probe)
                self.assertEqual(self.lib.uc_probe_release(probe), 0)
                self.assertEqual(waiting.result(timeout=3), 0)
        finally:
            self.lib.uc_probe_close(probe)


if __name__ == "__main__":
    unittest.main()
