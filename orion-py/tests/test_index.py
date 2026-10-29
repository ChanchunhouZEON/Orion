import concurrent.futures
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np
from orion_py import Index, GraphConfig, CascadeConfig, OrionError


def write_vectors(path, values):
    with path.open("wb") as file:
        if path.suffix in (".fbin", ".u8bin", ".custom"):
            file.write(struct.pack("<II", *values.shape))
            file.write(values.tobytes())
        else:
            for row in values:
                file.write(struct.pack("<I", len(row)))
                file.write(row.tobytes())


def write_staged(path, count, degree=8, extra=4):
    with path.open("wb") as file:
        file.write(struct.pack("<6I", 0x53544147, 3, count, degree, extra, 0))
        for node in range(count):
            # A connected deterministic graph, with all three STAG partitions.
            ids = [(node + offset) % count for offset in range(1, degree + extra + 1)]
            file.write(struct.pack("<3I", degree // 2, degree - degree // 2, extra))
            file.write(struct.pack(f"<{len(ids)}I", *ids))


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="orion-fixture-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(9284)
        self.staged = self.root / "graph.staged"
        write_staged(self.staged, 64)
        self.graph = GraphConfig(source="parlayann", staged_file=self.staged,
                                 degree=8, build_l=32, alpha=1.2, max_extra=4)

    def arrays(self, dtype=np.float32, dimension=32):
        base = self.rng.integers(0, 240, (64, dimension)).astype(dtype)
        queries = self.rng.integers(0, 240, (24, dimension)).astype(dtype)
        return base, queries

    def make(self, base, **kw):
        return Index.from_array(base, graph=self.graph, cache_dir=self.root / "array-cache", threads=2, **kw)

    def assert_results(self, left, right):
        np.testing.assert_array_equal(left.ids, right.ids)
        np.testing.assert_array_equal(left.counts, right.counts)
        self.assertEqual(left.ids.dtype, np.uint32)
        self.assertEqual(left.counts.dtype, np.uint32)

    def cli(self, base, query, cache, metric, storage, k):
        args = [os.environ["ORION_TEST_CLI"], "anonymous", "--base", str(base), "--query", str(query),
                "--no-groundtruth", "--dimension", "128" if storage == "u8" else "32", "--metric", metric,
                "--vector-storage", storage, "--graph-source", "parlayann", "--staged-file", str(self.staged),
                "--graph-degree", "8", "--build-l", "32", "--alpha", "1.2", "--max-extra", "4",
                "--threads", "2", "--window-size", "5", "--k", str(k), "--search-l", "24", "--cache-dir", str(cache)]
        if storage == "u8":
            args += ["--prefilter", "none", "--admission", "native-l2-u8", "--rerank", "none"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("ORION_") and k != "PA_ROOT"}
        result = subprocess.run(args, text=True, capture_output=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        return [v["ids"] for v in records if v.get("type") == "result"], next(v for v in records if v.get("type") == "summary")

    def test_file_array_cli_parity_and_cached_reload(self):
        for metric, dtype, dim, suffix in [("l2", np.float32, 32, ".fbin"),
                ("cosine", np.float32, 32, ".fvecs"), ("inner-product", np.float32, 32, ".fbin"),
                ("l2", np.uint8, 128, ".u8bin"), ("l2", np.uint8, 128, ".bvecs")]:
            with self.subTest(metric=metric, format=suffix):
                base, queries = self.arrays(dtype, dim)
                base_file, query_file = self.root / ("base" + suffix), self.root / ("query" + suffix)
                write_vectors(base_file, base); write_vectors(query_file, queries)
                folder = self.root / (metric + suffix)
                file_index = Index.from_file(base_file, metric=metric, graph=self.graph, cache_dir=folder, threads=2)
                array_index = self.make(base, metric=metric)
                for k in (3, 10):
                    fc = file_index.calibrate(query_file, k=k)
                    ac = array_index.calibrate(queries, k=k)
                    self.assertEqual(fc, ac)
                    a = array_index.search(queries, k=k, search_list_size=24)
                    f = file_index.search(query_file, k=k, search_list_size=24)
                    self.assert_results(a, f)
                    cli_ids, summary = self.cli(base_file, query_file, folder, metric, "u8" if dtype == np.uint8 else "f32", k)
                    self.assertEqual([row[:n].tolist() for row, n in zip(f.ids, f.counts)], cli_ids)
                    self.assertAlmostEqual(fc.threshold, summary["calibration"]["threshold"], places=7)
                    self.assertEqual(fc.early_exit_limit, summary["calibration"]["early_exit_limit"])
                snapshot = self.root / ("snapshot" + metric + suffix)
                file_index.save(snapshot)
                loaded = Index.open(snapshot, base=base_file, threads=1)
                self.assert_results(f, loaded.search(queries, k=10, search_list_size=24))
                self.assertEqual(loaded.info["threads"], 1)

    def test_array_snapshot_identity_and_calibration(self):
        base, queries = self.arrays()
        index = self.make(base)
        with self.assertRaisesRegex(OrionError, "calibrate"):
            index.search(queries, k=3)
        index.calibrate(queries, k=3)
        expected = index.search(queries, k=3, search_list_size=32)
        path = self.root / "saved"
        index.save(path)
        restored = Index.open(path, base=base.copy())
        self.assert_results(expected, restored.search(queries, k=3, search_list_size=32))
        with self.assertRaisesRegex(OrionError, "calibrate"):
            restored.search(queries, k=4)
        changed = base.copy(); changed[0, 0] += 1
        with self.assertRaisesRegex(OrionError, "identity"):
            Index.open(path, base=changed)
        with self.assertRaisesRegex(OrionError, "already exists"):
            index.save(path)
        with self.assertRaises(OrionError):
            Index.open(self.root / "missing", base=base)
        # Export changes after loading must not be misrecorded as the source
        # of the resident graph when another snapshot is saved.
        original_manifest = json.loads(next(path.glob("*.cache.json")).read_text())
        self.staged.write_bytes(b"changed after import")
        saved_again = self.root / "saved-again"; index.save(saved_again)
        self.assertEqual(original_manifest, json.loads(next(saved_again.glob("*.cache.json")).read_text()))
        self.assert_results(expected, Index.open(saved_again, base=base).search(queries, k=3, search_list_size=32))
        # Restored calibration cannot be silently combined with a changed cascade.
        metadata = json.loads((path / "index.json").read_text())
        metadata["options"]["window_size"] = 6
        (path / "index.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(OrionError, "configuration"):
            Index.open(path, base=base)

    def test_bad_inputs_are_errors(self):
        base, queries = self.arrays()
        for invalid in [base.astype(np.float64), base.reshape(-1), [[1, 2]], base.astype(">f4")]:
            with self.assertRaises(TypeError): self.make(invalid)
        with self.assertRaises(ValueError): self.make(base[:, ::-1])
        with self.assertRaises(OrionError): self.make(base[:, :31].copy())
        with self.assertRaises(OrionError): self.make(base[:0])
        broken = base.copy(); broken[0, 0] = np.nan
        with self.assertRaisesRegex(OrionError, "non-finite"): self.make(broken)
        with self.assertRaisesRegex(OrionError, "budget"):
            self.make(base, memory_budget_gib=1e-10)
        with self.assertRaisesRegex(OrionError, "threads"):
            Index.from_array(base, threads=0, graph=self.graph, cache_dir=self.root / "bad")
        index = self.make(base)
        for k, beam in [(0, 32), (65, 65), (10, 9)]:
            with self.assertRaises(OrionError): index.search(queries, k=k, search_list_size=beam, adaptive=False)
        with self.assertRaises(ValueError): index.search(queries[:, :31], adaptive=False)
        with self.assertRaises(TypeError): index.search(queries.astype(np.float64), adaptive=False)
        with self.assertRaises(OrionError): index.calibrate(queries[:0])
        result = index.search(queries[:0], k=3, adaptive=False)
        self.assertEqual(result.ids.shape, (0, 3))
        self.assertEqual(result.counts.shape, (0,))

    def test_owned_base_and_concurrent_calls(self):
        base, queries = self.arrays()
        index = self.make(base)
        index.calibrate(queries, k=3)
        expected = index.search(queries, k=3, search_list_size=24)
        base[:] = 0  # Caller mutation cannot change the index's owned base.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(index.search, queries, k=3, search_list_size=24) for _ in range(6)]
            for future in futures: self.assert_results(expected, future.result(timeout=20))

    def test_native_u8_constraints_and_array_roundtrip(self):
        base, queries = self.arrays(np.uint8, 128)
        with self.assertRaises(OrionError):
            Index.from_array(base, graph=GraphConfig(source="rust"), cache_dir=self.root / "bad")
        with self.assertRaises(OrionError):
            Index.from_array(base, graph=GraphConfig(), cache_dir=self.root / "no-export")
        index = self.make(base)
        exact = index.search(base, k=1, search_list_size=64, adaptive=False)
        np.testing.assert_array_equal(exact.ids[:, 0], np.arange(64))
        with self.assertRaises(TypeError): index.search(queries.astype(np.float32), adaptive=False)
        with self.assertRaises(OrionError):
            self.make(base, cascade=CascadeConfig(rerank="f32"))
        index.calibrate(queries, k=10)
        snapshot = self.root / "native-saved"; index.save(snapshot)
        restored = Index.open(snapshot, base=base)
        self.assert_results(index.search(queries, k=10), restored.search(queries, k=10))
        self.assertEqual(restored.info["storage"], "u8")

    def test_strict_cache_and_file_errors(self):
        base, queries = self.arrays()
        path = self.root / "base.custom"; write_vectors(path, base)
        index = Index.from_file(path, base_format="fbin", graph=self.graph, cache_dir=self.root / "file-cache")
        index.calibrate(queries, k=3)
        saved = self.root / "file-saved"; index.save(saved)
        graph = next(saved.glob("*.pgraph"))
        graph.write_bytes(graph.read_bytes()[:-4])
        with self.assertRaises(OrionError): Index.open(saved, base=path)
        bad_query = self.root / "bad.fbin"; write_vectors(bad_query, queries)
        bad_query.write_bytes(bad_query.read_bytes()[:-1])
        with self.assertRaises(OrionError): index.search(bad_query, k=3)
        saved2 = self.root / "file-saved2"; index.save(saved2)
        with path.open("r+b") as file: file.seek(8); file.write(struct.pack("<f", 1234))
        with self.assertRaisesRegex(OrionError, "identity"): Index.open(saved2, base=path)
        with self.assertRaisesRegex(OrionError, "changed"): index.save(self.root / "stale-save")

    def test_auto_rust_build_and_reuse(self):
        base, queries = self.arrays()
        index = Index.from_array(base, graph=GraphConfig(degree=8, build_l=32, max_extra=4), cache_dir=self.root / "rust-cache", threads=2)
        self.assertEqual(index.info["graph_source"], "rust")
        index.calibrate(queries, k=3)
        expected = index.search(queries, k=3, search_list_size=24)
        reused = Index.from_array(base, graph=GraphConfig(degree=8, build_l=32, max_extra=4), cache_dir=self.root / "rust-cache", threads=2)
        reused.calibrate(queries, k=3)
        self.assert_results(expected, reused.search(queries, k=3, search_list_size=24))
        self.assertIn("cache available", reused.info["graph_selection_reason"])

    def test_high_dimension_and_top100(self):
        # Exercise another compiled specialization and the high-dimensional
        # default JL/L2-KT cascade, including restored lazy sidecars.
        base, queries = self.arrays(dimension=960)
        index = self.make(base)
        index.calibrate(queries, k=10)
        saved = self.root / "high-saved"; index.save(saved)
        restored = Index.open(saved, base=base)
        self.assert_results(index.search(queries, k=10), restored.search(queries, k=10))
        write_staged(self.staged, 128)
        base = self.rng.integers(0, 240, (128, 32)).astype(np.float32)
        queries = base[:5].copy()
        index = self.make(base)
        calibration = index.calibrate(queries, k=100)
        self.assertEqual(calibration.search_list_size, 200)
        result = index.search(queries, k=100, search_list_size=128)
        self.assertEqual(result.ids.shape, (5, 100))
        self.assertTrue(np.all(result.counts <= 100))
        for row, count in zip(result.ids, result.counts):
            self.assertTrue(np.all(row[:count] < 128))
            self.assertTrue(np.all(row[count:] == np.iinfo(np.uint32).max))

    def test_widened_byte_files_and_file_prefix(self):
        base, queries = self.arrays(np.uint8, 128)
        path = self.root / "bytes.bvecs"; write_vectors(path, base)
        index = Index.from_file(path, storage="f32", graph=self.graph, cache_dir=self.root / "widened")
        array = self.make(base.astype(np.float32))
        index.calibrate(queries, k=3); array.calibrate(queries, k=3)
        self.assert_results(index.search(queries, k=3), array.search(queries, k=3))
        # A file prefix needs a graph for that prefix, and an open must preserve it.
        write_staged(self.staged, 32)
        index = Index.from_file(path, graph=self.graph, max_points=32, cache_dir=self.root / "prefix")
        index.calibrate(queries, k=3)
        saved = self.root / "prefix-saved"; index.save(saved)
        restored = Index.open(saved, base=path)
        self.assertEqual(restored.info["num_points"], 32)
        self.assert_results(index.search(queries, k=3), restored.search(queries, k=3))
