# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

import concurrent.futures
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np
from orion_ann import Index, SnapshotRef, GraphConfig, CascadeConfig, OrionError


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
        args = [os.environ["ORION_TEST_CLI"], "--base", str(base), "--query", str(query),
                "--no-groundtruth", "--metric", metric,
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

    def test_automatic_residency_policy_and_snapshot_cli_parity(self):
        base, queries = self.arrays()
        bp, qp = self.root / "auto-base.fbin", self.root / "auto-query.fbin"
        write_vectors(bp, base); write_vectors(qp, queries)
        index = Index.from_file(bp, graph=self.graph, cache_dir=self.root / "auto-cache", threads=2)
        index.calibrate(queries, k=3, search_list_size=24)
        residency = index.info["residency"]
        self.assertTrue(residency["automatic"]["enabled"])
        self.assertLessEqual(residency["pinned_bytes"], residency["automatic"]["budget_bytes"])
        snapshot = index.save(self.root / "auto-snapshot")
        expected = index.search(queries, k=3, search_list_size=24)
        off = Index.open(snapshot, base=bp, auto_pin=False)
        self.assertFalse(off.info["residency"]["automatic"]["enabled"])
        self.assertEqual(off.info["residency"]["pinned_bytes"], 0)
        self.assert_results(expected, off.search(queries, k=3, search_list_size=24))
        for disabled in (False, True):
            args = [os.environ["ORION_TEST_CLI"], "--snapshot", snapshot.path,
                    "--base", str(bp), "--query", str(qp), "--k", "3", "--search-l", "24"]
            if disabled:
                args.append("--no-auto-pin")
            run = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            rows = [json.loads(line) for line in run.stdout.splitlines()]
            report = rows[-1]["residency"]
            self.assertEqual(report["automatic"]["enabled"], not disabled)
            self.assertLessEqual(report["pinned_bytes"], report["automatic"]["budget_bytes"])
            self.assertEqual([r["ids"] for r in rows[:-1]],
                             [row[:n].tolist() for row, n in zip(expected.ids, expected.counts)])

        # An absent CLI override inherits the saved policy, just like Rust/Python.
        saved_off = off.save(self.root / "off-snapshot")
        inherited = subprocess.run([os.environ["ORION_TEST_CLI"], "--snapshot", saved_off.path,
            "--base", str(bp), "--query", str(qp), "--k", "3", "--search-l", "24"],
            capture_output=True, text=True)
        self.assertEqual(inherited.returncode, 0, inherited.stderr)
        report = json.loads(inherited.stdout.splitlines()[-1])["residency"]
        self.assertFalse(report["automatic"]["enabled"])
        self.assertEqual(report["pinned_bytes"], 0)

    def test_file_array_cli_parity_and_cached_reload(self):
        for metric, dtype, dim, suffix in [("l2", np.float32, 32, ".fbin"),
                ("cosine", np.float32, 32, ".fvecs"), ("inner-product", np.float32, 32, ".fbin"),
                ("l2", np.uint8, 128, ".u8bin"), ("l2", np.uint8, 128, ".bvecs"),
                ("l2", np.float32, 31, ".fbin"), ("cosine", np.float32, 33, ".fvecs"),
                ("inner-product", np.float32, 129, ".fbin"),
                ("l2", np.uint8, 3, ".u8bin"), ("l2", np.uint8, 129, ".bvecs")]:
            with self.subTest(metric=metric, format=suffix):
                base, queries = self.arrays(dtype, dim)
                base_file, query_file = self.root / ("base" + suffix), self.root / ("query" + suffix)
                write_vectors(base_file, base); write_vectors(query_file, queries)
                folder = self.root / (metric + suffix + str(dim))
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
                snapshot = self.root / ("snapshot" + metric + suffix + str(dim))
                ref_snapshot = file_index.save(snapshot)
                loaded = Index.open(ref_snapshot, base=base_file, threads=1)
                self.assert_results(f, loaded.search(queries, k=10, search_list_size=24))
                self.assertEqual(loaded.info["threads"], 1)

    def test_cli_snapshot_restores_python_calibration_across_batches(self):
        for metric, dtype, suffix in [("l2", np.float32, ".fbin"),
                                      ("cosine", np.float32, ".fvecs"),
                                      ("l2", np.uint8, ".u8bin")]:
            with self.subTest(metric=metric, dtype=dtype):
                folder = self.root / (metric + suffix)
                folder.mkdir()
                base, queries = self.arrays(dtype=dtype)
                base_path, query_path = folder / ("base" + suffix), folder / ("query" + suffix)
                write_vectors(base_path, base)
                write_vectors(query_path, queries)
                index = Index.from_file(base_path, metric=metric, graph=self.graph,
                                        cache_dir=folder / "build", threads=2)
                # The CLI will see different queries in batches of three. It must
                # restore this five-query calibration, including its original beam.
                calibration = index.calibrate(queries[:5], k=3, search_list_size=12)
                snapshot = index.save(folder / "snapshot")
                reopened = Index.open(snapshot, base=base_path, threads=1)
                expected = reopened.search(queries, k=3, search_list_size=24)
                command = [os.environ["ORION_TEST_CLI"], "--snapshot", snapshot.path,
                           "--base", str(base_path), "--query", str(query_path),
                           "--query-batch-size", "3", "--threads", "1", "--search-l", "24"]
                result = subprocess.run(command + ["--k", "3"], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                records = [json.loads(line) for line in result.stdout.splitlines()]
                rows, summary = records[:-1], records[-1]
                self.assertEqual([r["query_id"] for r in rows], list(range(len(queries))))
                self.assertEqual([r["ids"] for r in rows],
                                 [row[:n].tolist() for row, n in zip(expected.ids, expected.counts)])
                saved = dict(summary["calibration"])
                expected_calibration = vars(calibration).copy()
                self.assertEqual(np.float32(saved.pop("threshold")).tobytes(),
                                 np.float32(expected_calibration.pop("threshold")).tobytes())
                self.assertEqual(saved, expected_calibration)
                self.assertEqual(summary["queries"], len(queries))
                missing = subprocess.run(command + ["--k", "4", "--error-format", "json"],
                                         text=True, capture_output=True)
                self.assertNotEqual(missing.returncode, 0)
                self.assertIn("no saved calibration for k=4", missing.stderr)
                self.assertFalse(missing.stdout)
                conflict = subprocess.run(command + ["--cache-dir", str(folder / "wrong")],
                                          text=True, capture_output=True)
                self.assertNotEqual(conflict.returncode, 0)
                self.assertIn("--cache-dir", conflict.stderr)
                self.assertFalse((folder / "wrong").exists())

    def test_snapshot_requests_select_k_and_beam_without_mutating_calibration(self):
        # CLI snapshot search delegates to Rust Index::search; compare the same
        # restored generation against concurrent NumPy requests with mixed controls.
        for dtype, metric in [(np.uint8, "l2"), (np.float32, "cosine")]:
            with self.subTest(dtype=dtype, metric=metric):
                folder = self.root / metric
                folder.mkdir()
                base, queries = self.arrays(dtype=dtype, dimension=13)
                suffix = ".u8bin" if dtype == np.uint8 else ".fbin"
                base_path, query_path = folder / ("base" + suffix), folder / ("query" + suffix)
                write_vectors(base_path, base)
                write_vectors(query_path, queries)
                index = Index.from_file(base_path, metric=metric, graph=self.graph,
                                        cache_dir=folder / "build", threads=2)
                controls = [(3, 12), (5, 24), (10, 40)]
                for k, beam in controls:
                    index.calibrate(queries[:8], k=k, search_list_size=beam)
                reference = index.save(folder / "snapshot")
                before = index.info
                manifest = (Path(reference.path) / "index.json").read_bytes()
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(index.search, queries, k=k, search_list_size=beam)
                               for k, beam in controls]
                    for (k, beam), future in zip(controls, futures):
                        expected = future.result(timeout=20)
                        command = [os.environ["ORION_TEST_CLI"], "--snapshot", reference.path,
                                   "--base", str(base_path), "--query", str(query_path),
                                   "--query-batch-size", "3", "--threads", "2",
                                   "--k", str(k), "--search-l", str(beam)]
                        output = subprocess.run(command, text=True, capture_output=True, timeout=20)
                        self.assertEqual(output.returncode, 0, output.stderr)
                        records = [json.loads(line) for line in output.stdout.splitlines()]
                        rows = [record["ids"] for record in records if record["type"] == "result"]
                        self.assertEqual(rows, [row[:n].tolist() for row, n in zip(expected.ids, expected.counts)])
                self.assertEqual(index.info, before)
                self.assertEqual((Path(reference.path) / "index.json").read_bytes(), manifest)

    def test_array_snapshot_identity_and_calibration(self):
        base, queries = self.arrays()
        index = self.make(base)
        with self.assertRaisesRegex(OrionError, "calibrate"):
            index.search(queries, k=3)
        index.calibrate(queries, k=3)
        expected = index.search(queries, k=3, search_list_size=32)
        path = self.root / "saved"
        ref_path = index.save(path)
        restored = Index.open(SnapshotRef.from_json(ref_path.to_json()), base=base.copy())
        self.assert_results(expected, restored.search(queries, k=3, search_list_size=32))
        with self.assertRaisesRegex(OrionError, "calibrate"):
            restored.search(queries, k=4)
        changed = base.copy(); changed[0, 0] += 1
        with self.assertRaisesRegex(OrionError, "identity"):
            Index.open(ref_path, base=changed)
        with self.assertRaisesRegex(OrionError, "already exists"):
            index.save(path)
        local = Index.open(path, base=base)
        self.assert_results(expected, local.search(queries, k=3, search_list_size=32))
        with self.assertRaises(OrionError):
            Index.open(self.root / "missing", base=base)
        # Export changes after loading must not be misrecorded as the source
        # of the resident graph when another snapshot is saved.
        original_manifest = json.loads((path / "index.json").read_text())["payloads"]["provenance_sha256"]
        self.staged.write_bytes(b"changed after import")
        saved_again = self.root / "saved-again"; ref_saved_again = index.save(saved_again)
        self.assertEqual(original_manifest, json.loads((saved_again / "index.json").read_text())["payloads"]["provenance_sha256"])
        self.assert_results(expected, Index.open(ref_saved_again, base=base).search(queries, k=3, search_list_size=32))
        # Restored calibration cannot be silently combined with a changed cascade.
        metadata = json.loads((path / "index.json").read_text())
        metadata["options"]["window_size"] = 6
        (path / "index.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(OrionError, "manifest"):
            Index.open(ref_path, base=base)

    def test_runtime_dimensions_array_persistence_and_rust_builder(self):
        for dim in (3, 31, 33, 129, 1537):
            with self.subTest(dimension=dim):
                base, queries = self.arrays(dimension=dim)
                index = self.make(base)
                index.calibrate(queries, k=3)
                expected = index.search(queries, k=3, search_list_size=32)
                saved = self.root / ("odd-snapshot-" + str(dim))
                ref_saved = index.save(saved)
                self.assert_results(expected, Index.open(ref_saved, base=base.copy()).search(queries, k=3, search_list_size=32))
        base, queries = self.arrays(dimension=33)
        graph = GraphConfig(source="rust", degree=8, build_l=32, alpha=1.2, max_extra=4)
        index = Index.from_array(base, graph=graph, threads=2, cache_dir=self.root / "rust-odd")
        index.calibrate(queries, k=1)
        result = index.search(base[:3], k=1, search_list_size=64)
        np.testing.assert_array_equal(result.ids[:, 0], [0, 1, 2])

    def test_bad_inputs_are_errors(self):
        base, queries = self.arrays()
        for invalid in [base.astype(np.float64), base.reshape(-1), [[1, 2]], base.astype(">f4")]:
            with self.assertRaises(OrionError): self.make(invalid)
        with self.assertRaises(OrionError): self.make(base[:, ::-1])
        with self.assertRaises(OrionError): self.make(base[:, :0].copy())
        with self.assertRaises(OrionError): self.make(base[:0])
        broken = base.copy(); broken[0, 0] = np.nan
        with self.assertRaisesRegex(OrionError, "non-finite"): self.make(broken)
        with self.assertRaisesRegex(OrionError, "budget"):
            self.make(base, memory_budget_gib=1e-10)
        with self.assertRaises(OrionError) as invalid_threads:
            Index.from_array(base, threads=0, graph=self.graph, cache_dir=self.root / "bad")
        self.assertEqual(invalid_threads.exception.details["field"], "threads")
        index = self.make(base)
        for k, beam in [(0, 32), (65, 65), (10, 9)]:
            with self.assertRaises(OrionError): index.search(queries, k=k, search_list_size=beam)
        with self.assertRaises(OrionError): index.search(queries[:, :31])
        with self.assertRaises(OrionError): index.search(queries.astype(np.float64))
        with self.assertRaises(OrionError): index.calibrate(queries[:0])
        with self.assertRaises(OrionError) as missing:
            index.search(queries[:0], k=3)
        self.assertEqual(missing.exception.code, "ORION_INVALID_STATE")
        index.calibrate(queries, k=3)
        result = index.search(queries[:0], k=3)
        self.assertEqual(result.ids.shape, (0, 3))
        self.assertEqual(result.counts.shape, (0,))

    def test_owned_base_and_concurrent_calls(self):
        for dtype, metric in [(np.float32, "l2"), (np.float32, "cosine"), (np.uint8, "l2")]:
            with self.subTest(dtype=dtype, metric=metric):
                base, queries = self.arrays(dtype=dtype, dimension=13)
                original_queries = queries.copy()
                index = self.make(base, metric=metric)
                controls = [(3, 12), (5, 24), (10, 40)]
                for k, beam in controls:
                    index.calibrate(queries, k=k, search_list_size=beam)
                expected = [index.search(queries, k=k, search_list_size=beam) for k, beam in controls]
                before = index.info
                base[:] = 0  # Caller mutation cannot change the index's owned base.
                def request(i):
                    k, beam = controls[i % len(controls)]
                    return index.search(queries, k=k, search_list_size=beam)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(request, i) for i in range(24)]
                    for i, future in enumerate(futures):
                        self.assert_results(expected[i % len(controls)], future.result(timeout=20))
                np.testing.assert_array_equal(queries, original_queries)
                self.assertEqual(index.info, before)

    def test_native_u8_constraints_and_array_roundtrip(self):
        base, queries = self.arrays(np.uint8, 128)
        with self.assertRaises(OrionError):
            Index.from_array(base, graph=GraphConfig(source="rust"), cache_dir=self.root / "bad")
        built = Index.from_array(base, graph=GraphConfig(), cache_dir=self.root / "no-export")
        self.assertEqual(built.info["graph_source"], "parlayann")
        # Make every row reachable on the first expansion so exact self-matches
        # test native-u8 ranking independently of adaptive stopping.
        degree = len(base) - 1
        write_staged(self.staged, len(base), degree=degree, extra=0)
        graph = GraphConfig(source="parlayann", staged_file=self.staged,
                            degree=degree, build_l=64, max_extra=0)
        index = Index.from_array(base, graph=graph, threads=2, cache_dir=self.root / "complete")
        index.calibrate(queries, k=1)
        exact = index.search(base, k=1, search_list_size=64)
        np.testing.assert_array_equal(exact.ids[:, 0], np.arange(64))
        with self.assertRaises(OrionError): index.search(queries.astype(np.float32))
        with self.assertRaises(OrionError):
            self.make(base, cascade=CascadeConfig(rerank="f32"))
        index.calibrate(queries, k=10)
        snapshot = self.root / "native-saved"; ref_snapshot = index.save(snapshot)
        restored = Index.open(ref_snapshot, base=base)
        self.assert_results(index.search(queries, k=10), restored.search(queries, k=10))
        self.assertEqual(restored.info["storage"], "u8")

    def test_strict_cache_and_file_errors(self):
        base, queries = self.arrays()
        path = self.root / "base.custom"; write_vectors(path, base)
        index = Index.from_file(path, base_format="fbin", graph=self.graph, cache_dir=self.root / "file-cache")
        index.calibrate(queries, k=3)
        saved = self.root / "file-saved"; ref_saved = index.save(saved)
        manifest = json.loads((saved / "index.json").read_text())
        graph = Path(manifest["object_store"]) / (manifest["payloads"]["graph_sha256"] + ".pgraph")
        original = graph.read_bytes()
        graph.write_bytes(original[:-4])
        with self.assertRaises(OrionError): Index.open(ref_saved, base=path)
        with self.assertRaisesRegex(OrionError, "length mismatch"):
            index.save(self.root / "bad-graph-save")
        graph.write_bytes(original)
        bad_query = self.root / "bad.fbin"; write_vectors(bad_query, queries)
        bad_query.write_bytes(bad_query.read_bytes()[:-1])
        with self.assertRaises(OrionError): index.search(bad_query, k=3)
        saved2 = self.root / "file-saved2"; ref_saved2 = index.save(saved2)
        with path.open("r+b") as file: file.seek(8); file.write(struct.pack("<f", 1234))
        with self.assertRaisesRegex(OrionError, "identity"): Index.open(ref_saved2, base=path)
        with self.assertRaisesRegex(OrionError, "changed"): index.save(self.root / "stale-save")

    def test_auto_parlay_build_and_reuse(self):
        base, queries = self.arrays()
        index = Index.from_array(base, graph=GraphConfig(degree=8, build_l=32, max_extra=4), cache_dir=self.root / "rust-cache", threads=2)
        self.assertEqual(index.info["graph_source"], "parlayann")
        index.calibrate(queries, k=3)
        expected = index.search(queries, k=3, search_list_size=24)
        reused = Index.from_array(base, graph=GraphConfig(degree=8, build_l=32, max_extra=4), cache_dir=self.root / "rust-cache", threads=2)
        reused.calibrate(queries, k=3)
        self.assert_results(expected, reused.search(queries, k=3, search_list_size=24))
        self.assertIn("cache available", reused.info["graph_selection_reason"])

    def test_native_u8_memory_build_and_ffi_errors(self):
        base, queries = self.arrays()
        base = np.ascontiguousarray(np.clip(base * 10 + 128, 0, 255), dtype=np.uint8)
        queries = base[:8].copy()
        index = Index.from_array(base, graph=GraphConfig(degree=8, build_l=32, max_extra=4),
                                 cache_dir=self.root / "native-memory", threads=2)
        self.assertEqual(index.info["graph_source"], "parlayann")
        with self.assertRaises(OrionError) as missing:
            index.search(queries, k=3)
        self.assertEqual(missing.exception.code, "ORION_INVALID_STATE")
        index.calibrate(queries, k=3)
        self.assertEqual(index.search(queries, k=3).ids.shape, (8, 3))
        # This passes input validation but exceeds the C++ byte accumulator bound.
        with self.assertRaises(OrionError) as raised:
            Index.from_array(np.zeros((2, 33026), dtype=np.uint8),
                             graph=GraphConfig(degree=1, build_l=2, max_extra=0),
                             cache_dir=self.root / "bad-cpp", threads=1)
        self.assertEqual(raised.exception.code, "ORION_INVALID_ARGUMENT")
        self.assertEqual(raised.exception.operation, "build_graph")
        self.assertEqual(raised.exception.details["backend"], "parlayann")

    def test_high_dimension_and_top100(self):
        # Exercise another compiled specialization and the high-dimensional
        # default JL/L2-KT cascade, including restored lazy sidecars.
        base, queries = self.arrays(dimension=960)
        index = self.make(base)
        index.calibrate(queries, k=10)
        saved = self.root / "high-saved"; ref_saved = index.save(saved)
        restored = Index.open(ref_saved, base=base)
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
        saved = self.root / "prefix-saved"; ref_saved = index.save(saved)
        restored = Index.open(ref_saved, base=path)
        self.assertEqual(restored.info["num_points"], 32)
        self.assert_results(index.search(queries, k=3), restored.search(queries, k=3))
