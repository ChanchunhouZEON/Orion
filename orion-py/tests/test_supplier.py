# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Language-independent v4 producer and Python/CLI import parity."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from orion_ann import Index, OrionError, StorageOptions
from orion_ann._native import _Index


class SupplierAvailabilityTests(unittest.TestCase):
    def test_python_public_api_follows_compiled_feature(self):
        self.assertEqual(hasattr(Index, "from_staged"), hasattr(_Index, "from_staged"))


@unittest.skipUnless(hasattr(_Index, "from_staged"), "experimental-graph is disabled")
class SupplierTests(unittest.TestCase):
    def test_v4_file_import_and_cli_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dtype, element, suffix, metric in [(np.float32, "f32", ".fbin", "l2"),
                                                    (np.uint8, "u8", ".u8bin", "l2"),
                                                    (np.float32, "f32", ".fbin", "cosine")]:
                name = element + metric
                base = np.arange(48, dtype=dtype).reshape(16, 3) + 1
                queries = base[:8].copy()
                bp, qp = root / (name+suffix), root / (name+"q"+suffix)
                for path, array in [(bp, base), (qp, queries)]:
                    path.write_bytes(struct.pack("<II", *array.shape) + array.tobytes())
                spec = dict(origin=dict(name="python-producer", revision="demo"), points=16, dimension=3,
                            element=element, metric="squared-l2" if metric == "l2" else "cosine", entry=7,
                            max_degree=12, max_extra=3, base_sha256=hashlib.sha256(bp.read_bytes()).hexdigest())
                descriptor = json.dumps(spec).encode()
                staged = root / (name+".staged")
                payload = bytearray(struct.pack("<III", 0x53544147, 4, len(descriptor)) + descriptor)
                for node in range(16):
                    payload += struct.pack("<III", 6, 6, 3)
                    payload += struct.pack("<15I", *[(node+i) % 16 for i in range(1, 16)])
                staged.write_bytes(payload)
                index = Index.from_staged(bp, staged, metric=metric, threads=2, auto_pin=False,
                                          cache_dir=root/(name+"-py"), storage_options=StorageOptions(mode="mapped"))
                self.assertEqual(index.info["supplier"]["name"], "python-producer")
                index.calibrate(queries, k=3)
                result = index.search(queries, k=3, search_list_size=16, scores=True)
                args = [os.environ["ORION_TEST_CLI"], "--base", str(bp), "--query", str(qp), "--supplier-staged", str(staged),
                        "--metric", metric, "--threads", "2", "--k", "3", "--search-l", "16", "--scores",
                        "--no-auto-pin", "--cache-dir", str(root/(name+"-cli"))]
                cli = subprocess.run(args, capture_output=True, text=True)
                self.assertEqual(cli.returncode, 0, cli.stderr)
                rows = [json.loads(line) for line in cli.stdout.splitlines() if line.startswith("{")]
                rows = [row for row in rows if row.get("type") == "result"]
                np.testing.assert_array_equal([row["ids"] for row in rows], result.ids)
                np.testing.assert_allclose([row["scores"] for row in rows], result.scores, rtol=1e-6, atol=1e-6)
                snapshot = index.save(root/(name+"-snapshot"))
                restored = Index.open(snapshot, base=bp)
                np.testing.assert_array_equal(restored.search(queries, k=3, search_list_size=16).ids, result.ids)
                with self.assertRaises(OrionError):
                    Index.from_staged(bp, staged, metric="inner-product", cache_dir=root/"bad")
                # Duplicate a local edge into extras; metadata/length remain valid.
                offset = 12 + len(descriptor) + 12 + 12*4
                payload[offset:offset+4] = struct.pack("<I", 1)
                staged.write_bytes(payload)
                with self.assertRaises(OrionError):
                    Index.from_staged(bp, staged, metric=metric, cache_dir=root/"bad")


if __name__ == "__main__":
    unittest.main()
