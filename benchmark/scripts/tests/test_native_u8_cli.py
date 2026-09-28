"""Optional real CLI coverage; set ORION_BINARY to a freshly built executable."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

BINARY = os.environ.get("ORION_BINARY")


@unittest.skipUnless(BINARY, "set ORION_BINARY to exercise native-byte sweeps")
class NativeByteCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = [bytes(i * 7 + j % 3 for j in range(128)) for i in range(16)]
        records = [struct.pack("<I", 128) + row for row in rows]
        (self.root / "base.bvecs").write_bytes(b"".join(records))
        (self.root / "query.bvecs").write_bytes(b"".join(records[:4]))
        (self.root / "gt.ivecs").write_bytes(b"".join(struct.pack("<II", 1, i) for i in range(4)))
        with (self.root / "graph.staged").open("wb") as graph:
            graph.write(struct.pack("<6I", 0x53544147, 3, 16, 15, 0, 0))
            for i in range(16):
                graph.write(struct.pack("<18I", 15, 0, 0, *(j for j in range(16) if j != i)))
        self.options = ["--base", self.root / "base.bvecs", "--query", self.root / "query.bvecs",
                        "--groundtruth", self.root / "gt.ivecs", "--staged-file", self.root / "graph.staged",
                        "--cache-dir", self.root / "cache", "--graph-degree", 15, "--max-extra", 0,
                        "--k", 1, "--search-list-sizes", 16, "--threads", 2, "--trials", 1]

    def run_cli(self, dataset, *extra, success=True):
        result = subprocess.run([str(Path(BINARY).resolve()), dataset, *map(str, self.options), *extra],
                                text=True, capture_output=True, timeout=90)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_all_large_presets_use_one_byte_base(self):
        for dataset in ["sift10m", "sift100m", "sift1b"]:
            with self.subTest(dataset=dataset):
                report = json.loads(self.run_cli(dataset, "--preflight").stdout)
                self.assertEqual(report["vector_storage"], "u8")
                self.assertEqual(report["base_u8_bytes"], 16 * 128 + 64)
                self.assertEqual(report["base_f32_bytes"], 0)
                self.assertEqual(report["l2_u8_admission_bytes"], 0)

    def test_import_reload_and_search_have_no_f32_rerank_or_sidecar(self):
        self.run_cli("sift10m", "--prepare-only")
        (self.root / "graph.staged").unlink()
        result = self.run_cli("sift10m")
        log = result.stdout + result.stderr
        self.assertIn('"vector_storage": "u8"', log)
        self.assertIn('"rerank": "None"', log)
        self.assertRegex(log, r"R@1=1\.0000")
        self.assertRegex(log, r"f32=0\)")
        self.assertEqual(list((self.root / "cache").glob("*.qds*")), [])

    def test_incompatible_override_fails_before_loading(self):
        result = self.run_cli("sift100m", "--rerank", "f32", success=False)
        self.assertIn("native u8 requires", result.stderr)
        self.assertFalse((self.root / "cache").exists())


if __name__ == "__main__":
    unittest.main()
