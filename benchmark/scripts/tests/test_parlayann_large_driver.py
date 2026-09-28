"""Optional subprocess tests enabled by PARLAY_NEIGHBORS, with tiny generated data."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_parlayann_large.py"
BINARY = os.environ.get("PARLAY_NEIGHBORS")


@unittest.skipUnless(BINARY, "set PARLAY_NEIGHBORS to test the build driver")
class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base.bvecs"
        rows = [bytes((i * 17 + j * 13 + i * j) % 256 for j in range(128)) for i in range(128)]
        self.base.write_bytes(b"".join(struct.pack("<I", 128) + row for row in rows))
        self.output = self.root / "run"

    def run_driver(self, budget="1"):
        return subprocess.run([sys.executable, str(SCRIPT), "--base", str(self.base),
            "--neighbors", str(Path(BINARY).resolve()), "--out-dir", str(self.output),
            "--threads", "2", "--degree", "8", "--build-l", "16", "--max-extra", "4",
            "--memory-budget-gib", budget, "--no-resource-time"],
            capture_output=True, text=True, timeout=60)

    def test_conversion_preflight_build_and_provenance(self):
        result = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((self.output / "run.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual([stage["name"] for stage in report["stages"]], ["convert", "preflight", "build"])
        self.assertEqual(len(report["artifacts"]), 3)
        self.assertEqual(len(report["executable"]["sha256"]), 64)
        self.assertFalse(report["peak_rss_available"])
        self.assertEqual((self.output / "base.u8bin").stat().st_size, 8 + 128 * 128)

    def test_failed_preflight_is_recorded_and_build_does_not_run(self):
        self.assertNotEqual(self.run_driver("0.000001").returncode, 0)
        report = json.loads((self.output / "run.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stages"][-1]["name"], "preflight")
        self.assertEqual(report["stages"][-1]["status"], "failed")
        self.assertFalse((self.output / "graph.staged").exists())

    def test_existing_output_directory_is_not_reused(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("keep")
        self.assertNotEqual(self.run_driver().returncode, 0)
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
