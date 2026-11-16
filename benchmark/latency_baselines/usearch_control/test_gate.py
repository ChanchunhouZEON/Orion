# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
import json
from pathlib import Path
import tempfile
import unittest
from .__main__ import gate


class GateTests(unittest.TestCase):
    def test_running_process_blocks_even_with_completed_files(self):
        self.assertEqual(gate({"wait_identity": "original"}, "original")[0], False)

    def test_missing_or_failed_result_blocks_after_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(wait_identity="original", wait_results=tmp, wait_jobs=["a"])
            with self.assertRaises(RuntimeError):
                gate(config, "")
            path = Path(tmp) / "a"
            path.mkdir()
            for status in ["running", "failed", "preparing"]:
                (path / "result.json").write_text(json.dumps({"status": status}))
                with self.assertRaises(RuntimeError):
                    gate(config, "")
            for status in ["complete", "incomplete", "missing", "unsupported"]:
                (path / "result.json").write_text(json.dumps({"status": status}))
                self.assertTrue(gate(config, "")[0])
                self.assertTrue(gate(config, "reused PID, different start time")[0])
