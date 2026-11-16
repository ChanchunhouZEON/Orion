# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Tiny end-to-end harness check, not a performance measurement."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .worker import run
from .__main__ import summarize


@unittest.skipUnless(
    os.getenv("USEARCH_CONTROL_LIBRARY"), "build private adapter first"
)
class WorkerTests(unittest.TestCase):
    def test_three_arms_clients_rounds_and_owned_cache_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "tiny"
            out.mkdir()
            rng = np.random.default_rng(13)
            base = rng.normal(size=(256, 32)).astype("f4")
            query = rng.normal(size=(16, 32)).astype("f4")
            gt = np.argsort(
                ((query[:, None, :] - base[None, :, :]) ** 2).sum(axis=2), axis=1
            )[:, :10]
            for name, data in [("base", base), ("query", query), ("gt", gt)]:
                np.save(root / f"{name}.npy", data)
            dataset = dict(
                name="tiny",
                metric="squared-l2",
                base=str(root / "base.npy"),
                query=str(root / "query.npy"),
                groundtruth=str(root / "gt.npy"),
                engines={"usearch": {}},
            )
            job = dict(
                dataset=dataset,
                k=10,
                efforts=[32],
                library=os.environ["USEARCH_CONTROL_LIBRARY"],
                protocol=dict(
                    concurrency=[1, 8],
                    rounds=3,
                    min_requests=32,
                    min_seconds=0.002,
                    max_seconds=2,
                    warmup_requests=8,
                    warmup_seconds=0,
                    cooldown_seconds=0,
                ),
            )
            (out / "job.json").write_text(json.dumps(job))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(out / "job.json"), 0)
            record = json.loads((out / "result.json").read_text())
            self.assertEqual(record["status"], "complete")
            self.assertEqual(len(record["points"]), 6)
            self.assertEqual(len(list(out.glob("*.npy"))), 18)
            self.assertFalse(list(out.glob("shared-index-*")))
            for point in record["points"]:
                self.assertEqual(len(point["rounds"]), 3)
                self.assertTrue(point["complete"])
            summarize(root)
            self.assertIn("native-nogil", (root / "summary.md").read_text())
