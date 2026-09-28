import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "parlayann_convert.py"
spec = importlib.util.spec_from_file_location("converter", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / "source", self.root / "target"

    def test_byte_and_float_formats(self):
        for kind, pack in (("bvecs", lambda row: bytes(row)),
                           ("fvecs", lambda row: struct.pack("<3f", *row))):
            rows = [(0, 1, 255), (2, 3, 4), (5, 6, 7)]
            self.source.write_bytes(b"".join(struct.pack("<I", 3) + pack(row) for row in rows))
            converter.convert_vectors(self.source, self.target, kind, chunk_bytes=7)
            self.assertEqual(self.target.read_bytes(), struct.pack("<II", 3, 3) + b"".join(map(pack, rows)))
            binary = "u8bin" if kind == "bvecs" else "fbin"
            copy = self.root / "copy"
            converter.convert_vectors(self.target, copy, binary, limit=2, chunk_bytes=5)
            self.assertEqual(copy.read_bytes(), struct.pack("<II", 2, 3) + b"".join(map(pack, rows[:2])))

    def test_bad_dimensions_preserve_previous_output(self):
        self.source.write_bytes(struct.pack("<I3BI3B", 3, 1, 2, 3, 4, 1, 2, 3))
        self.target.write_bytes(b"old result")
        with self.assertRaises(ValueError):
            converter.convert_vectors(self.source, self.target, "bvecs", chunk_bytes=7)
        self.assertEqual(self.target.read_bytes(), b"old result")
        self.assertEqual(list(self.root.glob("*.tmp.*")), [])

    def test_truncated_and_empty_inputs(self):
        for payload in (b"", struct.pack("<I", 0), struct.pack("<I", 3) + b"12"):
            self.source.write_bytes(payload)
            with self.assertRaises(ValueError):
                converter.convert_vectors(self.source, self.target, "bvecs")

    def test_gt_layout_and_subset_rejection(self):
        self.source.write_bytes(struct.pack("<6I", 2, 0, 1, 2, 1, 2))
        converter.convert_ground_truth(self.source, self.target, 2, 3, chunk_bytes=5)
        self.assertEqual(self.target.read_bytes(), struct.pack("<6I", 2, 2, 0, 1, 1, 2) + bytes(16))
        with self.assertRaises(ValueError):
            converter.convert_ground_truth(self.source, self.target, 2, 2)

    def test_flush_failure_preserves_output(self):
        self.source.write_bytes(struct.pack("<I3B", 3, 1, 2, 3))
        self.target.write_bytes(b"old")
        with patch.object(converter.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                converter.convert_vectors(self.source, self.target, "bvecs")
        self.assertEqual(self.target.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
