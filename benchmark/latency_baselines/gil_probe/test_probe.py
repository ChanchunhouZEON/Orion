# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Run with the isolated interpreter, never the system Python."""
import concurrent.futures
import struct
import threading
import time
import unittest
import _gilprobe as probe


def decode(raw):
    return list(struct.iter_unpack('=QQQQ', raw))


class ProbeTests(unittest.TestCase):
    def tearDown(self):
        probe.finish()

    def test_disabled_capacity_overflow_and_restart(self):
        probe.begin(1)
        probe.end()
        self.assertEqual(probe.finish(), (b'', 0))
        for cap in (0, -1, 2000001):
            with self.assertRaises(ValueError): probe.start(cap)
        probe.start(2)
        probe.begin(99)
        probe.end()
        probe.begin(100)
        probe.end()
        raw, dropped = probe.finish()
        self.assertEqual([e[2] for e in decode(raw)], [13, 14])
        self.assertEqual(dropped, 2)
        self.assertEqual(probe.finish(), (b'', 0))
        probe.start(10)
        probe.start(20)
        self.assertEqual(probe.finish(), (b'', 0))

    def test_thread_local_order_and_gil_release(self):
        barrier = threading.Barrier(8)
        def worker(i):
            identity = probe.start(10000)
            try:
                barrier.wait(timeout=10)
                probe.begin(i + 1)
                time.sleep(.005)
                probe.end()
            finally:
                raw, lost = probe.finish()
            events = decode(raw)
            self.assertEqual(lost, 0)
            self.assertEqual([e[0] for e in events], sorted(e[0] for e in events))
            tagged = [e for e in events if e[1] == i + 1]
            codes = [e[2] for e in tagged]
            self.assertEqual(codes[0], 13)
            self.assertEqual(codes[-1], 14)
            self.assertIn(7, codes)
            self.assertIn(8, codes)
            self.assertIn(1, codes)
            self.assertIn(6, codes)
            for e in tagged:
                if e[2] == 6: self.assertEqual(e[3], identity)
            self.assertFalse(any(e[1] not in (0, i+1) for e in events))
            return identity
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            identities = list(pool.map(worker, range(8)))
        self.assertEqual(len(set(identities)), 8)


if __name__ == '__main__': unittest.main()
