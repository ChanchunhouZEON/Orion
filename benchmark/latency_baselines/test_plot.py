# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Publication selection must not inflate gains or silently mix configurations."""
import unittest
from .plot import aggregate, qualified, select, validate_latencies


def point(engine, recall, p50, p99, dataset="sift"):
    return dict(dataset=dataset, engine=engine, concurrency=8, recall=recall,
                effort=20, p50_us=p50, p95_us=(p50+p99)/2, p99_us=p99)


class PublicationTests(unittest.TestCase):
    def test_one_setting_supplies_all_percentiles(self):
        rows = [point("orion", .95, 10, 100), point("orion", .99, 20, 30)]
        chosen = select(rows, "sift", "orion", 8, .95)
        self.assertEqual(chosen["p99_us"], 100)
        self.assertEqual(select(rows, "sift", "orion", 8, .98)["p99_us"], 30)

    def test_no_interpolation_or_native_extrapolation(self):
        rows = [point("usearch-native", .96, 15, 20)]
        self.assertIsNone(select(rows, "sift", "usearch-native", 8, .98))
        self.assertIsNone(select(rows, "gist", "usearch-native", 8, .95))
        self.assertIsNone(select(rows, "sift", "usearch", 8, .95))

    def test_aggregate_uses_common_datasets_and_geomean(self):
        rows = [point("orion", .96, 10, 20), point("hnswlib", .97, 20, 40),
                point("orion", .96, 10, 20, "gist"), point("hnswlib", .97, 80, 160, "gist"),
                point("hnswlib", .99, 1000, 2000, "glove25")]
        result = aggregate(rows, .95, 8, "hnswlib")
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["ratios"]["p50_us"], 4)
        self.assertEqual(aggregate(rows, .99, 8, "hnswlib")["ratios"], None)

    def test_under_sampled_and_inconsistent_ids(self):
        protocol = dict(rounds=3, min_requests=10000, min_seconds=1)
        trial = dict(minimum_reached=True, latency=dict(requests=10000), wall_seconds=1.2)
        p = dict(rounds=[trial]*3, complete=True, concurrent_ids_match=True)
        self.assertTrue(qualified(p, protocol))
        p.update(rounds=[trial]*2, complete=False)
        self.assertFalse(qualified(p, protocol))
        p["concurrent_ids_match"] = False
        with self.assertRaises(ValueError):
            qualified(p, protocol)

    def test_nonfinite_and_reversed_percentiles_rejected(self):
        for p in [point("orion", .95, float("nan"), 20), point("orion", .95, 20, 10)]:
            with self.assertRaises(ValueError):
                validate_latencies(p)


if __name__ == "__main__":
    unittest.main()
