# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Native result wrapping preserves the public frozen dataclass contract."""
import copy
import dataclasses
import gc
import inspect
import pickle
from pathlib import Path
import unittest

import numpy as np
from orion_ann import SearchResult, LabelFilter, OrionError, _input, _search_result
import test_index as fixtures


class SearchResultContractTests(unittest.TestCase):
    def test_native_wrapper_keeps_fields_defaults_and_array_ownership(self):
        ids = np.arange(8, dtype=np.uint32).reshape(2, 4)
        counts = np.array([4, 2], dtype=np.uint32)
        for scored in (False, True):
            for external in (False, True):
                scores = np.zeros((2, 4), dtype=np.float32) if scored else None
                metric = "squared_l2" if scored else None
                business = ids.astype(np.uint64) + 1000 if external else None
                fields = (ids, counts, scores, metric, business)
                actual = _search_result(fields)
                ordinary = SearchResult(*fields)
                self.assertIs(type(actual), SearchResult)
                self.assertTrue(dataclasses.is_dataclass(actual))
                self.assertEqual(set(vars(actual)), {f.name for f in dataclasses.fields(SearchResult)})
                for field in dataclasses.fields(SearchResult):
                    self.assertIs(getattr(actual, field.name), getattr(ordinary, field.name))
                self.assertEqual(repr(actual), repr(ordinary))
                self.assertIs(actual.ids.base, ids.base)
        default = SearchResult(ids, counts)
        self.assertIsNone(default.scores)
        self.assertIsNone(default.score_metric)
        self.assertIsNone(default.external_ids)
        self.assertEqual(list(inspect.signature(SearchResult).parameters),
                         [f.name for f in dataclasses.fields(SearchResult)])
        # Wrapping takes ownership of Python references, without copying arrays.
        del ordinary, fields, ids, counts
        gc.collect()
        np.testing.assert_array_equal(actual.ids, np.arange(8).reshape(2, 4))

    def test_frozen_replace_copy_pickle_and_asdict_still_work(self):
        result = _search_result((np.array([[1, 2]], dtype=np.uint32),
                                 np.array([2], dtype=np.uint32),
                                 np.array([[0.1, 0.2]], dtype=np.float32),
                                 "squared_l2", np.array([[101, 102]], dtype=np.uint64)))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.ids = None
        with self.assertRaises(dataclasses.FrozenInstanceError):
            del result.counts
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.extra = True
        changed = dataclasses.replace(result, scores=None, score_metric=None)
        self.assertIs(changed.ids, result.ids)
        self.assertIsNone(changed.scores)
        self.assertIs(copy.copy(result).counts, result.counts)
        for restored in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
            self.assertIs(type(restored), SearchResult)
            self.assertEqual(restored.score_metric, result.score_metric)
            for field in ("ids", "counts", "scores", "external_ids"):
                np.testing.assert_array_equal(getattr(restored, field), getattr(result, field))
        materialized = dataclasses.asdict(result)
        self.assertEqual(set(materialized), set(vars(result)))
        np.testing.assert_array_equal(materialized["ids"], result.ids)

    def test_input_fast_path_preserves_subclass_and_path_behavior(self):
        class ArrayView(np.ndarray):
            pass

        class ArrayPath(np.ndarray):
            def __fspath__(self):
                return "queries.fbin"

        class CustomPath:
            def __fspath__(self):
                return "custom.fbin"

        array = np.zeros((2, 13), dtype=np.float32)
        view = array.view(ArrayView)
        self.assertIs(_input(array), array)
        self.assertIs(_input(view), view)
        self.assertEqual(_input(array.view(ArrayPath)), "queries.fbin")
        self.assertEqual(_input(CustomPath()), "custom.fbin")
        self.assertEqual(_input(Path("queries.fbin")), "queries.fbin")
        self.assertEqual(_input("queries.fbin"), "queries.fbin")
        unsupported = object()
        self.assertIs(_input(unsupported), unsupported)


class SearchResultIntegrationTests(unittest.TestCase):
    setUp = fixtures.IndexTests.setUp
    arrays = fixtures.IndexTests.arrays
    make = fixtures.IndexTests.make

    def test_search_and_filter_match_ordinary_result_construction(self):
        for dtype in (np.float32, np.uint8):
            for metric in ("squared-l2", "inner-product", "cosine"):
                with self.subTest(dtype=dtype, metric=metric):
                    base, queries = self.arrays(dtype=dtype, dimension=13)
                    original = queries.copy()
                    index = self.make(base, metric=metric, external_ids=list(range(1000, 1064)))
                    index.calibrate(queries, k=3, search_list_size=24)
                    for n in (0, 1, 4):
                        for scores in (False, True):
                            expected = SearchResult(*index._handle.search(
                                queries[:n], 3, 24, query_format=None, scores=scores))
                            actual = index.search(queries[:n], k=3, search_list_size=24, scores=scores)
                            self.assert_result(expected, actual)
                    # Filtered queries share wrapping, while keeping their existing
                    # exact subset search and immutable label revision semantics.
                    filtered = index.search_filtered(queries[:1], LabelFilter(), k=3, scores=True)
                    self.assertIs(type(filtered.result), SearchResult)
                    self.assertEqual(filtered.result.ids.shape, (1, 3))
                    self.assertEqual(filtered.label_revision, 0)
                    np.testing.assert_array_equal(queries, original)

    def assert_result(self, expected, actual):
        self.assertIs(type(actual), SearchResult)
        self.assertEqual(expected.score_metric, actual.score_metric)
        for name in ("ids", "counts", "scores", "external_ids"):
            left, right = getattr(expected, name), getattr(actual, name)
            if left is None:
                self.assertIsNone(right)
            else:
                self.assertEqual(left.dtype, right.dtype)
                np.testing.assert_array_equal(left, right)


class NativeBoundaryTests(unittest.TestCase):
    setUp = fixtures.IndexTests.setUp
    arrays = fixtures.IndexTests.arrays
    make = fixtures.IndexTests.make

    def test_direct_dataclass_keeps_ownership_freezing_and_serialization(self):
        base, queries = self.arrays(dimension=13)
        index = self.make(base, external_ids=list(range(1000, 1064)))
        index.calibrate(queries, k=3, search_list_size=24)
        result = index.search(queries[:2], k=3, search_list_size=24, scores=True)
        self.assertIs(type(result), SearchResult)
        self.assertEqual(set(vars(result)), {f.name for f in dataclasses.fields(SearchResult)})
        before = result.ids.copy()
        other = index.search(queries[2:4], k=3, search_list_size=24, scores=True)
        self.assertFalse(np.shares_memory(result.ids, other.ids))
        self.assertFalse(np.shares_memory(result.counts, other.counts))
        del index, queries, other
        gc.collect()
        np.testing.assert_array_equal(result.ids, before)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.ids = None
        with self.assertRaises(dataclasses.FrozenInstanceError):
            del result.counts
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.extra = True
        self.assertIs(dataclasses.replace(result, scores=None).ids, result.ids)
        self.assertIs(copy.copy(result).ids, result.ids)
        for restored in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
            self.assertIs(type(restored), SearchResult)
            np.testing.assert_array_equal(restored.ids, result.ids)
            np.testing.assert_array_equal(restored.scores, result.scores)
            np.testing.assert_array_equal(restored.external_ids, result.external_ids)
        np.testing.assert_array_equal(dataclasses.asdict(result)["ids"], result.ids)

    def test_native_input_dispatch_and_pathlike_precedence(self):
        class ArrayView(np.ndarray):
            pass
        base, queries = self.arrays(dimension=13)
        index = self.make(base)
        index.calibrate(queries, k=3, search_list_size=24)
        for dtype in (np.float32, np.uint8, np.int8, np.uint16, np.uint32):
            with self.subTest(dtype=dtype):
                values = np.ascontiguousarray(queries.astype(dtype))
                saved = values.copy()
                expected = index.search(values.astype(np.float32), k=3, search_list_size=24)
                for array in (values, values.view(ArrayView)):
                    actual = index.search(array, k=3, search_list_size=24)
                    np.testing.assert_array_equal(actual.ids, expected.ids)
                    np.testing.assert_array_equal(actual.counts, expected.counts)
                np.testing.assert_array_equal(values, saved)
        path = self.root / "queries.fbin"
        fixtures.write_vectors(path, queries)
        calls = []
        class QueryPath:
            def __fspath__(self):
                calls.append("path")
                return str(path)
        class ArrayPath(np.ndarray):
            def __fspath__(self):
                calls.append("array")
                return str(path)
        expected = index.search(queries, k=3, search_list_size=24)
        for source in (str(path), path, QueryPath(), queries.view(ArrayPath)):
            actual = index.search(source, k=3, search_list_size=24, query_format="fbin")
            np.testing.assert_array_equal(actual.ids, expected.ids)
        self.assertEqual(calls, ["path", "array"])
        index.calibrate(QueryPath(), k=3, search_list_size=24, query_format="fbin")
        index.calibrate_many(QueryPath(), ks=[3], search_list_size=24, query_format="fbin")
        filtered = index.search_filtered(QueryPath(), LabelFilter(), k=3, query_format="fbin")
        self.assertEqual(len(filtered.result.counts), len(queries))
        self.assertEqual(calls, ["path", "array", "path", "path", "path"])
        for bad in (object(), queries.astype(np.float64), queries.astype(np.bool_),
                    queries.astype(queries.dtype.newbyteorder("S")), queries[0], queries[:, ::2]):
            with self.subTest(bad_type=type(bad)):
                with self.assertRaises(OrionError):
                    index.search(bad, k=3, search_list_size=24)
        with self.assertRaises(OrionError):
            index.search(queries, k=3, search_list_size=24, query_format="fbin")
        # Rejected inputs do not poison a valid index.
        self.assertIs(type(index.search(queries[:1], k=3, search_list_size=24)), SearchResult)
