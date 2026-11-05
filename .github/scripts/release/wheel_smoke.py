# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Run using an isolated installed interpreter: python -I wheel_smoke.py."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import orion_ann
from orion_ann import GraphConfig, Index, OrionError
from orion_ann._native import _Index

assert Path(orion_ann.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), orion_ann.__file__
assert not hasattr(_Index, '_test_panic'), 'production wheel contains test hooks'


def expect_error(code, call):
    try:
        call()
    except OrionError as error:
        assert error.code == code, error.report
        assert error.operation, error.report
        return error
    raise AssertionError(f'Expected {code}')


with tempfile.TemporaryDirectory(prefix='orion-installed-') as directory:
    root = Path(directory)
    rng = np.random.default_rng(17)
    for dtype in [np.float32, np.uint8]:
        base = rng.integers(0, 200, (96, 13), dtype=np.uint8).astype(dtype)
        queries = base[:8].copy()
        folder = root / np.dtype(dtype).name
        folder.mkdir()
        # source defaults to auto; no cache or STAG file exists to hide a build.
        graph = GraphConfig(degree=8, build_l=24, max_extra=4)
        index = Index.from_array(base, threads=2, graph=graph, cache_dir=folder / 'build')
        assert index.info['graph_source'] == 'parlayann', index.info
        manifests = list((folder / 'build').rglob('*.cache.json'))
        assert manifests, 'builder did not write cache provenance'
        for path in manifests:
            meta = json.loads(path.read_text())
            assert meta['builder_recipe'] == 'parlay-memory-v1-unquantized-two-pass', meta
            assert meta.get('imported_from') is None, meta
        expect_error('ORION_INVALID_STATE', lambda: index.search(queries, k=3, search_list_size=24))
        index.calibrate(queries, k=3, search_list_size=24)
        result = index.search(queries, k=3, search_list_size=24)
        assert result.ids.shape == (8, 3) and result.ids.dtype == np.uint32
        assert np.all(result.counts == 3)
        snapshot = index.save(folder / 'snapshot')
        restored = Index.open(snapshot, base=base, threads=1)
        again = restored.search(queries, k=3, search_list_size=24)
        np.testing.assert_array_equal(result.ids, again.ids)
        np.testing.assert_array_equal(result.counts, again.counts)
        expect_error('ORION_DIMENSION_MISMATCH', lambda: restored.search(
            np.ones((1, 14), dtype=dtype), k=3, search_list_size=24))
        # Ordinary input errors must not poison the handle.
        np.testing.assert_array_equal(result.ids, restored.search(queries, k=3, search_list_size=24).ids)
    error = expect_error('ORION_INVALID_ARGUMENT', lambda: Index.from_array(
        np.zeros((2, 33026), dtype=np.uint8), threads=1, cache_dir=root / 'ffi-error',
        graph=GraphConfig(degree=1, build_l=2, max_extra=0)))
    assert error.operation == 'build_graph' and error.details['backend'] == 'parlayann'
print(json.dumps({'status': 'passed', 'python': sys.version, 'numpy': np.__version__,
                  'module': orion_ann.__file__, 'backend': 'default ParlayANN',
                  'storage': ['f32', 'u8'], 'production_hooks': False}))
