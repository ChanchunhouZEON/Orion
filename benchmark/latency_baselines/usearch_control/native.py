# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""CDLL releases the GIL. A lifecycle gate keeps the index alive during calls."""
import ctypes as ct
from contextlib import contextmanager
from pathlib import Path
import threading
import numpy as np


def library(path, *, release_gil=True):
    # USearch itself loads the installed SimSIMD symbols globally. Reuse those
    # exact kernels, instead of compiling an independently selected SIMD backend.
    import usearch

    if usearch.__version__ != "2.23.0":
        raise ValueError("this control is pinned to USearch 2.23.0")
    loader = ct.CDLL if release_gil else ct.PyDLL
    lib = loader(str(Path(path).resolve()))
    declarations = {
        "uc_version": ([], ct.c_char_p),
        "uc_error": ([], ct.c_char_p),
        "uc_open": (
            [ct.c_char_p, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_int],
            ct.c_void_p,
        ),
        "uc_close": ([ct.c_void_p], None),
        "uc_isa": ([ct.c_void_p], ct.c_char_p),
        "uc_configure": ([ct.c_void_p, ct.c_size_t], ct.c_int),
        "uc_search": (
            [
                ct.c_void_p,
                ct.POINTER(ct.c_float),
                ct.c_size_t,
                ct.c_size_t,
                ct.POINTER(ct.c_ulonglong),
                ct.POINTER(ct.c_float),
                ct.POINTER(ct.c_size_t),
            ],
            ct.c_int,
        ),
        "uc_probe_new": ([], ct.c_void_p),
        "uc_probe_wait": ([ct.c_void_p], ct.c_int),
        "uc_probe_release": ([ct.c_void_p], ct.c_int),
        "uc_probe_close": ([ct.c_void_p], None),
    }
    for name, (args, result) in declarations.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = result
    if lib.uc_version().decode() != usearch.__version__:
        raise ValueError("adapter/wheel version mismatch")
    return lib


class NativeIndex:
    def __init__(self, lib, path, *, dimension, count, metric, slots=8):
        if not 1 <= slots <= 64 or dimension <= 0 or count <= 0:
            raise ValueError("invalid shape/slots")
        self.lib, self.dimension, self.count = lib, dimension, count
        self._held_lib = library(lib._name, release_gil=False)
        self._gate = threading.Condition()
        self._active = 0
        self._closing = False
        self._handle = lib.uc_open(
            str(path).encode(), slots, dimension, count, int(metric)
        )
        if not self._handle:
            raise ValueError(lib.uc_error().decode())
        self.isa = lib.uc_isa(self._handle).decode()

    @contextmanager
    def _borrow(self):
        with self._gate:
            if self._closing or not self._handle:
                raise RuntimeError("native index is closed")
            self._active += 1
            handle = self._handle
        try:
            yield handle
        finally:
            with self._gate:
                self._active -= 1
                self._gate.notify_all()

    def configure(self, effort):
        if type(effort) is not int or effort <= 0:
            raise ValueError("effort must be positive")
        with self._borrow() as handle:
            if self.lib.uc_configure(handle, effort):
                raise ValueError(self.lib.uc_error().decode())

    def search(self, query, k, *, release_gil=True):
        if (
            not isinstance(query, np.ndarray)
            or query.dtype != np.float32
            or query.shape != (self.dimension,)
            or not query.flags.c_contiguous
        ):
            raise ValueError("query must be a contiguous f32 row matching the index")
        if type(k) is not int or not 1 <= k <= self.count:
            raise ValueError("invalid k")
        keys, distances = np.empty(k, dtype=np.uint64), np.empty(k, dtype=np.float32)
        count = ct.c_size_t()
        with self._borrow() as handle:
            call_lib = self.lib if release_gil else self._held_lib
            status = call_lib.uc_search(
                handle,
                query.ctypes.data_as(ct.POINTER(ct.c_float)),
                self.dimension,
                k,
                keys.ctypes.data_as(ct.POINTER(ct.c_ulonglong)),
                distances.ctypes.data_as(ct.POINTER(ct.c_float)),
                ct.byref(count),
            )
            if status:
                raise ValueError(self.lib.uc_error().decode())
        return keys[: count.value], distances[: count.value]

    def close(self):
        with self._gate:
            self._closing = True
            self._gate.wait_for(lambda: self._active == 0)
            if self._handle:
                self.lib.uc_close(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
