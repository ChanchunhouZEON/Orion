"""Owned Orion indexes with explicit calibration and NumPy batch search."""
from dataclasses import asdict, dataclass
import json
import os
from typing import Optional, Union

import numpy as np
from ._native import _Index, OrionError

PathLike = Union[str, os.PathLike]
Input = Union[PathLike, np.ndarray]
__all__ = ["Index", "GraphConfig", "CascadeConfig", "Calibration", "SearchResult", "OrionError"]


@dataclass(frozen=True)
class GraphConfig:
    """Construction policy: auto, rust or parlayann.

    A staged_file selects ParlayANN and conflicts with source='rust'. Native u8
    requires a PA graph. degree > 0, build_l >= degree and finite alpha >= 1
    are validated during construction; a dataclass alone does not validate.
    """
    source: str = "auto"
    staged_file: Optional[PathLike] = None
    degree: int = 64
    build_l: int = 100
    alpha: float = 1.2
    max_extra: int = 16


@dataclass(frozen=True)
class CascadeConfig:
    prefilter: Optional[str] = None
    admission: Optional[str] = None
    rerank: Optional[str] = None


@dataclass(frozen=True)
class Calibration:
    k: int
    search_list_size: int
    samples: int
    threshold: float
    early_exit_limit: int


@dataclass(frozen=True)
class SearchResult:
    """Row-ordered uint32 IDs; counts excludes trailing UINT32_MAX padding."""
    ids: np.ndarray
    counts: np.ndarray


def _input(value: Input):
    return os.fspath(value) if isinstance(value, (str, os.PathLike)) else value


def _options(metric, storage, graph, cascade, cache_dir, threads, window_size,
             memory_budget_gib, max_points, base_format):
    graph = asdict(graph or GraphConfig())
    if graph["staged_file"] is not None:
        graph["staged_file"] = os.fspath(graph["staged_file"])
    return json.dumps(dict(metric=metric, storage=storage, graph=graph,
        cascade=asdict(cascade or CascadeConfig()), cache_dir=os.fspath(cache_dir),
        threads=threads, window_size=window_size, memory_budget_gib=memory_budget_gib,
        max_points=max_points, base_format=base_format), allow_nan=False)


class Index:
    """An owned base, graph and dedicated thread pool.

    Construct with from_file/from_array/open. Operations on one Index serialize;
    each batch uses the configured number of workers with the Python GIL released.
    Array inputs must be C-contiguous float32 or uint8; no implicit dtype casting.
    """
    def __init__(self, handle):
        if not isinstance(handle, _Index):
            raise TypeError("use Index.from_file, Index.from_array or Index.open")
        self._handle = handle

    @classmethod
    def from_file(cls, base: PathLike, *, metric="l2", storage=None,
                  graph: Optional[GraphConfig] = None, cascade: Optional[CascadeConfig] = None,
                  cache_dir: PathLike = "cache/orion-ann", threads=8, window_size=5,
                  memory_budget_gib=None, max_points=None, base_format=None):
        """Load base into Rust; reuse, import or construct its graph.

        Dimension/count come from the file. memory_budget_gib is an estimate,
        not an allocation cap. This may write graph cache files; concurrent
        builders need separate cache directories. Raises OrionError with code,
        details and causes on input, configuration, cache, I/O or backend errors.
        """
        options = _options(metric, storage, graph, cascade, cache_dir, threads,
                           window_size, memory_budget_gib, max_points, base_format)
        return cls(_Index.from_file(os.fspath(base), options))

    @classmethod
    def from_array(cls, base: np.ndarray, *, metric="l2", storage=None,
                   graph: Optional[GraphConfig] = None, cascade: Optional[CascadeConfig] = None,
                   cache_dir: PathLike = "cache/orion-ann", threads=8, window_size=5,
                   memory_budget_gib=None, max_points=None):
        """Copy a nonempty 2-D C-contiguous float32/uint8 array into owned storage.

        The original array may be reused after return. Logical dimension is the
        column count; alignment/padding is internal. No implicit dtype casting.
        Raises OrionError for invalid arrays or construction failures. The input
        copy occurs before the estimated memory-budget check.
        """
        options = _options(metric, storage, graph, cascade, cache_dir, threads,
                           window_size, memory_budget_gib, max_points, None)
        return cls(_Index.from_array(base, options))

    @classmethod
    def open(cls, cache: PathLike, *, base: Input, threads=None, memory_budget_gib=None):
        """Strictly reopen a trusted snapshot, supplying its unchanged base again.

        A miss never rebuilds. Calibration is restored; threads/budget may be
        overridden. Raises OrionError for stale identity/configuration, malformed
        metadata or I/O failures. Provenance checks are not authentication of
        arbitrary hostile files. Keep the base/cache unchanged during this call.
        """
        return cls(_Index.open(os.fspath(cache), _input(base), threads, memory_budget_gib))

    @property
    def info(self) -> dict:
        """Shape, resolved graph/cascade, pool size and per-k calibration records."""
        return json.loads(self._handle.info())

    def calibrate(self, queries: Input, *, k=10, search_list_size=None,
                  query_format=None) -> Calibration:
        """Use all supplied warmup rows; the CLI default uses its first 200 rows.

        Default calibration L is max(48, 2*k), exactly as in the CLI. Calibration
        is specific to this index/cascade and k, and is included in save().
        Raises OrionError for empty/malformed input, invalid k/beam or backend
        failures. Successful records for other k values are retained.
        """
        record = self._handle.calibrate(_input(queries), k, search_list_size, query_format)
        return Calibration(**json.loads(record))

    def search(self, queries: Input, *, k=10, search_list_size=64,
               adaptive=True, query_format=None) -> SearchResult:
        """Search a batch. Adaptive search requires prior calibration for this k.

        Files are decoded as a complete batch. For bounded memory, pass separate
        ndarray batches. No recall evaluation or cache flushing occurs here.
        Require 1 <= k <= point_count and k <= search_list_size <= u32::MAX/4.
        Native-u8 queries must have exact integral values in [0, 255]. Returned
        uint32 IDs preserve row order; use counts to exclude UINT32_MAX padding.
        Empty batches are supported after control/calibration validation.

        Raises OrionError: missing calibration is ORION_INVALID_STATE; shape,
        value and backend failures carry structured reports. Ordinary invalid
        input can be corrected and retried. An unexpected Rust unwind instead
        invalidates the native handle: reopen it before further operations.
        """
        ids, counts = self._handle.search(_input(queries), k, search_list_size,
                                         adaptive, query_format)
        return SearchResult(ids, counts)

    def save(self, cache: PathLike) -> None:
        """Save graph/configuration/calibration in a new directory, without base.

        A staging directory is renamed after writes succeed. Serialize writers
        to the same destination. Raises OrionError for an existing destination,
        changed base, serialization or I/O failure; the loaded index remains
        usable. Rename visibility is not a full filesystem crash-durability claim.
        """
        self._handle.save(os.fspath(cache))
