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
                  cache_dir: PathLike = "cache/orion-py", threads=8, window_size=5,
                  memory_budget_gib=None, max_points=None, base_format=None):
        """Load base directly into Rust; reuse, import or construct the graph."""
        options = _options(metric, storage, graph, cascade, cache_dir, threads,
                           window_size, memory_budget_gib, max_points, base_format)
        return cls(_Index.from_file(os.fspath(base), options))

    @classmethod
    def from_array(cls, base: np.ndarray, *, metric="l2", storage=None,
                   graph: Optional[GraphConfig] = None, cascade: Optional[CascadeConfig] = None,
                   cache_dir: PathLike = "cache/orion-py", threads=8, window_size=5,
                   memory_budget_gib=None, max_points=None):
        """Copy base once into Rust-owned aligned storage; hash all array content."""
        options = _options(metric, storage, graph, cascade, cache_dir, threads,
                           window_size, memory_budget_gib, max_points, None)
        return cls(_Index.from_array(base, options))

    @classmethod
    def open(cls, cache: PathLike, *, base: Input, threads=None, memory_budget_gib=None):
        """Strictly reopen a saved snapshot; never fall back to constructing a graph."""
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
        """
        record = self._handle.calibrate(_input(queries), k, search_list_size, query_format)
        return Calibration(**json.loads(record))

    def search(self, queries: Input, *, k=10, search_list_size=64,
               adaptive=True, query_format=None) -> SearchResult:
        """Search a batch. Adaptive search requires prior calibration for this k.

        Files are decoded as a complete batch. For bounded memory, pass separate
        ndarray batches. No recall evaluation or cache flushing occurs here.
        """
        ids, counts = self._handle.search(_input(queries), k, search_list_size,
                                         adaptive, query_format)
        return SearchResult(ids, counts)

    def save(self, cache: PathLike) -> None:
        """Save graph/configuration/calibration into a new directory, without base."""
        self._handle.save(os.fspath(cache))
