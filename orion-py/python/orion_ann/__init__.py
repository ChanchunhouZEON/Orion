# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Owned Orion indexes with explicit calibration and NumPy batch search."""
from dataclasses import asdict, dataclass
import json
import os
from typing import Optional, Union

import numpy as np
from ._native import _Index, OrionError

PathLike = Union[str, os.PathLike]
Input = Union[PathLike, np.ndarray]
__all__ = ["Index", "GraphConfig", "CascadeConfig", "StorageOptions", "Calibration", "SearchResult", "SnapshotRef", "OrionError", "LabelKey", "LabelUpdate", "ExternalLabelUpdate", "LabelFilter", "FilteredSearchResult"]


@dataclass(frozen=True)
class LabelKey:
    """Exact case-sensitive namespace/value pair; Rust validates input limits."""
    namespace: str
    value: str


@dataclass(frozen=True)
class LabelUpdate:
    """Add/remove internal row IDs atomically with the other changes in a batch."""
    label: LabelKey
    add: tuple[int, ...] = ()
    remove: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExternalLabelUpdate:
    """Add/remove u64 business IDs; runtime converts the entire batch atomically."""
    label: LabelKey
    add: tuple[int, ...] = ()
    remove: tuple[int, ...] = ()


@dataclass(frozen=True)
class LabelFilter:
    """ALL(all) AND ANY(any) AND NOT ANY(none); empty any means unrestricted."""
    all: tuple[LabelKey, ...] = ()
    any: tuple[LabelKey, ...] = ()
    none: tuple[LabelKey, ...] = ()


@dataclass(frozen=True)
class FilteredSearchResult:
    """One immutable label revision for the complete query batch."""
    result: "SearchResult"
    label_revision: int


@dataclass(frozen=True)
class SnapshotRef:
    """Location and manifest identity returned by Index.save()."""
    path: str
    manifest_sha256: str

    def relocated(self, path: PathLike) -> "SnapshotRef":
        """Retain the manifest identity after moving the snapshot directory."""
        return SnapshotRef(os.fspath(path), self.manifest_sha256)

    def to_json(self) -> str:
        """Serialize for storage in trusted application configuration."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, value: str) -> "SnapshotRef":
        """Restore a reference previously saved by the application."""
        return cls(**json.loads(value))


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
    scores: Optional[np.ndarray] = None
    score_metric: Optional[str] = None
    external_ids: Optional[np.ndarray] = None


def _input(value: Input):
    return os.fspath(value) if isinstance(value, (str, os.PathLike)) else value


@dataclass
class StorageOptions:
    """Base, graph and sidecar placement; directory should be disk-backed.

    mode is auto, resident, or mapped. Placement does not
    change scores or calibration and is not a hard RSS limit.
    """
    mode: str = "auto"
    directory: Optional[PathLike] = None


def _storage_options(value):
    data = asdict(value or StorageOptions())
    if data["directory"] is not None:
        data["directory"] = os.fspath(data["directory"])
    return data


def _options(metric, storage, graph, cascade, cache_dir, threads, window_size,
             memory_budget_gib, max_points, base_format, auto_pin, storage_options):
    graph = asdict(graph or GraphConfig())
    if graph["staged_file"] is not None:
        graph["staged_file"] = os.fspath(graph["staged_file"])
    return json.dumps(dict(metric=metric, storage=storage, graph=graph,
        cascade=asdict(cascade or CascadeConfig()), cache_dir=os.fspath(cache_dir),
        threads=threads, window_size=window_size, memory_budget_gib=memory_budget_gib,
        max_points=max_points, base_format=base_format, auto_pin=auto_pin, storage_options=_storage_options(storage_options)), allow_nan=False)


class Index:
    """An owned base, graph and dedicated thread pool.

    Construct with from_file/from_array/open. Searches may share one Index;
    concurrent batches share the configured worker pool with the GIL released.
    Calibration holds an exclusive lock. Array inputs are copied before release.
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
                  memory_budget_gib=None, max_points=None, base_format=None, auto_pin=True, storage_options: Optional[StorageOptions] = None, external_ids=None):
        """Load base into Rust; reuse, import or construct its graph.

        Dimension/count come from the file. memory_budget_gib is an estimate,
        not an allocation cap. This may write graph cache files; concurrent
        builders sharing a cache receive an error on a writer conflict; use separate directories to build concurrently. Raises OrionError with code,
        details and causes on input, configuration, cache, I/O or backend errors.
        """
        options = _options(metric, storage, graph, cascade, cache_dir, threads,
                           window_size, memory_budget_gib, max_points, base_format, auto_pin, storage_options)
        return cls(_Index.from_file(os.fspath(base), options, external_ids))

    @classmethod
    def from_array(cls, base: np.ndarray, *, metric="l2", storage=None,
                   graph: Optional[GraphConfig] = None, cascade: Optional[CascadeConfig] = None,
                   cache_dir: PathLike = "cache/orion-ann", threads=8, window_size=5,
                   memory_budget_gib=None, max_points=None, auto_pin=True, storage_options: Optional[StorageOptions] = None, external_ids=None):
        """Copy a nonempty 2-D C-contiguous float32/uint8 array into owned storage.

        The original array may be reused after return. Logical dimension is the
        column count; alignment/padding is internal. No implicit dtype casting.
        Raises OrionError for invalid arrays or construction failures. The input
        copy occurs before the estimated memory-budget check.
        """
        options = _options(metric, storage, graph, cascade, cache_dir, threads,
                           window_size, memory_budget_gib, max_points, None, auto_pin, storage_options)
        return cls(_Index.from_array(base, options, external_ids))

    @classmethod
    def open(cls, snapshot: Union[PathLike, SnapshotRef], *, base: Optional[Input] = None, threads=None, memory_budget_gib=None, auto_pin=None, storage_options: Optional[StorageOptions] = None, object_store: Optional[PathLike] = None):
        """Verify and reopen a snapshot directory or a saved SnapshotRef.

        With a reference, the manifest must also match the one issued by save().
        Omit base only for bundles exported with include_base=True. object_store
        relocates shared payloads without bypassing verification.
        """
        source = (snapshot.to_json() if isinstance(snapshot, SnapshotRef)
                  else json.dumps(os.fspath(snapshot)))
        options = dict(threads=threads, memory_budget_gib=memory_budget_gib,
                       auto_pin=auto_pin,
                       storage_options=None if storage_options is None else _storage_options(storage_options),
                       object_store=None if object_store is None else os.fspath(object_store))
        return cls(_Index.open(source, _input(base), json.dumps(options)))

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

    def calibrate_many(self, queries: Input, *, ks, search_list_size=None,
                       query_format=None) -> list[Calibration]:
        """Explicitly prepare all requested k values; publish records together."""
        records = self._handle.calibrate_many(_input(queries), list(ks), search_list_size, query_format)
        return [Calibration(**value) for value in json.loads(records)]

    @property
    def calibrated_ks(self) -> list[int]:
        """Prepared k values; inspecting them never performs calibration."""
        return sorted(int(k) for k in self.info["calibrations"])

    def calibration(self, k: int) -> Optional[Calibration]:
        record = self.info["calibrations"].get(str(k))
        return None if record is None else Calibration(**record)

    @property
    def label_revision(self) -> int:
        return self._handle.label_revision()

    def update_labels(self, expected_revision: int, changes: list[LabelUpdate]) -> int:
        """Atomically publish a batch; stale revisions or invalid IDs publish nothing.

        Success is in-memory visibility. Save a snapshot for durability. Limits:
        4096 labels and 1M memberships; at most 1M supplied row IDs per batch.
        Duplicate IDs are idempotent; overlapping add/remove sets are rejected.
        """
        return self._handle.update_labels(expected_revision, json.dumps([asdict(c) for c in changes]))

    def resolve_external_ids(self, ids: list[int]) -> list[int]:
        """Resolve business IDs in input order; performs one scan of the ID mapping."""
        return self._handle.resolve_external_ids(ids)

    def update_labels_external(self, expected_revision: int,
                               changes: list[ExternalLabelUpdate]) -> int:
        """Resolve business IDs and atomically publish all label changes in Rust.

        Requires an external-ID mapping, even for an empty batch. Unknown add or
        remove IDs, invalid changes and revision conflicts publish nothing. The
        entire batch scans the mapping once; no permanent reverse map is kept.
        Budgets, duplicate handling and explicit-save durability match update_labels.
        """
        return self._handle.update_labels_external(
            expected_revision, json.dumps([asdict(c) for c in changes]))

    def search_filtered(self, queries: Input, filter: LabelFilter, *, k=10,
                        scores=False, max_candidates=100_000, query_format=None) -> FilteredSearchResult:
        """Exact subset scan, without adaptive calibration. Unknown labels match nothing.

        One label snapshot is held for all query rows. Rejects matching sets above
        max_candidates before vector evaluation. Original metric scores, count
        prefixes and partial-result padding follow search(). Input arrays are copied.
        This first implementation is an exact baseline, not a filtered ANN path.
        """
        values, revision = self._handle.search_filtered(
            _input(queries), json.dumps(asdict(filter)),
            json.dumps(dict(k=k, scores=scores, max_candidates=max_candidates)), query_format)
        return FilteredSearchResult(SearchResult(*values), revision)

    def search(self, queries: Input, *, k=10, search_list_size=64,
               query_format=None, scores=False) -> SearchResult:
        """Search a batch. Search uses the calibration for this k.

        Files are decoded as a complete batch. For bounded memory, pass separate
        ndarray batches. No recall evaluation or cache flushing occurs here.
        Require 1 <= k <= point_count and k <= search_list_size <= u32::MAX/4.
        Native-u8 queries must have exact integral values in [0, 255]. Returned
        uint32 IDs preserve row order; use counts to exclude UINT32_MAX padding.
        Empty batches are supported after control/calibration validation.

        scores=True adds float32 scores of the returned candidates and a
        score_metric label. Squared L2 is lower-is-better; inner product and
        cosine similarity are higher-is-better. Scores preserve result order
        and do not imply exact global top-k. Ignore NaN padding using counts.

        Raises OrionError: missing calibration is ORION_INVALID_STATE; shape,
        value and backend failures carry structured reports. Ordinary invalid
        input can be corrected and retried. An unexpected Rust unwind instead
        invalidates the native handle: reopen it before further operations.
        """
        ids, counts, values, metric, external = self._handle.search(_input(queries), k, search_list_size,
                                         query_format=query_format, scores=scores)
        return SearchResult(ids, counts, values, metric, external)

    def export_bundle(self, destination: PathLike, *, include_base=False) -> SnapshotRef:
        """Copy verified objects into a relocatable bundle.

        include_base requires a file-built index; arrays are supplied again to open.
        The reference names destination/snapshot. With an included base, open()
        needs no base argument. Existing destinations are never overwritten.
        """
        return SnapshotRef.from_json(self._handle.export_bundle(os.fspath(destination), include_base))

    def save(self, cache: PathLike) -> SnapshotRef:
        """Save a manifest referencing the shared object store and external base.

        A staging directory is renamed after writes succeed. Conflicting writers
        to the same destination are rejected. Raises OrionError for an existing destination,
        changed base, serialization or I/O failure; the loaded index remains
        usable. Rename visibility is not a full filesystem crash-durability claim.
        """
        return SnapshotRef.from_json(self._handle.save(os.fspath(cache)))
