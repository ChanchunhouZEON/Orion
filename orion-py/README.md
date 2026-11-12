<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Orion Python interface

The distribution `orion-ann` exposes `orion-ann` through PyO3 and NumPy.
Import it as `orion_ann`; the extension `orion_ann._native` is private.
The source directory remains `orion-py`.

## Local development

Run Cargo from the repository root to apply its target/SIMD configuration. Use
Rust 1.92 and a Python environment with NumPy and maturin installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy 'maturin>=1.9,<2'
maturin develop --manifest-path orion-py/Cargo.toml --profile python
```

The `python` Cargo profile uses release optimization with panic unwinding. Do
not substitute the workspace's `release` profile: it uses `panic=abort`, which
cannot translate unexpected Rust panics into Python exceptions. SIMD CPU
requirements are the same as for the CLI; see [platforms](../docs/platforms.md).

For a local test run without installing or building a wheel:

```bash
python3 orion-py/tests/run.py
# Optimized extension + CLI, with the same consistency checks:
python3 orion-py/tests/run.py --profile python
```

The runner builds the extension and CLI, creates a temporary Python package,
and removes its fixtures after testing. It does not read experiment datasets,
update figures, install a package globally, or publish anything.

## File inputs

```python
from orion_ann import Index, GraphConfig

index = Index.from_file(
    '/data/base.fbin', metric='l2', threads=8,
    cache_dir='/data/orion-build-cache',
    graph=GraphConfig(source='auto', degree=64, build_l=100),
)
index.calibrate('/data/warmup.fbin', k=10)
result = index.search('/data/queries.fbin', k=10, search_list_size=100)
print(result.ids, result.counts)
snapshot = index.save('/data/orion-snapshot')  # New directory; does not copy base vectors.

reopened = Index.open(snapshot, base='/data/base.fbin', threads=8)
result = reopened.search('/data/queries.fbin', k=10, search_list_size=100)
```

`fvecs`, `bvecs`, `fbin`, and `u8bin` use the CLI's readers. Dimensions come from
the file header. Use `base_format=` / `query_format=` for other extensions.
`max_points=` limits a file base to its prefix. File bases are loaded directly
into Rust aligned storage, without a Python base allocation.

Unlike CLI presets, embedded options do not read YAML or legacy CLI environment
variables. Graph and cascade settings are explicit or dimension/metric defaults.
File cache identities remain compatible with the CLI when base, representation,
graph source and construction settings match. Select `storage='f32'` explicitly
if byte files should be widened; otherwise byte inputs select native u8.

## NumPy inputs

```python
import numpy as np
from orion_ann import Index

base = np.ascontiguousarray(my_vectors, dtype=np.float32)
queries = np.ascontiguousarray(my_queries, dtype=np.float32)
index = Index.from_array(base, metric='l2', threads=8)
index.calibrate(queries[:200], k=10)
result = index.search(queries, k=10, search_list_size=100)
```

- Arrays must be two-dimensional, C-contiguous and native-endian `float32` or
  `uint8`. Float64 is rejected; there is no implicit quantization or cast.
- The base is copied once into Rust-owned, SIMD-aligned memory. Subsequent
  changes to the NumPy base do not change the index. Slice the base explicitly
  before construction if only a prefix is wanted.
- Query batches are copied before releasing the GIL; heavy graph operations,
  calibration and search execute without the GIL. Do not mutate inputs during
  their copy. Searches on one handle share a read lock and a fixed-size Rayon
  pool. Calibration uses an exclusive write lock. A backend panic invalidates
  the handle; reopen it before retrying.
- File queries are decoded as a complete batch. For bounded query memory, call
  `search` with successive ndarray batches; calibrate once using a warmup batch.
- Dimensions are inferred at runtime; each resident row is padded for SIMD.
  The Rust/DiskANN graph builder supports logical dimensions up to 1536 by
  padding its working input; larger dimensions need a PA graph.
  Native u8 supports squared L2, inner product and cosine, using the bundled
  PA builder or a PA cache/STAG export. It does not fall back to Rust construction.
  Query arrays accept float32, uint8, int8, uint16 and uint32 independently of
  base dtype. Search and calibration share this contract. File queries additionally
  support i8bin/u16bin/u32bin with a little-endian two-u32 shape header.
  Exact scoring retains wide integer coordinates instead of silently narrowing
  them to float32; returned scores remain float32.

For native SIFT bytes, `GraphConfig()` can build directly from the array or file.
Use `staged_file=...` to import an existing export. `source='auto'` follows the
CLI policy: PA cache, PA export, then in-memory PA build. Explicit sources stay
strict. Building from source requires a C++17 compiler; no external ParlayANN
executable or checkout is required.

## Calibration, results and persistence

`calibrate(queries, k=..., search_list_size=None)` uses all supplied rows and
returns `Calibration`. Default calibration L is `max(48, 2*k)`. The CLI uses its
first 200 query rows by default; supply the same rows to compare its parameters.
Different k values have separate calibration records. `CascadeConfig` selects
`prefilter`, `admission` and `rerank` using CLI names, with the same validation.

`search(queries, k=k)` uses the calibration for that `k`. A missing record
raises `OrionError` with code `ORION_INVALID_STATE`, including for an empty batch. Results contain uint32
`ids` of shape `(queries, k)` and uint32 `counts` of shape `(queries,)`. Only the
first `counts[i]` IDs of a row are valid; unused positions contain `UINT32_MAX`.
There is no fabricated distance array or automatic recall evaluation.

`save(new_directory)` publishes one manifest with configuration, calibration and
references to the SHA-256 objects in the sibling `<cache_dir>.objects` store.
Keep this store when clearing the working cache or copying a snapshot. Base vectors remain external. Reopen with
`Index.open(directory, base=...)` loads and verifies the current snapshot.
Pass the `SnapshotRef` returned by save to also pin the manifest identity.
Keep `snapshot.to_json()` in trusted application configuration and restore it
with `SnapshotRef.from_json(...)` after restarting. Missing stages are constructed
in memory. Rebuild snapshots created with older layouts.

`index.info` reports shape, metric, storage, actual graph source, cascade,
thread count and calibrations. Python type/shape errors raise `TypeError` or
`ValueError`; runtime validation, cache and I/O failures raise `OrionError` with
the underlying context. A worker search error is propagated rather than replaced
with an empty result.

`memory_budget_gib=` applies the shared lower-bound preflight to base/index
preparation; it is not a hard RSS cap or a bound on Python arrays, query/result
batches, allocator overhead or all builder scratch allocations.

## Errors

Orion validation, configuration, storage and search failures raise `OrionError`,
a subclass of `RuntimeError`. The Rust API, CLI and Python binding share one
error model. Python does not rephrase the backend message:

```python
from orion_ann import OrionError

try:
    result = index.search(queries, k=10)
except OrionError as error:
    print(error.code)       # e.g. ORION_INVALID_STATE
    print(error.message)    # human-readable explanation
    print(error.operation)  # closest operation when available
    print(error.details)    # expected/actual shape, path, memory components, ...
    print(error.causes)     # preserved source chain
    print(error.report)     # same JSON-compatible report used by the CLI
```

Use `code` and `details` to handle errors; do not parse exception text. Correcting
an ordinary validation error permits retrying the same index. `ORION_INTERNAL`
with `requires_reopen` indicates an unexpected Rust panic: discard/reopen that
index. Further calls on a poisoned handle return `ORION_INVALID_STATE`.
Python argument-binding errors (such as an unknown keyword or an integer that
cannot fit a Rust argument) remain normal Python exceptions.

The extension requires the `python` Cargo profile (`panic="unwind"`); an abort
build is rejected. The boundary catches unwinding panics, including those
propagated from Rayon workers. It cannot recover from aborts, process termination
or allocator failures that abort. It does not replace the process-wide panic
hook. The `test-hooks` Cargo feature is only for the local fault-injection suite
and is excluded from normal builds.

CLI services can request `--error-format json`; error reports go to stderr and
successful JSONL output is unchanged. For the embedded Rust API and signatures,
run `cargo doc -p orion-ann-core -p orion-ann --no-deps`.
