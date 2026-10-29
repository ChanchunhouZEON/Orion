# Orion Python interface

`orion-py` exposes the shared `orion-runtime` through PyO3 and NumPy. The import
name is `orion_py`; the extension is private (`orion_py._native`). No wheel is
published as part of this work.

## Local development

Run Cargo from the repository root to apply its target/SIMD configuration. Use
Rust 1.92 and a Python environment with NumPy and maturin installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy 'maturin>=1.5,<2'
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
from orion_py import Index, GraphConfig

index = Index.from_file(
    '/data/base.fbin', metric='l2', threads=8,
    cache_dir='/data/orion-build-cache',
    graph=GraphConfig(source='auto', degree=64, build_l=100),
)
index.calibrate('/data/warmup.fbin', k=10)
result = index.search('/data/queries.fbin', k=10, search_list_size=100)
print(result.ids, result.counts)
index.save('/data/orion-snapshot')  # New directory; does not copy base vectors.

reopened = Index.open('/data/orion-snapshot', base='/data/base.fbin', threads=8)
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
from orion_py import Index

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
  their copy. Each index owns a fixed-size Rayon pool; simultaneous operations
  on the same index serialize, while its batch search runs in parallel.
- File queries are decoded as a complete batch. For bounded query memory, call
  `search` with successive ndarray batches; calibrate once using a warmup batch.
- Supported physical dimensions: 32, 100, 128, 768, 784, 960 and 1536. Native u8
  currently requires 128-dimensional L2 and a PA cache or STAG export. It does
  not fall back to Rust construction, allocate a quantized admission copy, or
  rerank. Native-u8 ndarray queries must also be uint8.

For native SIFT bytes, supply `GraphConfig(source='parlayann', staged_file=...,
degree=..., build_l=..., max_extra=...)`. `source='auto'` follows the CLI policy:
PA cache, PA export, then Rust cache/build for f32. Explicit sources stay strict.

## Calibration, results and persistence

`calibrate(queries, k=..., search_list_size=None)` uses all supplied rows and
returns `Calibration`. Default calibration L is `max(48, 2*k)`. The CLI uses its
first 200 query rows by default; supply the same rows to compare its parameters.
Different k values have separate calibration records. `CascadeConfig` selects
`prefilter`, `admission` and `rerank` using CLI names, with the same validation.

`search(..., adaptive=True)` requires calibration for that k. Setting
`adaptive=False` disables convergence and early exit. Results contain uint32
`ids` of shape `(queries, k)` and uint32 `counts` of shape `(queries,)`. Only the
first `counts[i]` IDs of a row are valid; unused positions contain `UINT32_MAX`.
There is no fabricated distance array or automatic recall evaluation.

`save(new_directory)` writes the graph, configuration, base identity and all
calibrations, then publishes the directory. Derived sidecars are rebuilt lazily
from the base on reopen. `open(directory, base=...)` requires a complete matching
snapshot and never builds a missing graph. Base vectors remain external:

- File identity uses canonical path, size and modification time, matching the
  existing CLI convention. Keep that file unchanged and at the same path.
- Array identity uses SHA-256 of its complete original contents, dtype and shape.
- Graph file size/timestamps and an execution-settings signature bind saved
  calibrations to the graph/cascade/window. Truncation, stale input or changed
  execution settings fail explicitly; these are local caches, not an untrusted
  interchange format.

`index.info` reports shape, metric, storage, actual graph source, cascade,
thread count and calibrations. Python type/shape errors raise `TypeError` or
`ValueError`; runtime validation, cache and I/O failures raise `OrionError` with
the underlying context. A worker search error is propagated rather than replaced
with an empty result.

`memory_budget_gib=` applies the shared lower-bound preflight to base/index
preparation; it is not a hard RSS cap or a bound on Python arrays, query/result
batches, allocator overhead or all builder scratch allocations.
