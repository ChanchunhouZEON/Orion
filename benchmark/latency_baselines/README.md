<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
# Baseline request latency

`python -m benchmark.latency_baselines` compares **single-query request latency** for
Orion, ParlayANN, DiskANN/Vamana, hnswlib, USearch, LanceDB IVF-HNSW-SQ,
FAISS HNSW-SQ8, FAISS IVF-Flat, FAISS IVF-PQ and Annoy. These are the runnable
engines in the existing baseline scripts. The disabled Milvus reference is not
silently substituted with LanceDB or reported as a measured engine.

This harness times the **Python single-request adapter API**, including query
preparation, binding/GIL overhead and result assembly. ParlayANN and DiskANN use
private `ctypes` adapters that release the GIL around their existing native search.
Orion uses the current Python `Index.search`. This gives a common application
boundary, but it is **not** the pure-Rust `latency-sweep` boundary: do not mix those
percentiles or infer an algorithmic speedup from binding costs. See
[native Rust request latency](rust-request-latency.md) for that separate harness. The harness does
not measure HTTP, open-loop queueing, or latency under a fixed arrival rate.

## Prepare adapters and configuration

Use a Python environment containing NumPy and the desired engines; imports are
lazy and missing dependencies are reported. Versions tested locally are listed
in `requirements-tested.txt`. Install a **current** Orion wheel built with the
`python` profile; a stale editable installation may contain an incompatible
native extension. Build instructions remain in `orion-py/README.md`.

Build the two native benchmark adapters once, using flags appropriate for the
measurement host. The directory contains only local libraries and provenance;
no production crate API or ParlayANN search kernel is modified.

```bash
cd benchmark
RUSTFLAGS='-C target-cpu=native' CXXFLAGS='-march=native' \
  python -m latency_baselines.build_native \
  --parlay-root ../../ParlayANN --output ../../experiment/YYYY-MM-DD/baseline-native
```

`build.json` records commands, compiler version, flags, ParlayANN commit and
library SHA-256 values. This builds for the **local** host, not a cross target.
The native Rust helper has its own unpublished workspace/lockfile. Request
threads use sequential searches; ParlayANN explicitly initializes a one-worker
scheduler per caller so eight callers do not create eight eight-worker pools.
Index preparation can use the configured eight-worker pool outside timing.

Generate a comparison configuration from the seven existing Orion request-latency
JSON configs. This reuses the recorded ParlayANN recipes in `_parlay.sh`:

```bash
python -m latency_baselines.make_config \
  --orion-configs ../../experiment/YYYY-MM-DD/request-latency/configs \
  --parlay-root ../../ParlayANN \
  --native-dir ../../experiment/YYYY-MM-DD/baseline-native \
  --cache-root ../../experiment/YYYY-MM-DD/baseline-cache \
  --diskann-cache ../cache/diskann \
  --output ../../experiment/YYYY-MM-DD/baseline-comparison.json
```

`--diskann-cache` may be repeated. Exactly one matching cache is selected;
missing/ambiguous caches are printed and remain missing in the final report.
Review the resulting paths and effort ladders. Individual engines use independent
ladders: beam/ef, nprobe, or Annoy search_k. Equal numeric effort is **not** equal
recall. The generator keeps Orion's existing ladder; extend it explicitly if
needed. No graph or dataset is built by configuration generation.

Alternatively, copy `example.json` and edit paths. Paths are relative to that
JSON. Each dataset requires base/query/groundtruth, an explicit metric and an
engine map. Supported vector inputs: fvecs, bvecs, fbin, u8bin and npy; GT accepts
ivecs, ibin, npy or ParlayANN bin with an optional appended distance matrix.
Coordinates are supplied to baseline APIs as float32. No dataset prefix is
silently taken, since that would invalidate the ground truth.

## Run

From the repository root:

```bash
python -m benchmark.latency_baselines \
  --config ../experiment/YYYY-MM-DD/baseline-comparison.json \
  --output ../experiment/YYYY-MM-DD/baseline-latency
```

By default, all ten engines are requested and **no missing index is built**.
Missing recipes/dependencies/files, unsupported metrics, incomplete measurements
and failures stay visible in the report and yield nonzero exit status.
`--allow-missing` permits only missing/unsupported rows; real failures still fail.
Use `--engines orion,parlayann,diskann` or `--datasets sift,gist` for a subset.

Existing Python baseline scripts discarded several in-memory indices. If no
persisted index exists, explicitly add `--prepare-missing` to build **only missing
Python baseline indices**, once, before timing. The script never rebuilds native
DiskANN/ParlayANN graphs, overwrites an existing index, or falls back to another
engine. Orion opens a snapshot or reuses an explicit STAG recipe; it never
silently performs a fresh graph build. Python caches carry an input-stat/recipe
identity (not an adversarial integrity signature). A mismatch requires a new
cache path. Native DiskANN cached coordinates are checked against the logical
base, including normalization and zero padding. Legacy caches without manifests
require explicit `build: {degree, build_l, alpha}`; the generator extracts these
from unambiguous `*_l2.bin` recipe names and still verifies all cached coordinates.
Raw inner product is rejected because its L2 backend is not generally equivalent.
An explicit `unit_norm_ip: true` permits the established unit-norm IP comparison
only after every base row passes a norm check (rtol/atol 2e-6). It does not
normalize an arbitrary raw-IP base into a different benchmark. ParlayANN's
`normalize: true` likewise requires unit-norm canonical inputs for IP. Current
raw MS-MARCO data fails this equivalence check and remains unsupported by DiskANN.

Each dataset/engine runs in an isolated process, loads **once** and completes all
search efforts, caller counts and rounds. Process exit releases memory before
the next engine. An independent job timeout bounds stuck native calls. Graphs
and input files are read only. Only the selected output/cache directories receive
new files; existing result directories are rejected.

## Measurement and outputs

Defaults match the Orion latency experiment's load protocol:

- `k=10`, index/build thread budget 8, **1 and 8 concurrent callers**, **3 rounds**.
  FAISS/OpenMP, BLAS, hnswlib and USearch single requests use one worker; these
  thread-local settings are applied on every caller. Internal library scheduling
  is not a hard OS CPU quota. Environment and package versions are recorded.
- At least 1 s and 10,000 requests per condition, also enough to cover the full
  query set. A 30 s ceiling is checked between calls; unmet minima are incomplete.
- Warm up at least 1,000 requests total and 0.2 s per caller, then synchronize
  start. Cool down 1 s between trials. Reverse concurrency order each round.
- One seeded shared query cycle prevents a faster client from over-sampling its
  own query shard. Inputs are decoded before timing. Per-request normalization
  and quantization remain inside the adapter call where needed.
- Load/build/calibrate, recall evaluation, recording and result destruction are
  outside the request timer. Cache policy is **warm/steady-state**, no flush.
- Strict top-k ID recall, no distance-tie expansion. Each query is evaluated once
  per point, outside timing. Concurrent IDs must match serial IDs. Duplicate or
  out-of-range returned IDs fail; legitimate partial results reduce recall.
- ParlayANN retains its configured quantization/rerank recipe; its historical
  batch-prepared queries are deliberately replaced by per-request preparation.
  `native/test_parlay.cpp` compares the resulting IDs against upstream qsearchAll.

Outputs:

- `comparison.md`: measured equal-recall target table, P50/P95/P99 and QPS.
- `curves.csv`: every measured effort/recall/latency point, including incomplete
  flags. No interpolation. Each target selects the lowest pooled P50 among
  complete configurations reaching it; all reported values come from that point.
- `comparison.json`, per-engine `result.json`, `job.json`, `worker.log`: full
  provenance, status, per-round quantiles, P99 range and effective Orion settings.
- Per-round `.npy`: every raw latency in nanoseconds, with SHA-256. Pooled quantiles
  use exact nearest-rank statistics over concatenated samples, **not averaged
  percentiles**. Tail outliers are retained. QPS is median instrumented wall
  throughput including the harness, not reciprocal mean latency or batch QPS.

JSON is checkpointed atomically after each round. An interrupted run remains
running/failed rather than becoming a partial success. Reports can be regenerated
without running searches using `--report-only` with the same config/output.
Tiny smoke-test numbers verify plumbing; they are not performance evidence.

## Tests

```bash
cd benchmark
python -m unittest latency_baselines.test_protocol -v
python -m latency_baselines.smoke \
  --native-dir ../../experiment/YYYY-MM-DD/baseline-native \
  --diskann-sweep target/release/diskann_sweep \
  --output ../../experiment/YYYY-MM-DD/baseline-smoke
```

The smoke test explicitly builds generated 512-vector fixtures, runs all ten
adapters at 1/8 callers for three rounds, then repeats with **cache-only reload**.
It can also run `--metric cosine` or `--metric inner-product` (DiskANN must report
unsupported raw IP). Unit tests cover statistics, malformed files, recall,
missing rows, incomplete trials, and initialization/warmup/search failures without
barrier deadlocks. Baseline packages are not runtime dependencies of Orion.

The standalone ParlayANN parity test can be built against the same checkout:

```bash
c++ -std=c++20 -O2 -pthread -I../../ParlayANN -I../../ParlayANN/parlaylib/include \
  latency_baselines/native/test_parlay.cpp -o /tmp/parlay-latency-parity
PARLAY_NUM_THREADS=8 /tmp/parlay-latency-parity \
  /path/to/smoke/base.fbin /path/to/smoke/pa.graph /path/to/smoke/query.fbin
```

## USearch GIL control

The independent [USearch native control](usearch_control/README.md) compares the
stock Python API against identical native searches with and without the GIL.
Its deferred runner waits for the preceding experiment to finish.

## Combine finished sweeps and the USearch control

The plotting command reads frozen results; it never builds an index or starts a
search. It imports the existing `visualizations/chart_style.py` and produces
PNG/PDF P50/P95/P99 panels for 1 and 8 clients, a threshold
summary, and a portable `request-latency.json` for the homepage.

```bash
python -m benchmark.latency_baselines.plot \
  --inputs ../experiment/2026-11-15/baseline-latency-summary/094114/inputs \
  --usearch-control ../experiment/2026-11-15/usearch-native-control/analysis.json \
  --output ../experiment/2026-11-15/latency-publication
python -m unittest benchmark.latency_baselines.test_plot
```

Requires Matplotlib in addition to the harness dependencies. Optionally pass
`--engines orion,parlayann,diskann,hnswlib,usearch-native` for less dense panels.
The default shows Orion, ParlayANN, DiskANN, hnswlib and native USearch.
Use `--engines` to include other engines; all measurements stay in the JSON,
regardless of which series are plotted. Missing series are not fabricated.
The control input is the verified analysis produced after checking raw sample
hashes, pooled percentiles and serial/concurrent result parity. The plotter
checks recorded sampling/ID status and shared main-sweep input identities, but
does not re-read the raw latency sample files.

A threshold comparison selects the lowest **pooled P50** among qualified points
at Recall@10 ≥ the target. P95/P99 use that same setting, even if another setting
has a lower tail. Actual recalls can differ. Ratios are baseline / Orion; geometric
means use only the explicitly listed common datasets. Undersampled points stay
in the JSON's `excluded` list and never enter curves or aggregates.

The native USearch measurements are separate control points, not replacements
for the complete Python sweep. They cover three datasets at one effort each,
use the installed **serial** distance kernel, and retain the Python/ctypes request
boundary. Comparing native-GIL with native-no-GIL isolates the GIL change more
closely than comparing the stock Python wrapper with the native wrapper.
These measurements are descriptive, not a paired significance test.

Only the curated JSON and selected figures belong on `pages`; raw samples,
local paths and experimental caches do not belong in the public site. Source
JSON SHA-256 values are retained for traceability without publishing local paths.
