<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="../orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Benchmark configuration

[README](../README.md) · [CLI](cli.md) · [Graph sources](graph-sources.md)

The unpublished benchmark embeds [sweep.yaml](../benchmark/configs/sweep.yaml).
`orion-sweep --config` loads a replacement YAML file; rebuild only when changing
its embedded defaults. Dataset shortcuts, path presets and legacy experiment
environment variables are implemented in `benchmark/src/config`.

The production CLI, Rust and Python use shared `IndexOptions` defaults instead.
They never load YAML or resolve dataset shortcuts. The runtime owns validated
recipes, storage compatibility and execution; each frontend owns its input syntax.
See the [CLI guide](cli.md) for explicit file inputs and offline preflight.

## Resolution

- Presets are optional. Without one, the name is `anonymous`; an unknown named
  preset is an error. Base/query paths and metric must then be supplied, with
  dimension inferred from the base header when omitted.
- Benchmark input paths override corresponding preset paths. An overridden path
  discards its preset count. Ground truth is optional; `--no-groundtruth`
  disables preset GT. `--prepare-only` needs no queries.
- Construction parameters use CLI → dataset overrides → global defaults.
- Cascade selection uses dimension/metric defaults → dataset axes → CLI axes.
  Global legacy cascade fields are not the anonymous-data selector.
- Graph policy uses Benchmark flags/legacy environment → YAML → auto. See
  [the complete precedence and fallback rules](graph-sources.md).
- k selects its YAML L schedule; `--search-list-sizes` supplies an explicit
  schedule. Unknown k requires a configured schedule or that override.
  The separate production `orion --search-l` selects one beam width and consumes queries once.

`SearchPlan` validates storage, input encodings, metric and cascade together.
Serialized `vector_storage` and `cascade` are derived from it. Native u8 requires
byte inputs and native admission matching L2, IP or cosine, with no prefilter or
rerank. It uses ParlayANN or the experimental supplier path. Dimensions
are inferred at runtime, with internal row padding for the selected SIMD kernel.
It does not silently quantize f32 input or fall back to a Rust build.

## Benchmark residency and trusted caches

`orion-sweep` uses the same bounded auto pin policy as Rust `Index`, Python and
CLI: available RAM, lock-page allowance and the optional memory budget bound
best-effort locks for the graph and required datasets. Shared `prepare_search`
materializes those requirements and applies `auto_pin` in one step, independently
of the `cli` feature. The residency guard releases
locks after the run. `--no-auto-pin` disables these locks; query buffers are not
locked separately. The other benchmark Orion runners use this policy too.

Benchmark loaders trust experiment-owned cache contents and skip graph/sidecar
SHA-256 verification. They still validate headers, dimensions, lengths and graph
edges. This is an explicit benchmark execution policy, not a consequence of Cargo
feature unification: production `Index::open`, snapshots and the CLI retain their
integrity checks even in a build that also enables `benchmark`.

Default content-addressed lookup still hashes the base to identify its cache.
To skip that scan too, select the exact existing record or legacy metadata file:

```bash
cargo run --release --manifest-path benchmark/Cargo.toml --bin orion-sweep -- \
  sift --graph-source parlayann --cache-file /data/cache/<digest>.cache.json

# Historical V5/V6 metadata and its adjacent .pgraph; no graph rebuild.
cargo run --release --manifest-path benchmark/Cargo.toml --bin orion-sweep -- \
  sift --graph-source parlayann --cache-file /data/cache/sift-old.bin
```

With explicit selection, the experiment owner must match the base, graph and
construction recipe. V5 contains no dataset identity/shape; its node count and
layout are checked, but equal-shaped datasets cannot be distinguished without
hashing. Unknown binary versions are rejected. Compatible legacy sidecars are
loaded directly; absent or incompatible sidecars are prepared by the existing
lazy dataset path. New benchmark sidecars remain flat, without production
verification receipts. Newly built content-addressed graphs still need hashes
for their object names. These changes shorten preparation, which remains outside
search QPS timing.

## Preset construction recipes

These are configured parameters, not a claim of identical topology between
builders. GloVe25 is padded to 32 dimensions; DEEP10M to 128 in this workspace.

| Preset | Dimension | R | Construction L | Alpha | Resident storage |
|---|---:|---:|---:|---:|---|
| sift | 128 | 64 | 128 | 1.15 | f32 |
| sift10m / sift100m / sift1b | 128 | 64 | 128 | 1.15 | u8 |
| glove25 | 32 | 100 | 200 | 1.0 | f32 |
| glove100 | 100 | 100 | 200 | 1.0 | f32 |
| gist | 960 | 100 | 200 | 1.1 | f32 |
| deep10m | 128 | 64 | 128 | 1.05 | f32 |
| fashion-mnist | 784 | 40 | 80 | 1.1 | f32 |
| msmarco_bert_1M | 768 | 64 | 128 | 1.0 | f32 |
| wiki_ada_1M | 1536 | 64 | 128 | 0.98 | f32 |

The WikiAda recipe is an adapted, experimentally selected recipe, not an
official ParlayANN ada-002 preset. MS-MARCO uses raw inner product. Search metric
is the dataset-level `metric`, independent of the admission representation.
The Rust graph-construction objective is currently L2.

Global `max_extra=16` and `window_size=5` can be overridden. Graph stride is
`round_up((4 + degree + extra) × 4, 64)` bytes: 384 bytes at R=64/extra=16,
512 bytes at R=100/extra=16. Imported capacity comes from STAG/cache headers.

## Calibration and experiment controls

Defaults use 200 calibration queries and a base calibration L of 48. The
resolved calibration beam also accounts for k; inspect `--print-config` for the
actual value. Each graph/k is calibrated independently. Calibration is outside
timed search and does not change the requested measurement L schedule.

Prefetch controls are initialized once per process. The values below distinguish
compiled defaults from the recent local experiment settings:

| Environment variable | Units | Compiled default | Recent experiment |
|---|---|---:|---:|
| `ORION_DSTREAM_LA_Q` | cache lines | 10 | 32 |
| `ORION_DSTREAM_LA_TRUTH` | cache lines | 6 | 32 |
| `ORION_DSTREAM_SINK_BURST` | maximum far requests per group | 12 | 32 |
| `ORION_DSTREAM_BURST_TRIGGER_LINES` | computed cache lines per group | 4 | 8 |
| `ORION_DSTREAM_BURST_AHEAD_VECTORS` | vectors | 4 | 4 |

`SINK_BURST=0` disables far prefetch. Ahead is measured in vectors, not cache
lines. Grouping accounts for logical compute lines using `CACHE_LINE_BYTES`;
near requests may repeat far requests. These are tunable policies, not fixed
hardware MSHR counts or guarantees that lines remain in L1. Larger values need
not improve performance. The recent f32 sweeps also set `ORION_RERANK_FACTOR=2`.

For cache keys, provenance and source selection, use
[Graph sources](graph-sources.md); for memory-budget limits, use
[Large datasets](large-datasets.md).

### Cosine dispatch

For f32 `metric=cosine`, the default cascade is
`none → cosine-i8 → cosine-f32`. Base and query rows are L2-normalized; the
implementation then reuses i8/f32 IP kernels. `cosine-i16` and `pq-cosine` are
explicit alternatives. Quantized-vector norms are not corrected in these arms.
For raw inner product, use `ip-i8`/`ip-i16`/`pq-ip` and `ip-f32` instead.
Stage/metric mismatches are rejected. The CLI preflight table explains this
preparation and dispatch; requested result scores use the original metric.
