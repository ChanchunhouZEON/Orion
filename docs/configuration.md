# Configuration

[README](../README.md) · [CLI](cli.md) · [Graph sources](graph-sources.md)

The CLI embeds [sweep.yaml](../benchmark/configs/sweep.yaml). `--config` loads a
replacement YAML file; rebuilding is necessary only when changing embedded
configuration. The optional `cli` feature of `orion-runtime` owns shared resolution used by
`orion` and `orion-sweep`; the latter remains a benchmark executable. The validated
plan and execution operations also live in `orion-runtime`. Python uses programmatic
`IndexOptions` rather than CLI flags, YAML presets or legacy environment overrides.

## Resolution

- Presets are optional. Without one, the name is `anonymous`; an unknown named
  preset is an error. Base/query paths and metric must then be supplied, with
  dimension inferred from the base header when omitted.
- CLI input paths override corresponding preset paths. An overridden path
  discards its preset count. Ground truth is optional; `--no-groundtruth`
  disables preset GT. `--prepare-only` needs no queries.
- Construction parameters use CLI → dataset overrides → global defaults.
- Cascade selection uses dimension/metric defaults → dataset axes → CLI axes.
  Global legacy cascade fields are not the anonymous-data selector.
- Graph policy uses CLI/legacy environment → YAML → auto. See
  [the complete precedence and fallback rules](graph-sources.md).
- k selects its YAML L schedule; `--search-list-sizes` supplies an explicit
  schedule. Unknown k requires a configured schedule or that override.
  `orion --search-l` selects one beam width and consumes queries once.

`SearchPlan` validates storage, input encodings, metric and cascade together.
Serialized `vector_storage` and `cascade` are derived from it. Native u8 requires
L2, byte base/query input, and `none/native-l2-u8/none` with PA. Dimensions
are inferred at runtime, with internal row padding for the selected SIMD kernel.
It does not silently quantize f32 input or fall back to a Rust build.

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
