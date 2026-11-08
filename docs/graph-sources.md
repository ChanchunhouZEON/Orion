<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Graph sources and cache provenance

[README](../README.md) · [Configuration](configuration.md)

Graph source describes graph preparation. Both PA and Rust graphs are searched
by Orion after conversion to its `PhasedGraph` layout. Selecting `rust` does not
run the `diskann_sweep` search engine.

## Selection policy

`--graph-source auto` is the CLI default, matching Rust/Python `IndexOptions`.
Resolution selects one concrete source before preflight or payload loading:

1. Validate an existing PA cache for the current inputs and construction recipe.
2. Otherwise use an available PA STAG v3 export.
3. Otherwise build in memory with bundled ParlayANN, for f32 or native u8.
4. When the default `parlay` feature is disabled, f32 may use a Rust cache/build;
   native u8 reports that the builder or a PA cache/export is required.

Only absence permits fallback. An incomplete cache pair, incompatible manifest,
malformed existing STAG header, or permission error is an error. Payload checks
remain part of actual loading/import; a header probe is not a full integrity scan.

An explicit staged path selects PA strictly, including alongside `auto`.
`--graph-source rust --staged-file ...` is rejected. Production interfaces do not
read legacy environment variables or YAML. Only the benchmark supports source
precedence from experiment flags, `ORION_GRAPH`/`ORION_STAGED_FILE`, and YAML.


Explicit PA loads/imports/builds with PA; it never substitutes the Rust builder. Explicit
Rust ignores PA exports. Native u8 rejects explicit Rust before loading data.
For reproducible experiments, pin a concrete source even when auto would pick it.

## Paths and offline planning

Default cache roots are `cache/orion_parlayann` and `cache/orion`, chosen by the
**effective source**. A custom `--cache-dir` is shared as a location, but PA and
Rust still have separate namespace keys. With the default backend, auto uses the PA root even if a Rust cache exists.
Rust fallback is available only without the `parlay` feature.

Preset STAG paths expand `${PA_ROOT}` (default `../ParlayANN`) and
`${max_extra}`. The historical `_pct60.staged` suffix is accepted for default
paths; explicit paths are used exactly. STAG must match base order/count and
degree. Export capacity determines the imported slab stride.

`--print-config` reports `graph_source_policy`, concrete `graph_source`,
`graph_selection_reason`, namespace, and directory. It may inspect headers but
never loads orion-vector/graph payloads. `--preflight` uses the same resolved source
and reports `graph_action`, `cache_path`, and memory components. When inputs are
absent, an offline estimate can still succeed, including a PA estimate labeled
`build ParlayANN graph`; this is not an executable-input validation success.

## Identity and provenance

Graph lookup uses a SHA-256 key over the input identity, indexed row count
and resolved construction recipe. An available file base is identified by its
complete contents. Offline preflight without the base estimates resources from
explicit shape metadata (or benchmark presets). Query and timing settings do not change the graph key. The working
cache records the graph and metadata digests and checks them on load. An
explicit STAG export is checked against its recorded content when available.

Application snapshots store the graph, metadata, provenance and materialized
sidecars under their individual content hashes. Their entry point is
`index.json`; for shared storage, verify it with an independently held digest.
Earlier graph caches and snapshots must be rebuilt.

## Builder differences

The bundled PA backend borrows padded rows, builds with L2 (or f32 inner product),
and constructs adjacency/extras directly in the final Orion slab. After workers
join, partitions are finalized in place using bounded per-worker node scratch.
Cosine rows are normalized before L2 construction. Preflight counts the slab once;
construction statistics, insertion batches and scratch remain additional unknown memory.

PA can also import topology from the STAG-capable
[ParlayANN fork](https://github.com/ChanchunhouZEON/ParlayANN-staged). Rust builds
via `build_diskann_index`, using the configured construction parameters and
collecting extra candidates. The current Rust builder uses L2 even when search
uses inner product. Normalization can make rankings equivalent, but raw MIPS
vectors do not have that property.

Matching R, construction L and alpha is insufficient to make the two builders'
graphs identical. Pruning, passes, seeds, quantization, historical cache versions
and construction objective can all matter. See the
[2026-10-29 comparison](benchmarks.md#same-engine-different-graph-sources).
