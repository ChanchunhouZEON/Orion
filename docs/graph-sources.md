# Graph sources and cache provenance

[README](../README.md) · [Configuration](configuration.md)

Graph source describes graph preparation. Both PA and Rust graphs are searched
by Orion after conversion to its `PhasedGraph` layout. Selecting `rust` does not
run the `diskann_sweep` search engine.

## Selection policy

`--graph-source auto` is the default when neither CLI, legacy environment nor
YAML specifies a source. Bundled presets explicitly use `source: auto`.
Resolution selects one concrete source before preflight or payload loading:

1. Validate an existing PA cache for the current inputs and construction recipe.
2. Otherwise use an available PA STAG v3 export.
3. Otherwise, for f32 storage, use a Rust cache or build in-process.
4. For native u8, retain PA as required and report the missing input. There is
   no silent widening or native-u8 Rust construction path.

Only absence permits fallback. An incomplete cache pair, incompatible manifest,
malformed existing STAG header, or permission error is an error. Payload checks
remain part of actual loading/import; a header probe is not a full integrity scan.

Selection precedence is CLI source/CLI staged path, then `ORION_GRAPH`/
`ORION_STAGED_FILE`, then YAML, then auto. `ORION_GRAPH=pa` aliases `parlayann`.
An explicit staged path selects PA strictly, including alongside `auto`.
`--graph-source rust --staged-file ...` is rejected. CLI Rust overrides legacy
PA settings. A YAML `source: parlayann` or `source: rust` is strict as well;
`--graph-source auto` can override that policy.

Explicit PA requires a cache or an export; it never switches builders. Explicit
Rust ignores PA exports. Native u8 rejects explicit Rust before loading data.
For reproducible experiments, pin a concrete source even when auto would pick it.

## Paths and offline planning

Default cache roots are `cache/orion_parlayann` and `cache/orion`, chosen by the
**effective source**. A custom `--cache-dir` is shared as a location, but PA and
Rust still have separate namespace keys. Auto probes the PA root before the
Rust root; a PA export has priority over an existing Rust cache.

Preset STAG paths expand `${PA_ROOT}` (default `../ParlayANN`) and
`${max_extra}`. The historical `_pct60.staged` suffix is accepted for default
paths; explicit paths are used exactly. STAG must match base order/count and
degree. Export capacity determines the imported slab stride.

`--print-config` reports `graph_source_policy`, concrete `graph_source`,
`graph_selection_reason`, namespace, and directory. It may inspect headers but
never loads vector/graph payloads. `--preflight` uses the same resolved source
and reports `graph_action`, `cache_path`, and memory components. When inputs are
absent, an offline estimate can still succeed, including a PA estimate labeled
`PA input required`; this is not an executable-input validation success.

## Identity and provenance

An input-keyed `v3_*` namespace includes base file identity, metric, dimension,
graph construction settings, and **concrete graph source**. Native-u8 storage
has a separate discriminator. Query paths and the requested auto/explicit policy
do not change graph identity: automatic and explicit PA can reuse the same cache.

A graph cache consists of metadata `.bin`, slab `.pgraph`, and `.cache.json`
manifest. New manifests record `graph_source` and, for imports, the export's
path, size and modification time. These identify inputs rather than provide
cryptographic content verification. Existing version-1 manifests without the
new source field remain readable; their namespace already separates sources.
A conflicting source field is rejected.

Preset-compatible legacy names are considered only when the input-keyed cache
is absent. Matching requires the original dataset path/format and construction
settings, not merely the dataset name. Header validation cannot recover a legacy
export's historical provenance. Explicit exports cannot reuse an unproven legacy
cache. An explicit export conflicting with known provenance fails; select another
`--cache-dir` to import it separately. A cached export can have been deleted,
provided the requested path still matches recorded provenance.

## Builder differences

PA imports topology from the STAG-capable
[ParlayANN fork](https://github.com/ChanchunhouZEON/ParlayANN-staged). Rust builds
via `build_diskann_index`, using the configured construction parameters and
collecting extra candidates. The current Rust builder uses L2 even when search
uses inner product. Normalization can make rankings equivalent, but raw MIPS
vectors do not have that property.

Matching R, construction L and alpha is insufficient to make the two builders'
graphs identical. Pruning, passes, seeds, quantization, historical cache versions
and construction objective can all matter. See the
[2026-10-29 comparison](benchmarks.md#same-engine-different-graph-sources).
