<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Configure an index

Configure an embedded index with [`crate::IndexOptions`]. These options do not
read CLI presets or environment overrides. Validation happens when an index is
created/opened, not when a struct is constructed or deserialized.

```rust
use orion_ann::{GraphConfig, IndexOptions};
use orion_ann::GraphSourcePolicy;
let options = IndexOptions {
    threads: 8,
    graph: GraphConfig {
        source: GraphSourcePolicy::Rust,
        degree: 32, build_l: 100, alpha: 1.2, max_extra: 16,
        ..Default::default()
    },
    memory_budget_gib: Some(64.0),
    cache_dir: "cache/service".into(),
    ..Default::default()
};
assert_eq!(options.graph.degree, 32);
// JSON has the same field names as Python's configuration object.
let restored: IndexOptions = serde_json::from_str(&serde_json::to_string(&options)?)?;
assert_eq!(restored.threads, 8);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Selection and validation

- `Auto` reuses a compatible PA cache/staged export, then builds in memory with
  ParlayANN. The default `parlay` feature supports f32 and native u8; disabling it
  retains the f32 Rust fallback. Explicit `Rust`/`Parlayann` choices remain strict.
  Native u8 never silently becomes an f32 build. ParlayANN currently uses two
  unquantized construction passes and records the recipe in cache provenance.
- Supplying `staged_file` selects PA; combining it with `Rust` is an error.
- Dimensions and row counts come from input. Row padding is internal and does
  not change the logical dimension exposed by the API.
- `storage=None` follows input dtype. Native u8 supports L2, inner product and
  cosine without expanding the base. Defaults select `native-l2-u8`, `native-ip-u8`
  or `native-cosine-u8`, with no prefilter/rerank. Cosine rejects zero vectors and
  keeps squared norms: 4 bytes/vector through dimension 66,051; 8 bytes above it.
  Norms participate in memory planning, mmap and SHA-256 snapshot verification.
  Explicit cascade strings use the CLI vocabulary.
- `threads` must be positive and at most the compiled worker limit; each index
  owns its pool. `window_size` is in `1..=64`. `degree > 0`, `build_l >= degree`,
  and finite `alpha >= 1` are required. `max_points=0` is invalid.
- A memory budget is a finite, positive GiB estimate checked before loading.
  It is not an OS allocation limit or a guarantee against process OOM.

## Errors and ownership

Invalid settings produce the shared structured error. `InvalidArgument` covers
invalid bounds; `UnsupportedConfiguration` covers unsupported execution recipes;
`ResourceExhausted` describes an exceeded estimated budget. Match `ErrorCode`,
not message text. Options are owned/cloned by the session; editing the caller's
copy cannot change an existing index. No unsafe block is needed for this API.

## Automatic RAM residency

Rust, Python and the CLI apply the same best-effort policy after calibration,
when opening a calibrated snapshot, and after changing PQ code placement.
The planner observes host headroom, visible Linux cgroup limits and the process
lock allowance. It leaves at least 256 MiB and 10% of deployment memory available,
then bounds additional locks by that headroom, the remaining lock allowance and
`memory_budget_gib` when supplied. Unknown headroom or lock allowance disables automatic locks;
OS refusals leave storage pageable and searches usable. This is not a reservation
or an allocator cap, and it does not choose a different cascade.

The plan covers required base/sidecar buffers and the graph slab, attempting
smaller allocations first. It never builds a sidecar just to pin it. Mapped payloads remain pageable. Owned PQ codebooks retain their separate
bounded best-effort policy. RAM locks do not guarantee CPU-cache residency or THP.

`IndexOptions { auto_pin: false, ..Default::default() }` disables automatic search
locks. Python accepts `auto_pin=False` in constructors, and the CLI accepts
`--no-auto-pin`. `OpenOptions::auto_pin` and Python `Index.open(auto_pin=...)`
override the saved setting; resource observations are always refreshed on open.
Manual `Index::pin` remains available for an explicit selection/required policy.

Inspect `index.info()["residency"]` (Python: `index.info["residency"]`) or the CLI
summary's `residency` field for the detected budget, actual locked bytes and
skipped requests. Planning and lock calls happen outside search timing. The
owning index releases automatic locks before replacing storage or being dropped;
recalibration replaces the plan without changing immutable cache object identities.

## Automatic RAM residency

`IndexOptions::auto_pin` defaults to `true`. Search preparation first materializes
the selected datasets, then attempts bounded RAM locks using detected available
memory, the OS lock allowance and the configured budget. `Index::calibrate` and
opening a calibrated snapshot invoke this preparation automatically; no CLI
feature or explicit `pin` call is needed. An uncalibrated index defers it until
calibration. Ordinary `search(&self, ...)` calls reuse the retained locks.

Set `auto_pin: false` to prepare the same datasets without automatic locks.
OS lock refusal falls back to pageable memory. The index owns the lock lifetime:
recalibration or PQ storage replacement releases old locks before preparing the
new generation, and dropping the index releases its remaining locks. Inspect
`Index::info()["residency"]` to see the effective policy and outcome.

## Storage placement before allocation

`IndexOptions::storage_options` controls physical storage independently of the
cascade. `Auto` (default) keeps payloads owned when budget/headroom permits,
otherwise maps the largest arrays before allocation. `Resident` disables this
fallback; `Mapped` maps base vectors, graph adjacency and extras, and all
quantized payloads (including norms, projection parameters and PQ codebooks).
File loading, normalization, STAG import and ParlayANN slab construction write
directly into the selected backing. Array/NumPy inputs retain their input copy
contract; selecting mapped storage relocates that existing copy during preparation.
Automatic pinning skips mapped payloads; an explicit manual pin can still lock them.
PQ codebooks keep their independent best-effort pin policy only when resident.

```no_run
use orion_ann::{BaseInput, Index, IndexOptions, StorageOptions, StorageMode};
let mut index = Index::new(BaseInput::from_file("base.fbin"), IndexOptions {
    memory_budget_gib: Some(16.0),
    storage_options: StorageOptions {
        mode: StorageMode::Auto,
        directory: Some("/local-ssd/orion-tmp".into()),
    },
    ..Default::default()
})?;
// Calibration prepares sidecars using the plan; ordinary search reuses them.
# Ok::<(), orion_ann::Error>(())
```

The directory must already exist. It should be backed by disk, not tmpfs.
Private files require the mapped payload's disk space for the index lifetime;
failures remove partial files and return structured I/O errors. Mapping reduces
owned allocations, not the logical payload or a guaranteed RSS ceiling: touched
file pages consume page cache. A 64 MiB reserve covers bounded copy blocks and
some page-cache working space; growing query scratch, PQ training samples and
other process overhead are additional. The planner rejects a budget that cannot
fit the fixed base/graph plus remaining owned payloads and this reserve.
Offline preflight uses the supplied budget, so it can describe a different
machine. Live preparation also considers currently available memory. Placement
is fixed before the first sidecar allocation and reused during recalibration.

Decoding verifies the original stream's SHA-256 and validates layout before
publishing a sidecar. Packed rows are padded while streaming to the private file;
no full heap copy is required. Changing or removing the original cache after open
cannot invalidate that mapping. Existing cache encodings and object digests do
not change. Snapshot open can override placement using
`OpenOptions { storage_options: Some(...), ..Default::default() }` and uses the
same planner before restoring sidecars. Snapshot metadata may retain deployment
settings, but placement does not affect execution signatures or calibration.

Python accepts `storage_options=StorageOptions(mode="auto", directory=...)`
in constructors and `Index.open`. CLI equivalents are `--storage-mode auto`
and `--mmap-dir /local-ssd/orion-tmp`. Inspect `Index::info()` (Python: `index.info`)
for `mapped_base_bytes`, `mapped_graph_bytes` and `mapped_sidecar_bytes`;
this is not an RSS measurement. Explicit mapped storage is unsupported on WASM.

The placement budget is not an RSS cap. Mapped pages still occupy physical RAM
while accessed. For controlled Linux measurements, use a cgroup v2 memory limit
that includes anonymous memory and page cache, disable swap, and record page
faults and memory events alongside QPS. Do not use an address-space limit:
virtual mappings must remain large enough to cover the index. Construction and
query scratch remain ordinary bounded or workload-dependent allocations;
this storage policy does not turn the graph builder into an external-memory algorithm.

## Cascade names describe the search objective

`metric` selects the objective and input preparation. Cascade arms make that
choice explicit; sharing a SIMD kernel does not make the objectives interchangeable.

| Metric | Scalar admission | Exact rerank | Preparation and computation |
|---|---|---|---|
| `l2` | `l2-u8`, `l2-u16`, `l2-kt` | `l2-f32` | Coordinates unchanged; squared L2 |
| `inner-product` | `ip-i8`, `ip-i16` | `ip-f32` | Coordinates unchanged; negative IP internally |
| `cosine` | `cosine-i8`, `cosine-i16` | `cosine-f32` | Normalize base/query rows; negative IP internally |

The f32 cosine default is `none → cosine-i8 → cosine-f32`. Its scalar codes
share the IP representation and runtime-dispatched SIMD streams. Quantization
can change vector norms; these existing arms **do not** add a quantized-norm
correction. Final requested scores use cosine similarity, higher is better,
not the integer admission score. Exact scores of returned candidates do not
imply exact global top-k. Zero-norm cosine inputs are rejected.

PQ uses `pq-l2`, `pq-ip`, or `pq-cosine`, with the corresponding exact rerank.
`none` disables reranking without changing the metric of explicitly requested
result scores. `jl` automatically selects the existing objective-compatible
adapter; unsupported prefilter combinations fail validation.

```rust
use orion_ann::{CascadeConfig, IndexOptions, SearchMetric};
let options = IndexOptions {
    metric: SearchMetric::Cosine,
    cascade: CascadeConfig {
        admission: Some("cosine-i8".into()),
        rerank: Some("cosine-f32".into()),
        ..Default::default()
    },
    ..Default::default()
};
```

Rust, Python `CascadeConfig`, and CLI `--admission`/`--rerank` accept the same
names. `mips-i8`/`mips-i16` remain aliases of `ip-i8`/`ip-i16` for inner product;
`f32`/`u16` remain L2 aliases. A cosine configuration must use cosine arms,
not IP aliases. Mismatched explicit choices return a configuration error before
building. Preflight and `Index::info()` (Python `index.info`) report preparation,
admission kernel and score direction.


### Byte storage and cascades

Native `u8` is a base representation, not a separate search engine. Its default
uses exact byte admission without a quantized copy or rerank. An explicit
`CascadeConfig` can instead select JL filtering, PQ or scalar admission, and
exact reranking under the same metric and precision-order checks as f32 input.
The historical `l2-f32`, `ip-f32` and `cosine-f32` rerank names select exact base
reranking; with byte storage they read the original byte rows.

Sidecars convert logical rows into bounded reusable f32 scratch during fitting;
they do not require a second f32 base dataset. Cosine fits from unit-normalized
row views while preserving the original bytes and exact squared norms. Query
preparation applies the same transformation. Cache identity includes the input
representation and metric; snapshot restoration restores this context before
loading sidecars. Requested result scores always refer to the original metric,
not the quantized admission score.

### Squared L2 naming

Use `SearchMetric::SquaredL2` in Rust and `squared-l2` in CLI/Python configuration.
The input aliases `l2` and `squared_l2` resolve to the same objective. Returned
scores are labeled `squared_l2`: they are sums of squared coordinate differences,
without a square root, and smaller values are better. Renaming the public option
does not change persisted graph/sidecar identities or snapshot signatures.
