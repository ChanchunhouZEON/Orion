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
    dataset_name: Some("my-vectors".into()),
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

`dataset_name` is display metadata: Rust uses `IndexOptions::dataset_name`,
Python accepts `Index.from_file(..., dataset_name="my-vectors")` (also arrays),
and the CLI accepts `--dataset-name my-vectors`. The default is `anonymous`.
Names persist in snapshots; `OpenOptions::dataset_name`, Python `open` and CLI
`--snapshot --dataset-name` can override them. Names do not select presets or
change graph/cache/calibration identities. Empty, whitespace-only and control-character
names are rejected before loading the base payload.

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
otherwise prioritizes graph and admission storage, followed by prefilter storage,
then rerank-only or inactive payloads. Objects that do not fit are mapped; a
smaller lower-priority object may still use the remaining capacity. `Resident` disables this
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
        ..Default::default()
    },
    ..Default::default()
})?;
// Calibration prepares sidecars using the plan; ordinary search reuses them.
# Ok::<(), orion_ann::Error>(())
```

Per-object overrides take precedence over the global mode. Resident overrides
reserve their budget before automatic objects, regardless of allocation order.
A resident request fails with `ResourceExhausted` if it cannot fit; it is never
silently mapped. `Auto` remains an explicit option even under a global resident
or mapped policy. Overrides do not build representations absent from the request.

```rust
use orion_ann::{StorageOptions, StorageMode, StorageObject};
let storage = StorageOptions {
    mode: StorageMode::Auto,
    objects: [
        (StorageObject::Base, StorageMode::Mapped),
        (StorageObject::Graph, StorageMode::Resident),
        (StorageObject::L2U8, StorageMode::Resident),
    ].into(),
    ..Default::default()
};
assert_eq!(storage.objects[&StorageObject::Base], StorageMode::Mapped);
```

```python
storage = StorageOptions(objects={
    "base": "mapped", "graph": "resident", "l2-u8": "resident",
})
index = Index.from_file("base.fbin", storage_options=storage)
```

```bash
orion --base base.fbin --query query.fbin --preflight \
  --object-storage base=mapped \
  --object-storage graph=resident \
  --object-storage l2-u8=resident
```

Supported keys are `base`, `graph`, `norms`, `labels`, `l2-u8`, `l2-u16`,
`mips-i8`, `mips-i16`, `pq`, `l2-kt`, `jl-sparse`, `jl-sparse-mips`,
`jl-hadamard`, `rabitq`, and `rabitq4`. These identify representations, not
cascade stages or cache filenames. Shared sidecars receive one placement decision; shared backing allocations are
charged only once. `pq` selects the whole sidecar, including codebook and codes;
suballocation placement is not part of this configuration yet.

Labels remain resident-only: global mapped leaves label metadata in memory,
while explicit `labels=mapped` is rejected, even before any labels exist.
Label generations and external ID allocations are not part of the core-payload
accounting list; metadata accounting remains separate from region payload accounting.
Preflight lists `placement_objects` and `inactive_placement_overrides` (including
objects outside the selected phase/payload scope). The table shows the same
decisions when overrides are present. `storage_accounting.objects` in index info
reports actual prepared payloads, so it does not include unbuilt sidecars.

Preflight and preparation share the object descriptors and selection algorithm.
Offline planning uses the supplied budget; live preparation additionally probes
available RAM and preserves the backing of objects already created. This can
change automatic choices when resource observations differ. Build-only reports
exclude search sidecars; live index placement reserves room for later calibration.
Snapshot open inherits saved storage settings unless `storage_options` is supplied;
a supplied policy replaces the entire saved policy, consistently across interfaces.
Overrides do not alter graph/sidecar hashes or calibration signatures.

The directory must already exist. It should be backed by disk, not tmpfs.
Private files require the mapped payload's disk space for the index lifetime;
failures remove partial files and return structured I/O errors. Mapping reduces
owned allocations, not the logical payload or a guaranteed RSS ceiling: touched
file pages consume page cache. Mapped payloads are accounted separately and do
not consume the managed resident allocation budget. There is no fixed mmap
working-set allowance; the OS controls mapped-page residency and reclamation.
The planner rejects a budget that cannot fit fixed resident allocations and the
remaining resident payloads. Query scratch, copy buffers, PQ training samples
and other process overhead can still consume RAM; preflight reports known
working buffers separately rather than treating mapped storage as free RAM.
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
    dataset_name: Some("my-vectors".into()),
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


### Runtime ownership and budget accounting

Preparation is internal to `Index` construction, snapshot restoration and
calibration. Configure storage and pinning through the public options and
residency methods; applications do not call `prepare` directly. Only the opt-in
`benchmark` feature exposes raw-index preparation for controlled experiments.

`Index::info()["storage_accounting"]` reports managed resident payload bytes,
mapped payload bytes, the generation's resident budget limit, and remaining
unreserved resident capacity. These are **not RSS readings**.
The budget is per index generation; multiple indexes do not share a process-wide
cap. They still observe system headroom when their placement is prepared.

Storage policy is installed during preparation, independently of calibration.
Each managed base, graph, norm or sidecar buffer reserves its actual payload bytes
before allocation. Its lease follows ownership through construction and freezing;
shared views retain one lease until the last reference drops. An object containing
several buffers has several leases under the same object account. Construction
errors and unwinding release only allocations that are actually discarded; already
published sidecars stay charged. No post-hoc inventory or reconciliation is needed.
PQ codes replacement charges both old and new buffers during the copy and releases
the old charge only after replacement succeeds. Recalibration does not charge
existing buffers twice.

Base and graph still use header-based preflight before allocation. Array inputs
are copied before index options are available: preparation adopts that owned copy
once if resident, or allocates its accounted mapping and then drops the input copy.
The budget limit is captured once per generation, before managed allocation;
calibration neither reprobes it nor changes the installed placement policy.
Preflight remains a forecast and input check; it acquires no live reservations.
Accounting excludes builder/query scratch, temporary PQ training samples, label/ID
metadata, allocator overhead, application input and OS page faults. Mapped byte
counts describe backing storage, not its current RAM footprint.

Frequency is currently a preparation-time hint derived from the validated search
recipe, not a measured per-vector access count. There is no background migration,
cache eviction or asynchronous SSD I/O in this policy. Storage and search results,
cache identities, SIMD dispatch and prefetch scheduling remain independent.


Preparation dependencies are selected separately from the search recipe. A
`GraphOnly` run (`prepare_only`) omits quantized search sidecars; `SearchReady`
includes those needed by calibration and search. Preflight passes the same
phase-selected sidecar list to placement that it uses for its memory estimate.
Native cosine norms remain base auxiliary storage in either phase. Actual base
and graph placement continues to allow for eventual search dependencies, even
in a graph-only run; choosing this target does not mutate the run configuration.


### Preflight phases and memory limits

Partial `StorageOptions.objects` entries are overrides, not a complete allocation
plan. Planning collects the phase requirements and deduplicates shared objects,
reserves explicit resident choices and fixed buffers, then considers automatic
objects in priority order: graph, admission, prefilter, rerank, and cold objects.
Mapped overrides do not consume the resident allowance. Base/norm priority follows
the stage that reads them. Input order does not change those choices.

A bound core plan is different: its objects may already exist. Reusing it validates
required-object coverage, fixed placement and the budget. An incomplete binding is
rejected rather than silently allocating missing objects with the default policy.
It is not extended or migrated during calibration. Runtime prepares the complete
search policy before binding, including saved snapshot extras.

```bash
orion --base /data/sift/base.u8bin --base-format u8bin \
  --dimension 128 --max-points 1000000 --dataset-name sift1m \
  --preflight --preflight-format table \
  --memory-budget-gib 0.4 --storage-mode auto \
  --object-storage graph=resident --object-storage base=mapped \
  --mmap-dir /data/orion-mappings
```

`--memory-budget-gib` must be finite and positive. Offline preflight uses the
supplied budget, without substituting the local machine's available RAM. Live Auto
placement additionally considers available memory and keeps a fixed allocation
budget for that generation. The report lists both the resident lower bound and
mapped payload bytes. No separate mmap byte-limit option exists yet; `--mmap-dir`
chooses backing storage and does not declare its capacity. Backing free space and
user/project quotas are not checked by preflight. Mapping/creation failures still
return errors during preparation.

A passing report only establishes that counted resident payloads fit. It cannot
prove peak RSS fits: scratch and overhead remain additional, and mapped pages may
occupy RAM. A future mmap cap should constrain mapped backing bytes separately
from resident payload, not count the whole mapping as resident memory.

Add `--prepare-only` for a graph-only report; it excludes search sidecars and
query buffers. Omit it to check search-ready requirements before calibration.
These reports describe different phases and must not be treated as interchangeable.

## Explicit YAML index configuration

`IndexOptions::from_yaml` decodes the same generic configuration schema as Rust
and JSON. This describes the index's construction and resource policy, not an
entire run or a fixed search request. It does not load presets or search for a
default configuration file.

| Index configuration (YAML / `IndexOptions`) | Per-request input (separate API / CLI arguments) |
| --- | --- |
| Metric, base storage format, graph and cascade | Query vectors, `k`, search list size |
| Threads, memory budget, object placement and auto pin | Scores and filter selection |
| Dataset display name, cache path and row limit | Input paths, offline dimension, output and execution mode |

Searches reuse the index with their own `QueryBatch` and `SearchOptions`. Keeping
these requests separate allows different callers to choose different `k`, search
list sizes and result fields without modifying the index configuration. A future
whole-task format can compose these existing types without expanding
`IndexOptions` with CLI-specific run controls.

Use `{}` to explicitly select all defaults; empty/null documents and sequences
are rejected. Missing fields retain their defaults; unknown fields, duplicate keys, invalid
field types and multiple documents are rejected. Semantic and shape validation
run through the existing resolution path after caller overrides.

```rust
use orion_ann::IndexOptions;
let options = IndexOptions::from_yaml(r#"
threads: 8
memory_budget_gib: 1.0
storage_options:
  mode: auto
  objects:
    graph: resident
"#)?;
assert_eq!(options.threads, 8);
# Ok::<(), orion_ann::Error>(())
```

The CLI accepts `--config FILE`. Explicit CLI flags override YAML fields, which
in turn override defaults. Nested fields merge independently: overriding
`--graph-degree` preserves YAML `graph.build_l`, and `--object-storage base=mapped`
preserves YAML choices for other objects. `--no-auto-pin` disables locking;
when absent, the YAML value is retained. Relative paths use the process working
directory, not the YAML directory. Environment expansion and includes are disabled.
The file is read once for resolution and execution. `--snapshot` cannot be combined
with `--config`; snapshot open uses its persisted build configuration.

```bash
orion --config orion-runtime/examples/placement.yaml \
  --base /data/sift/base.u8bin --dimension 128 \
  --preflight --preflight-format table

# CLI overrides only the graph placement and the budget from the YAML file.
orion --config orion-runtime/examples/placement.yaml \
  --base /data/sift/base.u8bin --dimension 128 \
  --memory-budget-gib 0.1 --object-storage graph=auto \
  --preflight --preflight-format table
```

Input/query paths, offline dimension and query-run controls remain CLI arguments,
not fields of `IndexOptions`. The example file's dataset name is only a display
label; it does not infer dimensions, counts or a graph recipe.
