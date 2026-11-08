<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

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
- `storage=None` follows input dtype. Native u8 supports exact L2 with the
  `none / native-l2-u8 / none` cascade. Other metric/storage/cascade combinations
  are checked together. Explicit cascade strings use the CLI vocabulary.
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
smaller allocations first. It never builds a sidecar just to pin it. Explicit
on-demand PQ codes remain pageable; the small PQ codebook retains its separate
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
