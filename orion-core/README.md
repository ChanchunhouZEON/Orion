<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

<h1><img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Orion Astraeus · orion-ann-core</h1>

Adaptive graph search for approximate nearest neighbors.

Part of [Orion Astraeus](https://github.com/ChanchunhouZEON/Orion). Requires Rust 1.92 or newer. Licensed under MIT.

The runtime-dimensional `Orion<T>` supports native f32/u8 bases, padded SIMD
storage, adaptive cascades and graph cache I/O. For an owned service API with
file/array input, configuration, calibration and snapshots, use
[`orion-ann`](https://docs.rs/orion-ann).

```toml
[dependencies]
orion-ann-core = "0.1.0"
```

Build directly from row-major vectors, without runtime configuration or files:

```rust
use orion_ann_core::{BuildConfig, DynamicCascade, DynamicSearchConfig, Orion, ScoreMetric};
use orion_ann_core::search::{query::QueryInput, cascade::stage::{NoPrefilter, rerank::NoRerank}};
let base: Vec<f32> = (0..64 * 13).map(|i| (i % 251) as f32).collect();
let index = Orion::<f32>::build_from_slice(&base, 13, BuildConfig {
    degree: 8,
    build_l: 24,
    max_extra: 4,
    threads: 2,
    ..Default::default()
})?;
let exact = index.exact_stage(ScoreMetric::SquaredL2)?;
let recipe = DynamicCascade { prefilter: &NoPrefilter, admission: &exact, rerank: &NoRerank };
let result = index.search(QueryInput::F32(&base[..13]), DynamicSearchConfig {
    k: 1, search_list_size: 32, ..Default::default()
}, recipe, None)?;
assert_eq!(result.ids, vec![0]);
# Ok::<(), orion_ann_core::Error>(())
```

`build_from_slice` copies once into padded storage. `Orion::build(dataset,
config)` takes ownership of an existing `PaddedDataset` without copying it.
ParlayANN writes directly into the final graph slab, including extras. Builds
return an in-memory index. Use the runtime `Index` API for persisted snapshots. Adaptive
cascade calibration is a separate operation; exact admission with convergence disabled needs none.

## Native u8 and batch queries

Byte input stays byte input; no float base copy or dataset quantization is
required. Queries are packed logical rows, without SIMD padding.

```rust
# #[cfg(feature = "parlay")]
# {
use orion_ann_core::{BuildConfig, DynamicCascade, DynamicSearchConfig, Orion, ScoreMetric};
use orion_ann_core::search::{query::QueryInput, cascade::stage::{NoPrefilter, rerank::NoRerank}};
let base: Vec<u8> = (0..64 * 13).map(|i| (i % 251) as u8).collect();
let index = Orion::<u8>::build_from_slice(&base, 13, BuildConfig {
    degree: 8, build_l: 24, threads: 2, ..Default::default()
})?;
let exact = index.exact_stage(ScoreMetric::SquaredL2)?;
let recipe = DynamicCascade { prefilter: &NoPrefilter, admission: &exact, rerank: &NoRerank };
let neighbors = index.search_batch(QueryInput::U8(&base[..3 * 13]), DynamicSearchConfig {
    k: 3, search_list_size: 32, ..Default::default()
}, recipe, None)?;
assert_eq!(neighbors.len(), 3); // One result row per query, in input order.
assert!(neighbors.rows().all(|row| row.len() <= 3));
# }
# Ok::<(), orion_ann_core::Error>(())
```

`search` and `search_batch` with an exact admission stage use full-precision
distances, but traverse an
approximate graph: exact distance computation does not guarantee exact top-k.
Batch methods return the owned `SearchResult` shared with the runtime: packed
IDs, valid counts per query, and the requested k. Use `rows()` to skip padding.
`search` returns the same structure for a single row; `search_observed` returns
an ID vector and emits diagnostic events.
The core batch API uses the current Rayon pool. `BuildConfig::threads` controls
construction only; use runtime `Index` when you want an owned search worker pool.

## Select a construction backend

```rust
use orion_ann_core::{BuildBackend, BuildConfig, BuildMetric};
let config = BuildConfig {
    backend: BuildBackend::DiskAnn,
    metric: BuildMetric::SquaredL2,
    degree: 32,
    build_l: 64,
    alpha: 1.2,
    max_extra: 8,
    threads: 4,
    ..Default::default()
};
config.validate()?;
assert_eq!(config, config.clone());
# Ok::<(), orion_ann_core::Error>(())
```

`Auto` prefers ParlayANN when enabled. Explicit `ParlayAnn` or `DiskAnn` never
silently changes backend. Raw f32 inner-product construction uses
`BuildMetric::InnerProduct` with ParlayANN; cosine input must be normalized
before constructing its L2 graph. Configuration validation does not construct
an index; backend/input compatibility is checked by `build`.

## Transfer owned storage and inspect a graph

Add `orion-ann-vector = "0.1.0"` when selecting a padded layout by its kernel.
Moving a `PaddedDataset` into `build` transfers its allocation without copying.

```rust
use orion_ann_core::{BuildConfig, Orion, PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;
let values: Vec<f32> = (0..64 * 13).map(|i| (i % 251) as f32).collect();
let layout = PaddedLayout::for_kernel::<L2F32Distance>(13)?;
let dataset = PaddedDataset::from_packed(&values, layout)?;
let index = Orion::<f32>::build(dataset, BuildConfig {
    degree: 8, build_l: 24, threads: 2, ..Default::default()
})?;
assert_eq!(index.graph().num_nodes(), 64);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Inspect an index with `dataset()`, `graph()`, `len()` and `entry_point()`.
These borrow storage or return metadata without copying. To edit the base,
consume the index with `into_dataset()` and rebuild it, so fitted sidecars cannot
outlive the vectors they describe. Use the application crate’s
`orion_ann::Index::save` and `Index::open` to persist and verify a snapshot,
including its graph, base identity and calibration.

## Handle errors and inspect state

```rust
use orion_ann_core::{BuildConfig, ErrorCode, Orion};
let error = Orion::<f32>::build_from_slice(&[], 13, BuildConfig::default())
    .unwrap_err(); // Orion implements Debug, as required by unwrap_err.
assert_eq!(error.code(), ErrorCode::InvalidArgument);
eprintln!("{}: {}", error.code(), error);
let report = error.report(); // Structured payload shared with CLI and Python.
assert_eq!(report.operation.as_deref(), Some("build_graph"));
```

Public types implement `Debug`. Index, dataset, sidecar and scratch summaries
omit vector payloads and working buffers. Configurations implement `Clone` and
`PartialEq`; defaults exist where a useful valid configuration can be supplied.
Float-containing configurations intentionally do not implement `Eq`/`Hash`.
Large index handles do not implement `Clone` or `Default`: use `Arc` for shared
ownership, or explicitly build/load a separate index. `Debug` output is for
inspection, not a stable serialization format.

| Public types | Supported traits / ownership |
| --- | --- |
| Build and search configurations | `Debug`, `Clone`, `PartialEq`, useful `Default`; small search controls also `Copy` |
| `Orion`, `PaddedDataset`, `GraphView` | Summary `Debug`, `Send + Sync` for supported vector types; move or share with `Arc` |
| Runtime `Index`, `BaseInput` | Summary `Debug`, `Send + Sync`; no implicit deep clone; serialize mutable service operations |
| Runtime `SearchResult` | Summary `Debug`, `Clone`, `PartialEq`, `Eq`; cloning explicitly copies result vectors |
| `Error` | `Debug`, `Display`, `std::error::Error`; preserved sources are not required to be cloneable |
| `ErrorCode` / `ErrorReport` | Codes: `Copy`, `Eq`, `Hash`, `Display`, serialization; reports: `Clone`, `PartialEq`, serialization |

Serialization remains explicit: use configuration Serde, `Error::report`, or
the cache APIs. Raw in-memory indices and borrowed search stages are not Serde
snapshots. Their `Send`/`Sync` guarantees do not make every operation lock-free.


The default `parlay` feature bundles ParlayANN and needs a C++17 compiler.
With `default-features = false`, automatic construction uses Rust DiskANN for
f32 L2 data (up to 1536 dimensions); native u8 requires ParlayANN. Explicit
backend choices never silently fall back. Construction does not normalize
coordinates: normalize cosine data before building an L2 graph. See
[`BuildConfig`](crate::BuildConfig) for parameter bounds and
`Orion::build` for ownership and error guarantees.

Fallible operations use `Error`, `ErrorCode`, `ErrorReport` and `Result`.
Inspect error codes/details rather than parsing messages. These errors preserve
backend causes and do not log automatically. The low-level graph/cascade modules
require their documented invariants; they are not an untrusted graph-file parser.

The core accepts Orion's built-in cascade stages. The prefilter, admission and rerank
implementation traits are internal under every feature combination. Use
`orion-ann::IndexOptions` to select a built-in policy in an application.

The optional `benchmark` feature exposes phase counters, calibration samples,
profiling and ablation controls, fixed-dimension constants, and the historical
naive and async search prototypes. Environment overrides for experimental
search tuning take effect only with this feature. Ordinary builds use the
compiled search defaults.

## Experimental partition import

The default-disabled `experimental-graph` feature exposes `supplier`:
`PartitionSpec`, `PartitionSource`, borrowed `NodePartitions`, and STAG v4
stream codecs. `Orion::from_partition_source` writes validated rows directly to
its final resident or mapped slab. The runtime crate `orion-ann` supplies base
identity checks, atomic file publication and snapshot lifecycle. No supplier
callbacks or validation occur in the search hot path.
