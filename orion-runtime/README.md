<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# orion-ann

Owned build, calibration, search and cache API for Orion ANN.

Part of [Orion ANN](https://github.com/ChanchunhouZEON/Orion). Requires Rust 1.92 or newer. Licensed under MIT.

Use `orion_ann::{Index, IndexOptions, QueryBatch, SearchOptions, BaseInput}` for owned construction,
calibration, batch search and persistence. The `guide` module contains executable
examples for configuration, build, cache and search, with errors, concurrency and
panic contracts. Enable `cli` only for command-line adapter support.

```toml
[dependencies]
orion-ann = "0.1.0"
```

Dataset presets and YAML configuration belong exclusively to the unpublished benchmark package.
The CLI, Rust and Python interfaces share `IndexOptions` defaults.

The default `parlay` feature bundles the ParlayANN memory builder for f32/u8 and
requires a C++17 compiler on Linux/macOS. Cache/STAG loading remains available
with `default-features = false`; automatic construction then falls back to Rust
for f32 only. Explicit missing staged files and corrupt caches remain errors.

Ordinary applications need only this package. Search stages are supplied by Orion. Enable `benchmark` for research-only diagnostics and timing controls.

The default Rust API exports the owned index, input constructors, configuration
and result types, snapshot references, errors and search traces. Adapter execution helpers
are available under `execution` with the `cli` feature; argument parsing lives in `orion-cli`. Python enables
`file-queries` for query-file decoding. The separate experiment executables use
the `benchmark` module. Raw allocations, graph loaders and resolved recipes
are not part of the default API.

## Build, calibrate, search and reopen

Arrays are row-major f32 or native u8. `Index` owns a worker pool and reuses it
for construction and search. Adaptive search requires calibration for its `k`.

```rust
use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions};
# let root = std::env::temp_dir().join(format!("orion-runtime-readme-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
let base: Vec<f32> = (0..64 * 13).map(|i| (i % 251) as f32).collect();
let options = IndexOptions {
    threads: 2,
    graph: GraphConfig { degree: 8, build_l: 24, max_extra: 4, ..Default::default() },
    cache_dir: "cache/build".into(),
    ..Default::default()
};
# let options = IndexOptions { cache_dir: root.join("build"), ..options };
let mut index = Index::new(BaseInput::from_f32(&base, 13)?, options)?;
let queries = base[..8 * 13].to_vec();
index.calibrate(queries.clone(), 3, Some(24))?;
let results = index.search(QueryBatch::from_f32(&queries, index.dimension())?, &SearchOptions::new(3, 32))?;
for (row, &count) in results.ids.chunks(results.k).zip(&results.counts) {
    println!("{:?}", &row[..count as usize]); // Ignore any u32::MAX padding.
}
let snapshot = std::path::PathBuf::from("cache/snapshot"); // Must not exist.
# let snapshot = root.join("snapshot");
let snapshot = index.save(&snapshot)?;
let reopened = Index::open(&snapshot, BaseInput::from_f32(&base, 13)?)?;
assert_eq!(reopened.search(QueryBatch::from_f32(&queries, index.dimension())?, &SearchOptions::new(3, 32))?, results);
println!("{reopened:?}"); // Shape, worker count and calibration count, not vectors.
# drop(reopened);
# drop(index);
# std::fs::remove_dir_all(root)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `BaseInput::from_file("base.fbin")` instead of `from_f32` for file input,
or `BaseInput::from_u8(&bytes, dimension)` for native bytes. File format overrides
belong to `IndexOptions::base_format`; the runtime handles decoding and cosine
normalization. The same base must be supplied when reopening a snapshot.

Calibrate each requested `k` before search, or open a snapshot containing that
calibration. Missing calibration returns `InvalidState`. Search takes `&self` and a
borrowed `QueryBatch`; share an index through `Arc` for concurrent requests.
`SearchOptions` supplies each request's k and beam without changing index state.
Calibration still requires exclusive `&mut` access. `Index` and `BaseInput` are movable and implement summary `Debug`, but deliberately do
not implement `Clone` or `Default`. `SearchResult` implements `Clone`, `PartialEq`
(scores contain floats and NaN padding, so it does not implement `Eq`); its `Debug` shows dimensions/counts rather than every result ID.

## Experimental graph suppliers

Enable `experimental-graph` to use `Index::from_staged` or
`Index::from_partitions` with named local/remote/extra partitions. The common
boundary validates structure, preserves entry and neighbor order, and applies
ordinary memory planning. Calibration and `save` remain explicit. Imported
snapshots open without this feature or the original supplier. See the
`supplier` API and `guide::supplier` module when building documentation with the
feature enabled. The default ParlayANN builder is unchanged.
