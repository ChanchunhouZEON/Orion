# orion-ann-runtime

Owned build, calibration, search and cache API for Orion ANN.

Part of [Orion ANN](https://github.com/ChanchunhouZEON/Orion). Requires Rust 1.92 or newer. Licensed under MIT.

Use `orion_ann_runtime::{Index, IndexOptions, BaseInput}` for owned construction,
calibration, batch search and persistence. The `guide` module contains executable
examples for configuration, build, cache and search, with errors, concurrency and
panic contracts. Enable `cli` only for preset/command-line configuration support.

```toml
[dependencies]
orion-ann-runtime = "0.1.0"
```

The embedded preset is `configs/sweep.yaml`. Runtime use does not depend on the
repository's benchmark folder. Base vectors are external to saved snapshots.

The default `parlay` feature bundles the ParlayANN memory builder for f32/u8 and
requires a C++17 compiler on Linux/macOS. Cache/STAG loading remains available
with `default-features = false`; automatic construction then falls back to Rust
for f32 only. Explicit missing staged files and corrupt caches remain errors.

## Build, calibrate, search and reopen

Arrays are row-major f32 or native u8. `Index` owns a worker pool and reuses it
for construction and search. Adaptive search requires calibration for its `k`.

```rust
use orion_ann_runtime::{BaseInput, GraphConfig, Index, IndexOptions};
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
let results = index.search(queries.clone(), 3, 32, true)?;
for (row, &count) in results.ids.chunks(results.k).zip(&results.counts) {
    println!("{:?}", &row[..count as usize]); // Ignore any u32::MAX padding.
}
let snapshot = std::path::PathBuf::from("cache/snapshot"); // Must not exist.
# let snapshot = root.join("snapshot");
index.save(&snapshot)?;
let mut reopened = Index::open(&snapshot, BaseInput::from_f32(&base, 13)?, Some(2), None)?;
assert_eq!(reopened.search(queries, 3, 32, true)?, results);
println!("{reopened:?}"); // Shape, worker count and calibration count, not vectors.
# drop(reopened);
# drop(index);
# std::fs::remove_dir_all(root)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `BaseInput::File("base.fbin".into())` instead of `from_f32` for file input,
or `BaseInput::from_u8(&bytes, dimension)` for native bytes. File format overrides
belong to `IndexOptions::base_format`; the runtime handles decoding and cosine
normalization. The same base must be supplied when reopening a snapshot.

For uncalibrated graph/cascade search, pass `adaptive=false` to `search`. This
remains approximate search. Search requires mutable access to the handle;
serialize requests with a mutex when sharing one `Index` across threads. `Index`
and `BaseInput` are movable and implement summary `Debug`, but deliberately do
not implement `Clone` or `Default`. `SearchResult` implements `Clone`, `PartialEq`
and `Eq`; its `Debug` shows dimensions/counts rather than every result ID.
