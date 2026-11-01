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
