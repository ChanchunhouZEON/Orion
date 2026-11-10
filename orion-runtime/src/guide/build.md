<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

[`crate::Index::new`] is the recommended owned construction interface. It
loads/builds the graph and base; it does not require a query set or ground truth.

```rust
use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions, Result};
use orion_ann::GraphSourcePolicy;
# let dir = std::env::temp_dir().join(format!("orion-doc-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
let values: Vec<f32> = (0..24).map(|x| x as f32 / 24.0).collect();
let options = IndexOptions {
    threads: 1,
    cache_dir: dir.join("build"),
    graph: GraphConfig { source: GraphSourcePolicy::Rust, degree: 4,
        build_l: 8, max_extra: 2, ..Default::default() },
    ..Default::default()
};
let mut index = Index::new(BaseInput::from_f32(&values, 3)?, options)?;
assert_eq!(index.dimension(), 3);
assert_eq!(index.len(), 8);
# drop(index);
# std::fs::remove_dir_all(&dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Coordinate validation and normalization

The runtime prepares resident coordinates according to `IndexOptions::metric`
before graph construction or search. Calibration and search queries use the
same policy:

| Metric | Coordinate policy |
| --- | --- |
| `L2` | Check finite f32 coordinates; preserve values and vector lengths. |
| `InnerProduct` | Check finite coordinates; preserve vector lengths because they affect inner products. |
| `Cosine` | Check finite coordinates, reject zero vectors, and divide each row by its L2 norm. |

Cosine search uses inner products of unit vectors. File decoding applies this
policy to resident rows; array input applies it to Orion's owned copy. Neither
path changes the caller's array or source file, and SIMD padding is excluded.
Native-u8 storage supports L2 and keeps the original byte values.

Cache and snapshot fingerprints identify the **original input**, not the
normalized coordinates. Array input is fingerprinted and checked against a
snapshot before its owned copy is normalized. File input is hashed as original
bytes are read, while the decoded rows are prepared in memory. Reopening a
snapshot therefore requires the original base input, even for cosine search.

## Search a prefix of a native-u8 file

With the default `parlay` feature, Orion can build a graph directly from a
`.u8bin` file. Set `max_points` to try a prefix before building the full index.
IDs refer to zero-based rows in the base. The resident base stays in byte
storage. Convert only the query rows to the f32 values accepted by `Index`.

```rust
# #[cfg(feature = "parlay")]
# {
use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions, VectorStorageKind};
# let dir = std::env::temp_dir().join(format!("orion-byte-guide-{}", std::process::id()));
# std::fs::create_dir_all(&dir)?;
let path = std::path::Path::new("descriptors.u8bin");
# let path = &dir.join("descriptors.u8bin");
# let mut file = Vec::from(32u32.to_le_bytes());
# file.extend(3u32.to_le_bytes());
# for row in 0u8..32 { file.extend([row, row + 1, row + 2]); }
# std::fs::write(path, file)?;
let options = IndexOptions {
    max_points: Some(16),
    graph: GraphConfig { degree: 4, build_l: 8, max_extra: 2, ..Default::default() },
    ..Default::default()
};
# let options = IndexOptions { cache_dir: dir.join("build"), threads: 1, ..options };
let mut index = Index::new(BaseInput::from_file(path), options)?;
assert_eq!(index.len(), 16);
assert_eq!(index.storage(), VectorStorageKind::U8);

let warmup = [1u8, 2, 3, 6, 7, 8];
index.calibrate(warmup.into_iter().map(f32::from).collect(), 2, Some(8))?;
let result = index.search(QueryBatch::from_f32(&[9.0, 10.0, 11.0], 3)?, &SearchOptions::new(2, 8))?;
assert_eq!(result.counts, [2]);
# std::fs::remove_dir_all(dir)?;
# }
# Ok::<(), Box<dyn std::error::Error>>(())
```

For array input, slice the packed rows before calling `BaseInput::from_u8` or
`from_f32`. For file input, retain the full original file when you save a
snapshot, even if you indexed only a prefix.

## Import an existing ParlayANN graph

Supply the matching base in its original row order and set `degree` to the
degree recorded in the STAG v3 header. Import streams directly into the graph slab.

```no_run
use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions};
use orion_ann::GraphSourcePolicy;
let index = Index::new(BaseInput::from_file("base.u8bin"), IndexOptions {
    graph: GraphConfig { source: GraphSourcePolicy::Parlayann,
        staged_file: Some("pa.staged".into()), degree: 64,
        ..Default::default() },
    ..Default::default()
})?;
# Ok::<(), orion_ann::Error>(())
```

## Errors

Malformed shape, invalid coordinates and controls return input errors. Native
u8 must have a compatible graph; missing/invalid staged files return an error.
I/O causes and paths are preserved where available. Backend construction errors
propagate through the same `Error` type. Budget failures occur before the payload
load, but constructing a copied array input already allocates its own buffer.

## Ownership, side effects and panic policy

`BaseInput::from_f32/from_u8` copies the caller's row-major slice into owned,
aligned storage; the caller may reuse its original slice immediately. `Index`
takes ownership of that storage and its Rayon pool. Construction can create
cache files. Builders sharing a directory use a nonblocking writer lease; a conflict returns InvalidState. Use separate directories to build concurrently. A
failed build may leave temporary artifacts; reuse only a complete validated cache.

Input errors are recoverable `Result`s, not a promise that all Rust code is
panic-free. Low-level invariant failures and allocator/process failures follow
the host's panic/allocation policy. Python translates unwinding Rust panics and
rejects subsequently poisoned handles. No Rust panic recovery is installed by
the core library.

## Core construction without the service layer

`orion_ann_core::Orion::build` and `build_from_slice` accept `orion_ann_core::BuildConfig`
and return an in-memory index without cache I/O. `Index::new` delegates to the
same builder after resolving files and cache policy. Programmatic graph defaults
come from `BuildConfig`; the CLI uses the same defaults. Dataset-specific YAML
recipes belong only to the benchmark workspace.
The runtime normalizes cosine input before passing it to the core L2 builder.
