[`crate::Index::new`] is the recommended owned construction interface. It
loads/builds the graph and base; it does not require a query set or ground truth.

```rust
use orion_ann_runtime::{BaseInput, GraphConfig, Index, IndexOptions, Result};
use orion_ann_runtime::config::GraphSourcePolicy;
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

The default backend builds directly from f32 or u8 rows without an external
executable. A C++17 compiler is required when building the default runtime from
source. To import an existing ParlayANN STAG v3 export instead:

```no_run
use orion_ann_runtime::{BaseInput, GraphConfig, Index, IndexOptions};
use orion_ann_runtime::config::GraphSourcePolicy;
let index = Index::new(BaseInput::File("base.u8bin".into()), IndexOptions {
    graph: GraphConfig { source: GraphSourcePolicy::Parlayann,
        staged_file: Some("pa.staged".into()), degree: 64,
        ..Default::default() },
    ..Default::default()
})?;
# Ok::<(), orion_ann_runtime::Error>(())
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
cache files, so give each concurrent builder a separate cache directory. A
failed build may leave temporary artifacts; reuse only a complete validated cache.

Input errors are recoverable `Result`s, not a promise that all Rust code is
panic-free. Low-level invariant failures and allocator/process failures follow
the host's panic/allocation policy. Python translates unwinding Rust panics and
rejects subsequently poisoned handles. No Rust panic recovery is installed by
the core library.
