A saved snapshot contains graph metadata, graph files, provenance and per-k
calibration. **The base vectors remain external** and must be supplied again.

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
index.calibrate(values[..6].to_vec(), 2, Some(8))?;
let snapshot = dir.join("snapshot");
index.save(&snapshot)?;
let mut reopened = Index::open(&snapshot,
    BaseInput::from_f32(&values, 3)?, Some(1), None)?;
assert_eq!(reopened.len(), index.len());
assert_eq!(reopened.search(values[..3].to_vec(), 2, 8, true)?.counts, vec![2]);
# drop(reopened);
# drop(index);
# std::fs::remove_dir_all(&dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Integrity and errors

`Index::open` never rebuilds on a miss. It checks metadata version, input identity,
configuration, recorded graph stamps and calibration. A mismatch returns
`CacheIncompatible`; malformed serialized data returns `InvalidData`; filesystem
failures return `Io`. File bases are identified by their canonical path and file
stamp; copied arrays use their content identity. This is cache provenance,
not cryptographic authentication of every graph/sidecar byte.

Caches are native trusted artifacts produced by this library, not an interchange
format for arbitrary hostile files. Keep bases/cache files unchanged while
opening or saving. Do not assume graph slab files are portable between every
architecture or future version. STAG import and native cache loading are different
operations; use the runtime to enforce provenance and compatibility.

## Publication and concurrency

`save` writes into a sibling staging directory and renames it after successful
writes. It rejects an existing destination and attempts staging cleanup on error;
the in-memory index stays usable. Serialize publishers targeting the same
location. Directory rename visibility does not guarantee complete crash durability
on all filesystems (parent directory fsync is not performed).

The lower-level `Orion::save/load_from_cache` pair stores graph metadata and a
`.pgraph` file, not a runtime snapshot. Its two renames are not a multi-file
transaction, and it does not provide the runtime's base identity/calibration
contract. Do not substitute it for `Index::save/open` in an application service.
