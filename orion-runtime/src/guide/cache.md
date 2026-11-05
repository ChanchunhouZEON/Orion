<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

## Working cache and snapshot

Use a **working cache** to reuse the cost of building or importing a graph.
Set `IndexOptions::cache_dir` (CLI `--cache-dir`) when creating an index.
Orion can reuse a matching graph and lazily load or rebuild quantized sidecars.
Even on a cache hit, `Index::new` starts with no calibration records.

Use a **snapshot** to record a particular configuration and its per-`k`
calibrations. `Index::save` publishes a single `index.json` in a new directory.
It records the SHA-256 identities of the graph, its metadata, build provenance
and each materialized sidecar. Adding a calibration writes another small
manifest. The graph and sidecars keep their existing files.

With `cache_dir = "service/work"`, two snapshots can share this layout:

```text
service/
  work/                  # Disposable build and sidecar lookup records
  work.objects/          # Persistent objects shared by this cache and its snapshots
    <sha256>.pgraph      # Graph slab
    <sha256>.bin         # Graph header metadata
    <sha256>.cache.json  # Build provenance
    <sha256>.qds         # Example f32 admission sidecar, created on first use
  k10/index.json
  k10-k100/index.json
```

Keep snapshots outside the working directory. Stop active cache writers before
removing `work/`. `Index::open` (CLI `--snapshot`) needs the snapshot manifest,
the object store at the absolute path recorded in `object_store`, and the original
base input. It can open either snapshot after `work/` is gone. Back up all three.
Clearing the working cache never garbage-collects objects, so older manifests
continue to resolve. Independent working directories have independent sibling
stores. Older standalone snapshots can also be opened; their directory serves
as the object store for snapshots subsequently saved from that handle.

```rust
use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions, OpenOptions, SnapshotRef};
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
index.calibrate(values[..6].to_vec(), 2, Some(8))?;
let expected = index.search(QueryBatch::from_f32(&values[..3], 3)?, &SearchOptions::new(2, 8))?;
let snapshot = index.save(&dir.join("snapshot"))?;
drop(index);
std::fs::remove_dir_all(dir.join("build"))?; // The sibling build.objects/ stays in place.

// Keep this reference in trusted application state between service runs.
let record = serde_json::to_string(&snapshot)?;
let reference: SnapshotRef = serde_json::from_str(&record)?;
let mut reopened = OpenOptions { threads: Some(2), ..Default::default() }
    .open(&reference, BaseInput::from_f32(&values, 3)?)?;
assert_eq!(reopened.search(QueryBatch::from_f32(&values[..3], 3)?, &SearchOptions::new(2, 8))?, expected);
# drop(reopened);
# std::fs::remove_dir_all(&dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The reference serialization in this example uses `serde_json` as an application dependency.

## Integrity and publication

The manifest binds graph metadata, native graph bytes, build provenance, materialized
sidecars and the external base to the manifest in `index.json`. File input uses
all original bytes, including rows outside a selected prefix. Array input uses
its typed shape and row contents. Opening rejects inconsistent inputs and
can materialize additional derived stages into the same store when first needed.

A session retains the identities of the graph generation and sidecars it loaded.
Each graph or sidecar decoder hashes its input stream during the read. With an
authenticated base, working sidecar keys include the source identity, row prefix,
dimension, representation, metric and codec. The first use of an older working
sidecar record rebuilds that record. Existing snapshot references still identify
the original objects.

Repeated searches use resident data. Opening a new session verifies its inputs
again, and saving a file-backed session rechecks the external base because another
process could have replaced it since opening.

Keep the `SnapshotRef` returned by `Index::save` in trusted application state.
A local application can simply call `Index::open("cache/sift", base)?` to
load the current snapshot in that directory. Both forms verify all payloads.
Passing a reference additionally checks the manifest identity before parsing it.
Serialize the reference with Serde when it needs to survive a restart.
Payload filenames use the full SHA-256 of their bytes. Start a fresh cache when
upgrading from older cache layouts.

Object publication syncs the encoded payload before exposing its content name.
A save syncs the manifest in a new sibling staging directory (owner-only on Unix),
then renames that directory. Unix also syncs the staging and parent directories. If the last sync fails after rename, inspect the
destination before retrying. Serialize writers to the same destination and
keep the external input immutable. Search and dropping an index never commit
a snapshot.
