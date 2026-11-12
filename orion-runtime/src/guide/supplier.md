<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Bring your own partitioned graph

Enable `experimental-graph` on `orion-ann` to import a graph prepared by your own
supplier. This is an experimental extension contract, not a new default builder.
The ordinary ParlayANN and Rust construction paths are unchanged.

The supplier owns graph construction and partitions each node's ordered neighbors
into `local`, `remote` and `extra`. Orion imports these partitions directly into
its final graph slab. It does not prune, sort, add reverse edges, replace the
entry point, or silently invoke a different builder. Existing SIMD kernels,
search stages, and prefetch scheduling remain unchanged.

## Staged file to calibrated index

```no_run
use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions};

let mut index = Index::from_staged(
    BaseInput::from_file("base.fbin"),
    "supplier.staged",
    IndexOptions { cache_dir: "supplier-cache".into(), threads: 8,
                   ..Default::default() },
)?;
let queries = vec![0.0_f32; index.dimension() * 32]; // use representative queries
index.calibrate(queries.clone(), 10, None)?;
let results = index.search(QueryBatch::from_f32(&queries, index.dimension())?,
                           &SearchOptions::new(10, 64).with_scores(true))?;
let saved = index.save(std::path::Path::new("supplier-snapshot"))?;
let restored = Index::open(saved, BaseInput::from_file("base.fbin"))?;
# Ok::<(), orion_ann::Error>(())
```

`IndexOptions.graph` must retain its default on this path. The supplied descriptor
owns capacities and entry; Vamana `alpha`/`build_l` do not describe imported edges.
Other options still control metric, cascade, threads, budgets, auto pinning and
resident/mapped placement. `options.metric` must match the supplier descriptor.
Both f32 and native u8 support squared L2, inner product and cosine. A cosine
graph must use cosine geometry: f32 construction uses unit-normalized base rows,
while native u8 retains raw bytes and uses separate squared norms. Orion applies
the corresponding query preparation. Supplier metric metadata must agree with
the requested metric.

An ordinary snapshot contains the imported native graph, base identity, supplier
provenance and calibration. Restoring it does **not** require `experimental-graph`,
ParlayANN, the original staged file, or the supplier implementation. A staged
import publishes immutable objects, but does not establish a recipe-based working
cache lookup: call `save` and subsequently `open` to skip re-importing the graph.

## In-memory or streaming producer

`Index::from_partitions(base, spec, source, options)` consumes an owned
`PartitionSource + Send + 'static` on the index's configured pool. The source
returns borrowed `NodePartitions` in node order, so a decoder can reuse one row
buffer. `OwnedPartitions` moves a precomputed graph into the import and releases
consumed rows. At the lower core boundary, `Orion::from_partition_source` also
accepts a borrowed `SlicePartitions` and explicit storage options.

Both paths validate node count, entry, dimension/type, per-partition capacities,
ID bounds, self edges, and duplicate IDs **across all three lists for each node**.
They reject missing/extra rows. They preserve the supplier's original list order.
Validation is structural: connectivity, partition semantics, geometric quality
and recall remain the supplier's responsibility. Calibration is still explicit.

`PartitionOrigin { name, revision }` records human-readable provenance. The name
is bounded printable text, never used as a path or trusted backend selector.
Cache payloads are addressed by SHA-256; graph identity includes base identity,
graph bytes, entry metadata and metric. Renaming a supplier does not duplicate
its identical graph objects. Source revision is unrelated to file format version.

## Producer files and compatibility

`write_staged_file(path, &spec, &mut source)` validates all rows, writes in 8 MiB
buffered blocks, flushes and synchronizes the temporary file, then publishes it
atomically without overwriting an existing destination. Errors before publication
remove the temporary file. A directory-sync failure after publication may return
an error with the complete destination already present. Concurrent mutation of
input files is unsupported; keep artifacts immutable during import.

STAG v4 carries bounded JSON metadata and little-endian partition rows. A
language-independent description is in the repository's `docs/supplier-staged.md`.
The public `StagedReader`/`write_staged` codec uses `Read`/`Write`, not filesystem
paths. A decoder retains one row; no second full graph is allocated by Orion.

Legacy v3 can be imported with `Index::from_legacy_staged(base, path, spec, options)`.
The caller supplies the metadata missing from v3, including origin, metric and
dimension. Node count, capacities and entry must agree with the v3 header. This
adapter preserves entry, unlike the existing ParlayANN-specific import policy.
Absent base identity cannot establish that a graph belongs to a particular base.

## Python and CLI

Build each adapter with its own `experimental-graph` feature:

```text
cargo build -p orion-ann-cli --features experimental-graph
maturin develop --manifest-path orion-py/Cargo.toml --features experimental-graph
```

Python exposes staged import, not callbacks into Python during C++/Rust work:

```python
index = orion_ann.Index.from_staged("base.fbin", "supplier.staged", threads=8)
index.calibrate(queries, k=10)
result = index.search(queries, k=10, search_list_size=64, scores=True)
index.save("supplier-snapshot")
```

```text
orion --base base.fbin --query query.fbin --supplier-staged supplier.staged \
      --k 10 --search-l 64 --threads 8 --cache-dir supplier-cache
```

`--supplier-staged` conflicts with builder-specific flags. `--preflight` uses its
capacities for the allocation estimate. The normal `--staged-file` remains the
ParlayANN v3 path. Python/CLI/Rust route through the same importer and errors.
