<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Orion SSD search foundation

This experimental crate supplies explicit file reads through the same
`ObjectReader` / `ReadRequest` / `ReadLease` contract used by resident and mapped
regions in `orion-ann-platform`. `SearchDriver` executes the core traversal with
page-backed graph records and the existing resident/mapped base and sidecar sessions.

```rust,no_run
use orion_ann_platform::reading::{ObjectReader, ReadRequest};
use orion_ann_ssd::{FileReader, FileReaderOptions};

async fn read_record() -> std::io::Result<()> {
    let reader = FileReader::open("graph.bin", FileReaderOptions::default())?;
    let lease = reader.read(ReadRequest::new(4096, 4096)?).await?;
    let words = lease.as_slice::<u32>()?;
    assert_eq!(words.len(), 1024);
    Ok(())
}
```

Nonempty file reads must be polled in a Tokio runtime. The baseline uses bounded
`spawn_blocking` positioned reads, not io_uring or direct I/O. File-system caching
still applies. Resident/mapped reads need no runtime; mapped views can fault on
access, so immediate completion does not mean nonblocking memory access.

Each reader is bound to one regular file and captures its size at open. Publish
immutable generations; this reader does not verify hashes or defend against
concurrent logical file modifications. Short reads fail without exposing partial
results. Returned owned buffers remain valid even if the source file is changed
or removed afterwards. Offsets need not be aligned in this buffered baseline.
Buffer alignment is checked independently; typed views also validate element
alignment and size. Dataset layout adapters remain responsible for SIMD padding.

Cloned readers share an in-flight semaphore. Allocation occurs after acquiring a
slot and uses the optional shared `AllocationAccount`. The slot remains owned by
the blocking operation if the caller cancels; the allocation remains owned by
the result until its final drop. This bounds submitted/running blocking operations, not the number of futures
waiting for a slot or completed leases retained by the caller. The query scheduler
must separately bound incoming requests. Supply a shared allocation budget when
that live payload must also be bounded. These limits are not RSS/page-cache caps.

## Page cache and search

```rust,no_run
use orion_ann_ssd::{CachedReader, FileReader, FileReaderOptions, PageCacheOptions, SearchDriver};

async fn open_graph() -> orion_ann_core::Result<()> {
    let file = FileReader::open("graph.bin", FileReaderOptions::default())?;
    let cache = CachedReader::new(file, PageCacheOptions {
        page_bytes: 4096,
        capacity_bytes: 256 * 1024 * 1024,
        ..Default::default()
    })?;
    let driver = SearchDriver::open(cache, 0).await?;
    // Supply the verified manifest's entry point above. Search accepts borrowed
    // base rows, QueryInput, DynamicSearchConfig and the existing DynamicCascade.
    assert!(driver.graph().num_nodes() > 0);
    Ok(())
}
```

`model::graph::storage_graph::StorageGraph` owns the graph's layout, validation and node
leases, alongside the resident `PhasedGraph` model. `search::storage::SearchDriver`
owns query execution over that model. To prepare them separately, use
`StorageGraph::open(reader, entry).await?` followed by `SearchDriver::new(graph)`;
the driver takes ownership without copying the graph. Both models implement the
internal `model::graph::graph_access` contract; neither graph model depends on search.

One cache owns one immutable object generation. Concurrent misses for the same
page share a load. LRU eviction skips pages still held by a query; capacity
waiters wake when the final caller releases its lease. The capacity counts full
page slots, including in-flight reads and leases retained after `clear()`. A short
last page still consumes one slot. `clear()` does not cancel pending loads.
Keep leases scoped to computation: awaiting another page while retaining every
slot can prevent progress. The search driver releases a node before requesting
the next one.

A cross-page request assembles contiguous bytes while holding at most one page
at a time, so even a one-slot cache can service a record crossing page boundaries.
Its output is bounded by `max_read_bytes` and charged through
`assembly_accounting`, separately from cache capacity. Use a shared
`AllocationBudget` for both file-reader and assembly accounts to bound their
combined live payload. Metadata, input objects, query scratch and OS page cache
are not part of the retained-page limit. Limit concurrent requests at the caller.

Search suspension is query-local: the future retains the beam, visited set,
convergence state, prepared sessions and private scratch at a graph-read boundary.
Repolling after I/O completes resumes the same traversal. Dropping a query future
releases its state; a shared page load continues for remaining waiters. A started
blocking read cannot be forcibly cancelled. Runtime shutdown cleans pending cache
entries, while already running blocking I/O retains its own buffer until it ends.
This is in-process suspension, not a serialized/resumable search checkpoint.

The driver shares the existing compute body, stage sessions, SIMD streams,
partition order, flush scheduling and near/far prefetch policy. The resident
adapter returns an immediately ready future and is polled on the caller without
Tokio. Explicit graph reads require Tokio only when using this cache/file backend;
computation runs on the polling thread between I/O waits. Cached queries may run
to completion without yielding; cancellation takes effect at suspension points.

This stage pages the graph only. Base rows and sidecars still use their ordinary
resident/mapped sessions; there is no async vector fetch, speculative I/O batching,
new disk layout, label-filtered driver, runtime/CLI deployment path or physical SSD
performance claim.
Records use the existing native graph codec. Headers and records are validated
before neighbor IDs enter distance streams, but validation is not authentication:
open verified immutable generations and supply the matching base/cascade/entry
point. Do not modify an opened generation. Runtime still owns snapshot and
calibration policy; the experimental core feature is `experimental-storage`.
