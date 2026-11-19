<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> orion-ann-platform

Aligned I/O, block writes and memory support for Orion ANN.

Part of [Orion ANN](https://github.com/ChanchunhouZEON/Orion). Requires Rust 1.92 or newer. Licensed under MIT.

This component is versioned with the Orion ANN workspace. Most applications
should depend on `orion-ann` or `orion-ann-core` instead of this lower-level
implementation crate. See rustdoc for contracts of individual APIs.


The `reading` module defines a runtime-independent `ObjectReader` contract.
`ReadRegion` implements it for both resident and mapped backing. Reads return
borrowed `ReadLease` values without allocation or reference-count operations;
`into_owned` explicitly retains the backing when a lease must outlive its reader.
Typed access validates alignment and element size. Mapped views may still fault
on access. File readers are implemented separately in `orion-ann-ssd`; the
platform contract does not depend on Tokio.

The `read_path` example compares direct resident access against a read lease
around the unchanged SIMD distance stream, using synthetic 128- and 960-D rows.
It alternates order over seven pairs and measures each variant for at least one
second. This isolates acquisition overhead, not end-to-end ANN performance:

```sh
cargo run --release -p orion-ann-platform --example read_path
```

### Private file-cache reclamation

`MutableRegion::flush_and_reclaim` and `ReadRegion::reclaim_file_pages` validate
ranges against their owners, write back shared file pages and request cache
reclamation without dropping ownership or allocation leases. Resident buffers
are unchanged; anonymous/COW memory is never passed to the eviction operations.
Linux combines `MS_SYNC`, `MADV_DONTNEED` and `POSIX_FADV_DONTNEED`; macOS uses
`MS_SYNC | MS_INVALIDATE`. Errors are returned, not silently treated as eviction.
Success only confirms OS acceptance: pinned pages, concurrent readers and the
filesystem may prevent or immediately undo reclamation.

`RegionOptions::reclaim_after_write` opts mapped producers into bounded completed
write checkpoints. It defaults to false; low-memory storage experiments enable
it consistently across all arms. `file_cache::sync_and_reclaim_file` provides the
corresponding operation for private staging files after userspace buffers flush.
There is no global cache drop or per-query eviction in the search hot path.
