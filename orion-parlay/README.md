<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> orion-ann-parlay

In-memory ParlayANN Vamana construction for Orion. The runtime enables this
backend by default; Python and CLI use the same runtime. Nothing is downloaded
by build.rs. Build from source requires a C++17 compiler on Linux or macOS.

`Builder::f32` and `Builder::u8` borrow immutable rows, including an explicit
logical dimension and element stride. The returned handle cannot outlive those
rows. Native u8 stays u8; f32 remains unquantized. L2 and f32 inner product are
supported. The runtime normalizes cosine inputs before invoking the L2 builder.
The recipe uses two passes, captures capped extras only on the final pass, and
preserves the original distance-only partition ordering and tie behavior.

The production runtime uses `Builder::f32_into` / `Builder::u8_into`: Rust owns
one final aligned slab and C++ exclusively borrows it for both adjacency and
extras. A temporary degree field sits immediately before the neighbor array;
extras occupy fixed-capacity slots until finalization. Once construction joins,
each worker reads one node into bounded scratch and writes its final local,
remote and extra partitions back into that same row. No second full graph or
N-sized extras/count allocation is created. Rust validates the completed rows
before publication; failures discard the unpublished slab.

The owning `Builder::f32` / `Builder::u8` and `next_partition` path is retained
as an independent storage baseline. Only that path allocates a C++ graph/extras
and bounded partition batches alongside the destination slab. Both paths use
the same pruning and partition arithmetic. Base vectors are always borrowed.
Construction statistics, insertion batches, worker scratch and allocator overhead
remain additional to the production preflight's resident payload lower bound.

Errors are explicit status/message values across a private C ABI. The runtime
maps these to its shared `Error`/`OrionError` contract. Worker exceptions are
joined and rethrown on the calling thread before crossing the ABI. Unrecoverable
process failures (for example an OS OOM kill) are outside this contract.
Each operation uses a private scheduler with the requested worker count and
does not modify environment variables or install global logging/panic hooks.

## Vendored sources

`vendor/SOURCE.json` pins the ParlayANN/Parlaylib revisions and original hashes.
The snapshot includes the local Orion fork's extras/STAG and large-I/O changes.
Original licenses are retained in `vendor/PARLAYANN-LICENSE`,
`vendor/PARLAYLIB-LICENSE` and source headers. Bridge patches:

- share per-node partition arithmetic between in-place output and STAG export;
- add borrowed, strided graph/extras views without changing pruning/search;
- replace active graph/search invariant aborts with C++ exceptions;
- join parallel worker exceptions before returning to the host;
- use `std::allocator` for Parlay sequences to avoid process-global allocator
  shutdown ordering failures and retained construction allocations;
- keep benchmark progress output disabled for library construction.

`reference-tests` enables the frozen pre-extraction STAG writer for tests. Tests
compare that independent writer byte-for-byte with the shared writer, then
compare both with the resulting Orion slab. It is not needed by consumers.
Portable builds deliberately omit `-march=native`; Rust search SIMD dispatch is
independent of this C++ construction backend. Graph outputs are checked within a
orion-platform/build; identical floating-point ties across architectures are not promised.
