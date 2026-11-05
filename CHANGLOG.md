<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Changelog

User-visible changes to Orion Astraeus. Entries are grouped by behavior rather
than by commit. The manifests currently use `0.1.0`; the entries below are an
unreleased development baseline, not a dated release announcement.

[README](README.md) · [简体中文](README.zh-CN.md) · [Contributing](CONTRIBUTION.md)

## Unreleased

### Added

- Native API CI matrix entries for Linux ARM64 and macOS x86-64 alongside the
  existing Linux x86-64 and macOS ARM64 checks; OS and CPU badges are separate.

- A shared Rust `Index` API for file and in-memory inputs, graph construction,
  calibration, batch search and persistence, with Python/NumPy and CLI adapters.
- Dynamic vector dimensions with row padding, f32 cascades, native-u8 L2 search
  and automatic runtime SIMD selection on supported AArch64 and x86 CPUs.
- Default in-memory ParlayANN construction through an FFI bridge, including
  construction into Orion graph storage and staged graph import.
- Read-only Rust `QueryBatch` inputs, `SearchOptions` and concurrent searches
  through a shared index. Python retains an input copy before releasing the GIL.
- SHA-256-verified persistence with immutable graph/sidecar objects shared by
  working caches and calibrated snapshot manifests.
- Structured errors shared by Rust, Python and CLI, plus diagnostics and
  WebAssembly search-trace replay.
- Regression tests for DiskANN index lifecycles, sidecar codecs and restoration,
  malformed inputs, query streams, snapshot consistency and platform I/O.
- An AArch64 coverage workflow with a 90% line-coverage floor. Existing safety
  and package validation include Miri, randomized stress, SIMD checks,
  cross-interface comparisons, and clean crate/wheel/sdist installation tests.
- A Chinese README, contributor guidance and this change record.

### Changed

- Public search uses adaptive traversal and requires calibration for the
  requested `k`; missing calibration reports `InvalidState`.
- Shared configuration and persistence orchestration live in the runtime facade;
  algorithm, platform and language-adapter responsibilities remain separate.
- Research counters and ablation controls require the `benchmark` feature.
  The experiment harness has its own Cargo workspace.
- Large dataset reads and persistent writes use block I/O. Linux large
  allocations request transparent huge pages on a best-effort basis.
- README examples, support information, performance comparisons and centered
  badges now share a bilingual presentation with linked validation evidence.

### Fixed

- DiskANN deletion-file endianness and deletion bounds, including rejecting an
  invalid deletion batch before mutation.
- Reused vertex IDs when extending an in-memory DiskANN graph.
- An overlapping slab copy during benchmark-baseline construction.
- Overflowing mmap read ranges and corruption of non-ASCII preset paths.
- Process-global environment mutation in a concurrently executed preset test.

### Removed

- The unused `CacheLineDistanceBuffer` module, public export and dedicated
  tests. Its legacy chunk-size and alignment assumptions caused an AVX-512
  test crash; production search did not use this buffer. Current SIMD kernels
  and prefetch scheduling remain in place.

### Validation

The 2026-11-05 local Rust 1.92/AArch64 snapshot reports **504 passing Rust tests**,
**42 passing doctests**, and **90.20% line coverage** (24,931 / 27,640 lines).
Four ordinary tests and one doctest are explicitly ignored. These are dated
measurements, not a claim of runtime coverage on every supported architecture.
See [the validation scope and evidence](docs/testing.md#readme-validation-snapshot).

Future PQ feature isolation is tracked in the
[contribution guide](CONTRIBUTION.md#planned-follow-up-optional-product-quantization)
and is not implemented by this change record. When creating a release, move the
applicable entries under its version and actual release date, and leave a new
`Unreleased` section for subsequent work.
