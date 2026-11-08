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

- ADSampling rotation, admission and distance kernels are available only through
  the opt-in `benchmark` feature; production Rust, CLI and Python configurations
  reject ADS aliases with an explicit unsupported-feature error.

- Resource-aware automatic RAM pinning, enabled by default in Rust, Python and
  CLI. Search preparation detects available memory, cgroup limits and OS lock
  allowances, reserves headroom, and applies one bounded budget to the graph,
  base and required sidecars. OS lock denial falls back to pageable memory.
- Index-owned lock guards across searches, with release before recalibration,
  PQ code-storage replacement and index destruction. Calibrated snapshot open
  reapplies the policy; `auto_pin=false` / `--no-auto-pin` disables automatic locks.
- Benchmark-only trusted cache loading and `--cache-file` selection for current
  content-addressed records and compatible V5/V6 standalone graph caches.
  Graph/sidecar SHA-256 verification is skipped for these experiment paths;
  shape, size and edge checks remain. Explicit selection also avoids scanning
  the base for its cache identity. Production verification remains mandatory,
  including when Cargo feature unification enables `benchmark`.
- Regression coverage for resource budgets, lock denial and ownership lifetimes,
  cache-format compatibility, trusted/verified loader isolation, cross-interface
  configuration errors, and stable cache/snapshot identities during refactoring.

- Production prefetch defaults now match the validated benchmark schedule:
  32-cache-line near windows and bursts, with an 8-line group trigger.
  Rust, Python and CLI consumers use it without benchmark environment overrides.

- Independent slice-based PQ training, encoding and L2/IP ADC admission using
  `DistanceStream`; no ndarray dependency in Orion core and no DiskANN PQ reuse.
- Default best-effort PQ codebook pinning, optional private-file code paging,
  explicit code pinning and authenticated PQ sidecar/snapshot round trips.
- Requirements-driven, scoped RAM pinning with per-dataset selection, a bounded
  best-effort default, required/off modes and shared page-lock lifetimes.
- Cascade precision validation rejects quantized reranking after exact/native-u8
  admission and prevents binary sketches from acting as final rerankers.
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

- Centralized sidecar preparation and automatic pinning in the shared search
  preparation phase, independent of the `cli` feature. Calibration and snapshot
  restoration invoke it automatically; ordinary searches reuse the retained
  guards. Residency checks use the live graph without reopening source files.
- Split runtime configuration by data, graph, cascade, execution and resource
  properties, sharing validation and parsing across adapters. Service defaults,
  cache identity calculation and snapshot execution signatures remain stable.
- Moved CLI parsing and precedence into `orion-cli`; moved dataset presets,
  legacy benchmark environment handling and `sweep.yaml` into the independent
  benchmark workspace. Public Rust/Python/CLI services use explicit inputs and
  shared service defaults, without dataset shortcuts or YAML dependencies.
- Renamed source directories `platform` and `vector` to `orion-platform` and
  `orion-vector`; package/import names remain unchanged. Updated workspace,
  dependency, CI, packaging and documentation paths.
- Updated benchmark Orion runners to use the shared bounded auto-pin policy;
  removed separate graph/query `mlock` calls and unbounded pin budgets.
- Extended CI/package audits to enforce the production/benchmark boundary and
  exercise benchmark cache compatibility without weakening production checks.

- Quantized datasets are grouped by representation family; exact sessions share
  that organization. Product quantization is available without a separate
  Cargo feature; training and buffer allocation remain on demand.
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

- A DiskANN PQ regression test now isolates its generated output in a temporary
  directory, preventing ordinary and coverage test processes from overwriting
  the same `validation.bin` file.

- DiskANN deletion-file endianness and deletion bounds, including rejecting an
  invalid deletion batch before mutation.
- Reused vertex IDs when extending an in-memory DiskANN graph.
- An overlapping slab copy during benchmark-baseline construction.
- Overflowing mmap read ranges and corruption of non-ASCII preset paths.
- Process-global environment mutation in a concurrently executed preset test.

### Removed

- The runtime `driver` module, production dataset shortcuts and bundled sweep
  YAML. Experiment-specific parsing now lives only in the benchmark workspace.
- Legacy flat-cache loading from production services. Compatible historical
  caches remain usable through the explicit trusted benchmark path; production
  caches and snapshots retain authenticated content-addressed objects.

- The unused `CacheLineDistanceBuffer` module, public export and dedicated
  tests. Its legacy chunk-size and alignment assumptions caused an AVX-512
  test crash; production search did not use this buffer. Current SIMD kernels
  and prefetch scheduling remain in place.

### Validation

The 2026-11-08 local Rust 1.92/macOS ARM64 validation passed the workspace,
standalone service, benchmark, warning-free documentation and randomized safety
checks (519 Rust tests and 46 doctests passed; 5 explicitly ignored). Python
debug and optimized profiles each passed 20 interface tests.
Actual WASM replay, Miri and AVX2/AVX-512 cross-compilation checks passed.
All 10 packaged crates passed an external-consumer and CLI-install check; the
macOS ARM64 wheel and independently rebuilt sdist passed fresh-environment
installation checks. Artifact SHA-256 records were retained; nothing was published.
Line coverage is **90.17%** (26,428 / 29,308), above the unchanged 90% CI floor.
Linux and macOS Intel native execution remain checks for their respective CI
hosts; cross-compilation is not evidence of runtime execution on those systems.

The 2026-11-08 paired seven-dataset check reused existing graphs and compared
8-thread, k=10 sweeps at identical beam widths. All 254 matched points preserved
recall, node visits and distance counts. Equal-dataset geometric-mean QPS changed
by +0.79%; each version ran one measured trial per point, so this is a regression
smoke check rather than a significance claim. The following trusted-loader check
also reproduced SIFT search counts with both current and V5 caches. SIMD kernels
and prefetch scheduling were unchanged in this round. Paper results and figures
were not updated.

The 2026-11-05 local Rust 1.92/AArch64 snapshot reports **504 passing Rust tests**,
**42 passing doctests**, and **90.20% line coverage** (24,931 / 27,640 lines).
Four ordinary tests and one doctest are explicitly ignored. These are dated
measurements, not a claim of runtime coverage on every supported architecture.
See [the validation scope and evidence](docs/testing.md#readme-validation-snapshot).
