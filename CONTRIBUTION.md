<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Contributing to Orion

Orion Astraeus welcomes reproducible bug reports, tests, documentation and
implementation improvements. Start with a small change whose behavior and
validation a reviewer can assess independently.

[README](README.md) · [简体中文](README.zh-CN.md) · [Changelog](CHANGLOG.md) ·
[Architecture](docs/architecture.md) · [Testing and CI](docs/testing.md)

## Set up a checkout

Use the toolchain pinned in [rust-toolchain.toml](rust-toolchain.toml), currently
Rust 1.92. Default graph construction also requires a C++17 compiler. Linux
x86-64 and macOS Apple Silicon are tested in CI; see [platform support](docs/platforms.md)
before choosing explicit SIMD compiler flags.

```bash
git clone https://github.com/ChanchunhouZEON/Orion.git
cd Orion
cargo check --locked --workspace --exclude orion-ann-python
```

Python development additionally needs a virtual environment, NumPy and maturin;
follow the [Python setup guide](orion-py/README.md#local-development). Use the
`python` Cargo profile for optimized extension builds: it preserves panic
unwinding so the binding can translate unexpected panics into exceptions.

The [benchmark harness](benchmark/Cargo.toml) has a separate Cargo workspace.
Check it explicitly when changing shared APIs. Application defaults should not
enable its research-only controls through feature unification.

## Choose the owning module

| Area | Owner |
|---|---|
| Public Rust `Index`, input validation, configuration, calibration and persistence orchestration | `orion-runtime` (package `orion-ann`) |
| Graph construction and search algorithms, graph access and official sidecars | `orion-core` |
| Distance kernels, padded layouts, SIMD dispatch and prefetch streams | `vector` |
| Shared allocation and file I/O primitives | `platform` |
| Bundled ParlayANN bridge and DiskANN implementation | `orion-parlay`, `orion-diskann` |
| Error codes and cross-interface error reports | `orion-error` |
| Search observations and browser replay | `orion-diagnostics`, `orion-wasm` |
| Language and command-line adapters | `orion-py`, `orion-cli` |
| Research sweeps and ablations | `benchmark` |

Keep adapters thin and put shared behavior in its owning module. Prefer private
or `pub(crate)` items; document the use case before adding a public export.
Experimental counters and ablation controls belong behind `benchmark`.
Official prefilter, admission and rerank contracts remain sealed.

## Implement and test a change

For a bug, include a regression that reproduces the incorrect behavior. Use
small generated inputs and an independent oracle where possible: scalar
distances, known graph neighborhoods, or a save/open round trip. Large randomized
stress tests belong in separate, explicitly invoked tests rather than slowing
every small edit.

Give generated files a unique path for each test invocation and clean them up
afterward. Avoid mutating process-wide environment variables in parallel tests.

Public APIs need examples and documented input constraints and errors. Explain
alignment, initialized length, aliasing and lifetime assumptions at unsafe
boundaries. Test malformed shapes, integer overflow, truncated input and invalid
state where relevant. A SIMD test must size its inputs from the kernel's
`CHUNK_BYTES` contract rather than assuming a width shared by every architecture.

For changes to search, preserve observable result and error semantics across
Rust, Python and the CLI. Test concurrent read-only searches when changing
shared state. Persistence changes should cover verification failures, object
reuse, interrupted writes and snapshot restoration as appropriate.

Run the following checks sequentially from the repository root. Some legacy
DiskANN tests still share fixture-output filenames, so simultaneous test suites
in the same checkout can interfere with one another.

```bash
cargo fmt --all --check
cargo test --locked --workspace --exclude orion-ann-python
cargo test --locked --workspace --exclude orion-ann-python --all-features
cargo test --locked --workspace --exclude orion-ann-python --all-features --doc
cargo test --locked -p orion-ann-vector --release
cargo check --locked --manifest-path benchmark/Cargo.toml --all-targets
git diff --check
```

For Python or shared runtime changes, also run:

```bash
python3 orion-py/tests/run.py
python3 orion-py/tests/run.py --profile python
```

Use the relevant [WASM checks](orion-wasm/README.md) for replay changes. See
[Testing and CI](docs/testing.md) for Miri, malformed-input tests and stress
test commands. The workflows are the authoritative full validation matrix:

| Workflow | What it checks |
|---|---|
| [Orion CI](.github/workflows/ci.yml) | Native tests, SIMD variants, Miri, malformed input, randomized stress, Python/CLI consistency and WASM |
| [Rust coverage](.github/workflows/coverage.yml) | At least 90% native AArch64 Rust line coverage, with report artifacts |
| [Release validation](.github/workflows/release-check.yml) | Packaged Rust consumers, wheel/sdist builds and clean installation tests; does not publish |

Compiling an AVX-512 binary does not establish that it ran on AVX-512 hardware.
Include the runtime-support result when reporting a SIMD check. Coverage is also
not a memory-safety guarantee. Keep the reporting scope intact; refresh numeric
README badges and their [evidence record](docs/testing.md#readme-validation-snapshot)
together after a new measurement.

## Measure performance changes

Follow the [measurement protocols](docs/benchmarks.md). Record the revision,
compiler flags, CPU/OS, thread count and placement, dataset and graph identity,
`k`, recall definition, warmup, timing boundaries and repetitions. Compare QPS
at equivalent measured recall and retain the per-run results, not only a peak.

Search timing must distinguish loading, construction and calibration from the
search itself. A performance claim should link to reproducible evidence. Do not
replace paper figures or published comparison tables as a side effect of a
code change; review data updates separately.

## Prepare a pull request

Describe the problem, the resulting behavior and the checks run. Call out public
API or persistence-format changes and any migration steps. Keep commit subjects
within 50 characters and use the body for module-specific details. Sign commits
when your signing setup is available.

Update the `Unreleased` section of [CHANGLOG.md](CHANGLOG.md) for user-visible
changes. Keep English and Chinese README examples, numeric badges and benchmark
tables consistent. Do not present planned work as an implemented capability.

Before submitting, inspect `git diff` and the staged paths. Keep datasets, cache
objects, build products, downloaded archives and raw experiment output out of
the commit. Small, reviewed fixtures and result summaries may be appropriate.
Preserve existing copyright and license notices, including upstream notices in
vendored or inherited code. Contributions are distributed under the project's
[MIT license](LICENSE).

## Planned follow-up: optional product quantization

This is a design plan for the next change, **not an available feature yet**.
The current default cascade does not require product quantization (PQ), but PQ
code and types are still compiled through several core and DiskANN modules.

The intended boundary is a **default-off `pq` feature**. The implementation should:

1. Gate product-quantizer training, codebooks, encoded data, ADC computation and
   PQ-only serialization in `orion-core` and `orion-diskann`.
2. Gate their associated builder options/results, index fields, exports and
   `PQScratch` storage together; do not leave the default build allocating or
   training PQ data. Audit the benchmark-only async path separately.
3. Forward the feature only through crates that actually need it. `benchmark`
   and `pq` serve different purposes; enabling general experiment controls must
   not automatically enable PQ. Only PQ-specific targets require both.
4. Keep scalar quantization, JL, kernel-trick, RaBitQ, native-u8 search and normal
   graph construction independent of this feature. Variables named `pq` that
   refer to a **priority queue** are unrelated and must remain available.
5. Test default, no-default and PQ-enabled builds, plus benchmark combinations.
   Retain PQ training/round-trip tests in an explicitly enabled CI job and give
   each test invocation unique temporary output paths. Keep all-features
   coverage exercising PQ rather than excluding it to improve a badge.

Acceptance requires unchanged default search results, no PQ-only public types
or storage in the default path, and successful PQ-enabled construction,
encoding and persistence tests. Record measured build/test-time and dependency
changes rather than assuming that all numeric dependencies are PQ-specific.
