<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Contributing to Orion

Orion Astraeus welcomes reproducible bug reports, tests, documentation and
implementation improvements. Start with a small change whose behavior and
validation a reviewer can assess independently.

[README](README.md) · [简体中文](README.zh-CN.md) · [Changelog](CHANGELOG.md) ·
[Architecture](docs/architecture.md) · [Testing and CI](docs/testing.md)

## Set up a checkout

Use the toolchain pinned in [rust-toolchain.toml](rust-toolchain.toml), currently
Rust 1.92. Default graph construction also requires a C++17 compiler. The native
CI matrix covers Linux and macOS on x86-64 and ARM64; see the
[validation status](docs/platforms.md#native-ci-matrix) before choosing explicit
SIMD compiler flags.

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

Update the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md) for user-visible
changes. Keep English and Chinese README examples, numeric badges and benchmark
tables consistent. Do not present planned work as an implemented capability.

Before submitting, inspect `git diff` and the staged paths. Keep datasets, cache
objects, build products, downloaded archives and raw experiment output out of
the commit. Small, reviewed fixtures and result summaries may be appropriate.
Preserve existing copyright and license notices, including upstream notices in
vendored or inherited code. Contributions are distributed under the project's
[MIT license](LICENSE).

## Product quantization

PQ is a regular quantization implementation, available in default and
`--no-default-features` builds without a separate Cargo feature. Its public
module is `quantized_dataset::pq`. Orion owns its slice-based training, packed
u8 codes, checked codec and ADC kernel. It does not use DiskANN's PQ implementation
or ndarray. ADC shares `DistanceStream` scheduling with the other kernels.

Compilation does not trigger training or allocate PQ buffers. Builders train
PQ only when requested, and the existing scalar/JL/RaBitQ/native-u8 search
recipes retain their defaults. Research-only async search and measurements
still require `benchmark`; ordinary PQ APIs do not.

New quantized representations belong under
`orion-core/src/model/dataset/quantized_dataset/`. Official sidecar implementations
use the sealed build, computation, resident-memory and residency-region contracts.
Representation requirements drive preparation, budgeting and optional pinning;
these operations must not build data during a pin request.

Check default, no-default and benchmark builds when changing these boundaries.
Native CI tests PQ training, encoded-data attachment and persistence in ordinary
builds. Keep all-features coverage exercising those paths. Variables called
`pq` can also refer to a **priority queue**, independently of product quantization.
