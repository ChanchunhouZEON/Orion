<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Testing and CI

[README](../README.md)

Run checks with the pinned toolchain from the repository root:

```bash
cargo test --locked -p orion-ann-cli
cargo check --locked --manifest-path benchmark/Cargo.toml --bins
cargo test --locked -p orion-ann-vector --release
cargo test --locked -p orion-ann --release --lib
```

The CLI suite includes real small-data import/build/cache-reload tests, input
format and stream checks, optional-GT behavior, native-u8 compatibility, offline
preflight, and budget enforcement. Automatic graph selection tests cover:

- PA cache/export priority over a Rust cache;
- default in-memory ParlayANN construction and explicit Rust construction;
- explicit CLI, environment and YAML source strictness;
- native-u8 construction, input validation and offline estimates;
- malformed exports and incomplete caches without silent fallback;
- default/custom cache roots, source provenance and older manifests.

These tests use temporary fixtures, not the large experimental datasets.
SIMD tests need CPUs supporting the compiled feature set. Compilation is a
separate check from execution.

[The Orion CI workflow](../.github/workflows/ci.yml) compiles all x86 variants,
runs supported variants, and records skipped runtime coverage. It also includes
Apple Silicon and Python/CLI consistency checks. The AVX2 Python job runs
byte-distance regressions in debug mode with integer overflow checks enabled.
Review the job summaries rather than treating a green
compile as evidence that an unsupported instruction path executed.

No static coverage percentage is claimed. Coverage needs a fresh instrumented
run with an explicit package/test scope; historical percentages are not an
up-to-date guarantee. Benchmark timing regressions require controlled repeated
experiments in addition to correctness tests.

## Python / shared runtime

```bash
cargo test --locked -p orion-ann -p orion-ann-cli
python3 orion-py/tests/run.py
```

The Python suite uses generated small STAG graphs and NumPy data. It compares
file/array inputs and CLI IDs/calibration for L2, inner product, cosine and native
u8, then checks strict cache reopening, invalid inputs and concurrent mixed-k/beam
searches. The runner builds a local extension and CLI; no wheel is published.

Release preparation additionally checks executable Rust configuration/build/cache/
search examples, CLI/Python error-report consistency and panic-boundary recovery.
The `portable-api` CI job compiles without global AVX flags and exercises automatic
L2 dispatch. [Release validation](../.github/workflows/release-check.yml) verifies all ten
Cargo packages (including shared errors and diagnostics), builds an external Rust consumer from the extracted packages,
and installs the CLI. It also checks default ParlayANN construction with f32/u8
in installed wheels and independently rebuilds and installs the sdist. These
checks run on every push and pull request and never publish artifacts.

On 2026-11-01, the macOS ARM64 Rust 1.92 run across all eight public crates
completed with **415 passed and 2 ignored** tests, including doctests. This is
the scope of the README test-count badge, not a claim of Windows/WASM coverage.
DiskANN CSR stride/capacity boundary regressions are included in `portable-api`.

## Diagnostics and WASM

`trace_search` regression tests compare f32/u8 adaptive traces with public search
across beam widths. They also exercise uncalibrated diagnostic traces and, with
the `benchmark` feature, compare their IDs with the non-adaptive baseline.
They check real admission scores,
peel/beam events, budget truncation and error contracts. The pure diagnostics
configuration (`--no-default-features`) tests schema validation and replay limits.
The `wasm-replay` CI job builds the browser module and executes it in Node's real
WebAssembly VM against a freshly generated native trace. Windows and browser-side
graph search are not covered by that job.


The root manifest and lockfile describe the production workspace. The
experiments use their own workspace and `benchmark/Cargo.lock`, and depend on
the local Orion crates with the `benchmark` feature. Open the repository root
in the editor to work with production features; open `benchmark/` as a separate
project for the experimental build. To use the usual ParlayANN default as well,
leave rust-analyzer's Cargo default features enabled.
