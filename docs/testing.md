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

## README validation snapshot

The test-count, doctest and line-coverage badges describe the **2026-11-05**
local macOS/AArch64 run on Rust 1.92, including the coverage changes on top of
`97fe271`, including removal of the unused distance buffer. They are dated
measurements; the separate GitHub Actions badges show live `main`-branch status.

| Check | Result | Scope |
|---|---:|---|
| Rust tests | 504 passed, 4 ignored | Workspace, all features, excluding the Python extension |
| Doctests | 42 passed, 1 ignored | Same workspace; counted separately from Rust tests |
| Rust line coverage | 90.20% (24,931 / 27,640 lines) | 204 compiled Rust files; test/example/vendor paths excluded |

The final change adds **42 tests**, removes five unit tests belonging to the
unused distance buffer, and raises coverage from **81.17% to 90.20%**. All
504 tests and 42 doctests pass; four ordinary tests and one doctest remain
explicitly ignored. Default-feature tests and all 63 remaining vector release
tests also pass, with no compiler warnings.

Before removing the legacy module, coverage was 90.17% across 27,785 lines.
The module and its dedicated integration test were removed after confirming
that production search and benchmarks do not use them. No coverage exclusion
was added; the reporting scope and 90% CI gate remain unchanged.

Coverage uses `cargo-llvm-cov 0.6.21`. It includes the Rust DiskANN code and
inline unit tests within source files, but not C++, Python, doctest execution or
x86-only paths absent from this ARM build. The native WASM wrapper is included;
actual WebAssembly execution is checked separately by `wasm-replay`.
Coverage measures executed lines, not correctness or memory safety.

Reproduce from the repository root, running the commands sequentially. Some
legacy DiskANN tests share output filenames within a checkout:

```bash
cargo +1.92 test --locked --workspace --exclude orion-ann-python --all-features
cargo +1.92 test --locked --workspace --exclude orion-ann-python --all-features --doc
rustup component add llvm-tools-preview --toolchain 1.92
RUSTFLAGS='' CARGO_ENCODED_RUSTFLAGS='' cargo +1.92 llvm-cov \
  --locked --workspace --exclude orion-ann-python --all-features \
  --summary-only --json --ignore-filename-regex '/(tests|examples|vendor)/' \
  --fail-under-lines 90 --output-path coverage.json
```

[Machine-readable results, per-module coverage and evidence hashes](results/2026-11-05/validation.json).
Refresh the numeric badges and this snapshot together when rerunning the same
scope. Compiler/platform changes or a different feature set can change counts.
Benchmark timing regressions require controlled repeated experiments separately.

The [coverage workflow](../.github/workflows/coverage.yml) repeats this scope on
an AArch64 runner for pushes and pull requests. It requires at least **90%** line
coverage and retains the JSON report with a per-module job summary. The native
OS/CPU matrix also runs DiskANN integration and platform I/O tests. These checks do not
shrink the reporting scope to meet the threshold.

The coverage expansion adds small-data lifecycle and contract tests for:

- DiskANN file/memory builds, insertion, deletion, search, mmap exports and
  graph/candidate cache round trips;
- all scalar quantization codecs, sparse JL, Hadamard, kernel-trick and RaBitQ
  decoding, malformed metadata, truncated readers and sidecar restoration;
- interruption-aware query streams, ground-truth validation and decoding across
  batch boundaries;
- CLI snapshot restoration against Rust search results, persisted calibration
  and non-overwriting result output;
- async search phase transitions, benchmark baseline parity and diagnostic counters.

These tests caught and fixed little-/big-endian disagreement in deletion files,
out-of-range deletion requests, reused IDs after graph extension, an overlapping
copy in the naive benchmark baseline, overflow in mmap read bounds, and corrupted
UTF-8 paths during preset substitution. SIMD kernels and prefetch scheduling are
unchanged. The baseline copy fix only affects benchmark index preparation.

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
