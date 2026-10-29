# Testing and CI

[README](../README.md)

Run checks with the pinned toolchain from the repository root:

```bash
cargo test --locked -p orion-cli
cargo check --locked -p benchmark --bins
cargo test --locked -p vector --release
cargo test --locked -p orion --release --lib
```

The CLI suite includes real small-data import/build/cache-reload tests, input
format and stream checks, optional-GT behavior, native-u8 compatibility, offline
preflight, and budget enforcement. Automatic graph selection tests cover:

- PA cache/export priority over a Rust cache;
- Rust cache reuse and construction when PA inputs are absent;
- explicit CLI, environment and YAML source strictness;
- native-u8 missing-input behavior and offline estimates;
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
cargo test --locked -p orion-runtime -p orion-cli
python3 orion-py/tests/run.py
```

The Python suite uses generated small STAG graphs and NumPy data. It compares
file/array inputs and CLI IDs/calibration for L2, inner product, cosine and native
u8, then checks strict cache reopening, invalid inputs and serialized concurrent
calls. The runner builds a local extension and CLI; no wheel is published.
