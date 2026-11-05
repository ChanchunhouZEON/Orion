<h1 align="center"><img src="orion-core/assets/orion-astraeus.png" alt="" width="64" align="center" /> Orion Astraeus: Admission-Aware Graph Search</h1>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md" lang="zh-CN">简体中文</a></p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img alt="Safety tests: Miri and adversarial inputs" src="https://img.shields.io/badge/safety-Miri_%2B_adversarial_tests-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="orion-runtime/src/guide/cache.md"><img alt="Verified persistence: SHA-256" src="https://img.shields.io/badge/integrity-SHA--256-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/platforms.md"><img alt="Tested platforms: Linux x86-64 and macOS ARM64" src="https://img.shields.io/badge/platforms-Linux_x86__64_%7C_macOS_ARM64-377EAD?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="#examples"><img alt="Rust and Python APIs" src="https://img.shields.io/badge/API-Rust_%2B_Python-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/platforms.md"><img alt="Automatic runtime SIMD dispatch" src="https://img.shields.io/badge/SIMD-auto_dispatch-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="rust-toolchain.toml"><img alt="Rust 1.92" src="https://img.shields.io/badge/Rust-1.92-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-377EAD?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="docs/testing.md#readme-validation-snapshot"><img alt="Rust tests: 504 passed" src="https://img.shields.io/badge/Rust_tests-504_passed-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/testing.md#readme-validation-snapshot"><img alt="Doctests: 42 passed" src="https://img.shields.io/badge/doctests-42_passed-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/testing.md#readme-validation-snapshot"><img alt="Rust line coverage: 90.20%" src="https://img.shields.io/badge/Rust_line_coverage-90.20%25-A77A2E?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/ci.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/coverage.yml"><img alt="Coverage check" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/coverage.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/release-check.yml"><img alt="Release validation" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/release-check.yml/badge.svg?branch=main" /></a>
</p>

Orion Astraeus (**Orion**) is a **Vamana-style graph algorithm and in-memory
approximate nearest neighbor engine**, with Rust and Python APIs. Its
admission-aware search tracks which candidates enter the search queue, adapts
graph traversal as the neighborhood converges, and uses a distance cascade to
spend less work on unpromising candidates.

Give Orion your vectors: it builds a graph with bundled ParlayANN, selects a
search recipe from the metric and dimension, and dispatches to the SIMD backend
your CPU supports. Calibrate with representative queries, then search through a
shared index. No dataset preset, prebuilt graph, ground truth or manual SIMD
flags are needed for the library APIs.

**Built for embedded search:** f32 and native-u8 inputs, dynamic dimensions,
concurrent queries, structured errors, and verified snapshots that reuse cached
graph and sidecar objects.

[Performance](#performance) · [Examples](#examples) ·
[Support](#support) · [CLI](#cli) · [Benchmarks and development](#benchmarks-and-development)

## Performance

**Highest measured QPS at Recall@10 ≥ 0.95 across the paper's six-dataset wider
baseline panel**, extended here with billion-vector SIFT1B and high-dimensional
Wiki-Ada-1M results. All columns use **8 threads, k=10, and recall ≥ 0.95**;
we select the fastest measured median meeting the target, without interpolation.
K = 1,000 QPS; the first six columns retain the paper's rounding.

| Engine · QPS | SIFT1M | GloVe-25 | GloVe-100 | GIST | Deep10M | MS-MARCO | SIFT1B | Wiki-Ada-1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Orion** | **341K** | **383K** | **47K** | **48K** | **170K** | **52K** | **33,897** | **143,954** |
| ParlayANN | 257K | 247K | 37K | 29K | 143K | 38K | 33,086 | 151,258 |
| Microsoft Vamana (DiskANN) | 91K | 151K | 12K | 7K | 39K | — | 10,371 | 39,915 |
| USearch HNSW | 59K | 102K | — | — | 21K | 8K | Not tested | Not tested |
| LanceDB HNSW (native) | 1K | 1K | — | — | — | — | Not tested | Not tested |
| HNSW (hnswlib) | 51K | 139K | — | — | 21K | 6K | Not tested | Not tested |
| FAISS HNSW+SQ | 54K | — | — | — | 15K | 5K | Not tested | 17,730 |
| FAISS IVF-Flat | 9K | 33K | 3K | — | 2K | — | Not tested | Not tested |
| Annoy | 1K | 3K | — | — | — | — | Not tested | Not tested |

The six-dataset panel runs on Apple M4 Max. Wiki-Ada-1M contains 1M × 1,536-D
normalized embeddings on Apple Silicon; SIFT1B contains 1B × 128-D native-u8
vectors on Xeon Platinum 8358P, Linux/AVX-512 with Orion transparent huge pages.
Loading, graph construction and calibration are outside search timing.

`Not tested` marks combinations without retained measurements. `—` marks
measured curves below the target or a non-comparable result. Microsoft Vamana
is the DiskANN in-memory implementation; its MS-MARCO result is excluded for
metric mismatch. FAISS IVF-PQ stays below 0.62 recall on the six-dataset panel
and is omitted. **Bold highlights Orion.** [Exact QPS, actual recalls and run metadata](docs/results/2026-11-05/readme-performance.json) ·
[Experiment protocols](docs/benchmarks.md) · [SIFT1B resource planning](docs/large-datasets.md).

## Examples

Source builds require **Rust 1.92 and a C++17 compiler** for the bundled graph
builder.

### Rust: one dependency, build and search

In an application next to your `Orion` checkout:

```toml
[dependencies]
orion-ann = { path = "../Orion/orion-runtime" }
```

```rust
use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions};

fn main() -> orion_ann::Result<()> {
    // Replace this small example with your row-major embeddings.
    let base: Vec<f32> = (0..1_024 * 128)
        .map(|i| ((i * 73 + i / 128 * 11) % 251) as f32)
        .collect();
    let queries = &base[..32 * 128];
    let mut index = Index::new(
        BaseInput::from_f32(&base, 128)?,
        IndexOptions::default(),
    )?;

    // Use representative application queries in production; no ground truth needed.
    index.calibrate(queries.to_vec(), 10, None)?;
    let result = index.search(
        QueryBatch::from_f32(queries, 128)?,
        &SearchOptions::new(10, 100),
    )?;
    for neighbors in result.rows() {
        println!("{neighbors:?}");
    }
    Ok(())
}
```

Use `BaseInput::from_file("base.fbin")` for file input, or
`BaseInput::from_u8(&bytes, dimension)?` for native bytes. Orion handles row
padding internally. Search borrows Rust queries and takes `&self`, so an
`Arc<Index>` can serve concurrent requests through its shared worker pool.
Calibrate once per requested `k`; missing calibration returns `InvalidState`.
[More Rust examples](orion-runtime/README.md) · [Configuration guide](orion-runtime/src/guide/configuration.md).

### Python: NumPy in, neighbor IDs out

From the repository root, in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.24,<3' 'maturin>=1.9,<2'
maturin develop --manifest-path orion-py/Cargo.toml --profile python
```

```python
import numpy as np
from orion_ann import Index

rng = np.random.default_rng(42)
base = rng.random((1_024, 128), dtype=np.float32)
queries = rng.random((32, 128), dtype=np.float32)

index = Index.from_array(base, metric="l2", threads=8)
index.calibrate(queries, k=10)
result = index.search(queries, k=10, search_list_size=100)
print(result.ids[0, :result.counts[0]])
```

Use `metric="cosine"` for embeddings that need normalization, or `metric="ip"`
for raw inner product. C-contiguous `float32` and `uint8` arrays are accepted;
native bytes use L2. Python copies inputs before releasing the GIL, and
concurrent searches share the configured Rust worker pool.

### Load files and keep a calibrated snapshot

```python
index = Index.from_file("base.fbin", metric="l2", cache_dir="cache/build")
index.calibrate("queries.fbin", k=10)
result = index.search("queries.fbin", k=10, search_list_size=100)

snapshot = index.save("snapshots/k10")  # Choose a new directory.
restored = Index.open(snapshot, base="base.fbin", threads=8)
result = restored.search("queries.fbin", k=10, search_list_size=100)
```

A snapshot is a small manifest: the graph and sidecars stay in a shared,
SHA-256-addressed object store. Calibration is restored on open. Keep the
external base and object store with the manifest; base vectors are not embedded
in the snapshot. Rust offers the same `Index::save` / `Index::open` workflow.
[Persistence guide](orion-runtime/src/guide/cache.md) · [Python guide](orion-py/README.md).

## Support

| Capability | Available today |
|---|---|
| Native platforms tested in CI | Linux x86-64 and macOS Apple Silicon |
| APIs | Rust `Index`; Python 3.9+ / NumPy; standalone CLI |
| Metrics | f32 L2, cosine and inner product; native-u8 L2 |
| Dimensions | Runtime dimensions with automatic row padding |
| SIMD | Automatic scalar, NEON/dot-product, AVX2/FMA and AVX-512 selection, subject to CPU support and row layout |
| Graph construction | Bundled in-memory ParlayANN by default; f32 Rust DiskANN builder; STAG import |
| Inputs | Packed Rust slices, NumPy arrays, `fvecs`, `bvecs`, `fbin`, `u8bin`; CLI query streams |
| Persistence | SHA-256-verified graph/sidecar objects and calibrated snapshots |
| Error handling | Shared error codes and details across Rust, Python and CLI; Python converts unwinding panics into exceptions |
| Safety validation | Miri, malformed-input tests, randomized stress and cross-interface consistency checks |
| Large allocations | Best-effort Linux transparent huge-page advice |
| Browser tooling | WebAssembly trace replay through `orion-wasm` |

No application ISA flags are needed for runtime SIMD dispatch. Explicit
architecture flags remain available for a fixed deployment target; see
[Platforms](docs/platforms.md). For a lean f32 Rust build without the C++ builder,
use `default-features = false`. [Graph selection](docs/graph-sources.md) explains
cache/import precedence and explicit backend selection.

For usage questions or reproducible bug reports, use
[GitHub Issues](https://github.com/ChanchunhouZEON/Orion/issues).
Include the platform, package revision, metric/dimension and structured error
report when relevant.

## CLI

Build the CLI when you want file-to-file search without an application wrapper:

```bash
cargo build --locked --release -p orion-ann-cli --bin orion
./target/release/orion \
  --base /data/base.fbin --query /data/query.fbin --metric l2 \
  --k 10 --search-l 100 --output results.jsonl
```

The CLI builds or reuses a graph and calibrates from the query stream.
Dataset presets and ground truth are optional. Add `--snapshot DIR` to restore
calibration, `--groundtruth /data/gt.ivecs` to evaluate recall, or
`--query - --query-format bvecs` to read stdin. `--prepare-only` builds the working
graph cache; `--preflight` estimates memory. [CLI guide](docs/cli.md).

## Benchmarks and development

The experiment harness has its own Cargo workspace and opts into the
`benchmark` feature. From the repository root:

```bash
cargo build --locked --release --target-dir target \
  --manifest-path benchmark/Cargo.toml --bin orion-sweep
./target/release/orion-sweep sift --graph-source parlayann \
  --k 10 --threads 8 --trials 3
```

Configure dataset paths in [sweep.yaml](orion-runtime/configs/sweep.yaml).
The sweep loads once and runs its L schedule and trials in one process.
Record hardware, CPU placement, graph source, build flags and timing settings
when reproducing a comparison. [Benchmark scripts](benchmark/scripts/README.md) ·
[Measurement protocols](docs/benchmarks.md).

```bash
cargo test --locked -p orion-ann -p orion-ann-cli
cargo test --locked -p orion-ann-vector --release
cargo doc --locked -p orion-ann --no-deps --open
```

[Architecture](docs/architecture.md) · [Optimizations](docs/optimizations.md) ·
[Testing and CI](docs/testing.md) · [Code structure](docs/code-structure.md) ·
[Ablation studies](docs/ablation-study.md) · [Historical figures](docs/historical-benchmarks.md).

## License

[MIT](LICENSE). Copyright Chanchunhou; inherited Microsoft DiskANN files retain
their original Microsoft copyright notices.
