# Orion: Admission-Aware Graph Search

[![Cross-platform smoke test](https://img.shields.io/badge/Colima%20x86__64-compile--verified-success?logo=docker&logoColor=white)](benchmark/scripts/ci_smoke_sift.sh)
[![Tests](https://img.shields.io/badge/tests-268%20passing-success?logo=rust&logoColor=white)](docs/testing.md)
[![Coverage](https://img.shields.io/badge/line%20coverage-85%25%20(ours)-success?logo=codecov&logoColor=white)](docs/testing.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Rust](https://img.shields.io/badge/rust-1.86.0-orange?logo=rust&logoColor=white)](rust-toolchain.toml)

[![Apple Silicon NEON](https://img.shields.io/badge/Apple%20Silicon-NEON-success?logo=apple&logoColor=white)](docs/platforms.md)
[![x86_64 AVX--512](https://img.shields.io/badge/x86__64-AVX--512-success?logo=intel&logoColor=white)](docs/platforms.md)
[![PyO3 binding](https://img.shields.io/badge/PyO3-Py%203.9%2B-blue?logo=python&logoColor=white)](orion_py/)
[![Rust LOC](https://img.shields.io/badge/Rust-40k%20LOC-orange?logo=rust&logoColor=white)](#)

Orion is an in-memory approximate nearest neighbor engine in Rust. It combines
an admission-aware search loop, a phased graph layout, and selectable distance
precision to reduce search work. Graphs can be imported from ParlayANN or built
in-process with the Rust DiskANN builder.

The current implementation supports f32 cascades, native-u8 SIFT search,
AArch64 NEON and x86 SIMD kernels, and billion-vector in-memory runs. Linux
large allocations receive best-effort transparent huge-page advice. The
standalone CLI is separate from the benchmark harness.

## Contents

- [Quick start](#quick-start)
- [How search works](#how-search-works)
- [Graph selection and caches](#graph-selection-and-caches)
- [Measured performance](#measured-performance)
- [Large datasets](#large-datasets)
- [Reproducing a sweep](#reproducing-a-sweep)
- [Development and documentation](#development-and-documentation)
- [License](#license)

## Quick start

Use the pinned **Rust 1.92** toolchain and run Cargo from the repository root so
[the target configuration](.cargo/config.toml) applies. SIMD requirements and
portable x86 build flags are in [Platforms](docs/platforms.md).

```bash
rustup toolchain install 1.92
cargo build --locked --release -p orion-cli --bin orion
cargo build --locked --release -p benchmark --bin orion-sweep
```

With your own data, a preset and ground truth are optional. Dimension is read
from the base header; specify the metric explicitly. Supported input formats
are `fvecs`, `bvecs`, `fbin`, and `u8bin`.

```bash
./target/release/orion \
  --base /data/base.fbin --query /data/query.fbin --metric l2 \
  --k 10 --search-l 100 --output results.jsonl
```

This is an `anonymous` dataset. With no PA graph available, the default f32 path
builds a Rust graph and caches it. Add `--groundtruth /data/gt.ivecs` to measure
ID recall; otherwise recall is absent. `--query - --query-format bvecs` accepts
a query stream on stdin. See [CLI and inputs](docs/cli.md).

Presets are in [sweep.yaml](benchmark/configs/sweep.yaml). They provide paths,
metric, dimension, construction settings, and cascade overrides:

```bash
./target/release/orion sift --print-config
./target/release/orion sift --prepare-only
./target/release/orion-sweep sift --k 10 --threads 8 --trials 3

# Plan SIFT1B before downloading data; missing inputs are labeled assumptions.
./target/release/orion sift1b --preflight --preflight-format table
```

`--prepare-only` builds/imports the graph cache without queries or calibration.
`--config path.yaml` selects an alternative configuration; bundled defaults are
embedded at compile time. Ordinary `orion` consumes queries once; `orion-sweep`
loads replayable queries and runs all requested L values/trials in one process.

## How search works

1. **Graph preparation:** import a STAG v3 graph or build a Rust graph, then
   store local, remote, and extra candidate regions in a `PhasedGraph` slab.
2. **Calibration:** sample queries to set convergence and early-exit parameters.
3. **Search:** track admissions into the candidate queue, switch traversal
   phases as the neighborhood converges, and apply the selected
   prefilter → admission → rerank cascade.
4. **Evaluation:** optionally compare final IDs with ground truth. Graph loading,
   preparation, calibration, and recall evaluation are outside sweep search timing.

Native-u8 SIFT uses `none → native-l2-u8 → none`: original byte coordinates
serve both admission and final ranking. It avoids an f32 base and a duplicate
quantized sidecar. Generic f32 paths use the metric/dimension selector, followed
by preset and CLI overrides. See [Architecture](docs/architecture.md) and
[Configuration](docs/configuration.md).

Convergence and early exit are approximate search decisions. Their empirical
recall depends on the graph, data, calibration, and beam width.

## Graph selection and caches

The default policy is **auto**:

| Policy | Behavior on startup |
|---|---|
| `auto` | Valid PA cache → available PA staged export → Rust cache/build for f32 |
| `parlayann` | PA cache or staged export required; never substitutes Rust |
| `rust` | Rust cache/build only; never imports PA |

An explicit `--staged-file` also selects PA strictly. CLI and legacy environment
settings override YAML; bundled presets use `source: auto`. **Native u8 never
falls back to Rust.** Offline preflight can estimate its memory without PA
inputs, but execution requires a compatible cache/export.

```bash
# Pin a source for reproducible comparisons.
./target/release/orion-sweep sift --graph-source parlayann --threads 8 --trials 3
./target/release/orion-sweep sift --graph-source rust --threads 8 --trials 3
```

To generate exports, use the [STAG-capable ParlayANN fork](https://github.com/ChanchunhouZEON/ParlayANN-staged)
and [preparation scripts](benchmark/scripts/README.md). `PA_ROOT` defaults to
`../ParlayANN`. A validated cache does not require the original export to remain
on disk. Corrupt/incompatible caches and existing malformed exports fail rather
than trigger fallback.

Effective source is part of cache identity. Default directories are
`cache/orion_parlayann` and `cache/orion`; `--cache-dir` overrides the directory,
not source separation. Printed configuration, preflight and new cache manifests
record the concrete source. [Graph sources](docs/graph-sources.md) explains
precedence, provenance, legacy cache compatibility, and builder differences.

## Measured performance

These dated results describe specific builds and protocols. They do not replace
or regenerate the existing paper figures. Complete summaries, measured brackets,
and provenance are in [Benchmarks](docs/benchmarks.md).

**Six datasets, k=100, 2026-10-27:** local Apple Silicon/macOS, 8 threads,
3 trials per parameter, PA-derived Orion graphs. Values below are Orion QPS
change relative to ParlayANN, estimated by linear interpolation over measured
recall frontiers. `—` means a target was outside a measured range.

| Dataset | R=0.95 | R=0.97 | R=0.98 | R=0.99 |
|---|---:|---:|---:|---:|
| `sift` | — | +20.99% | +6.27% | -5.90% |
| `glove25` | +58.03% | +31.43% | +21.47% | +13.75% |
| `glove100` | +23.65% | +2.16% | -13.99% | — |
| `gist` | +191.75% | — | — | — |
| `deep10m` | +11.09% | +12.19% | +8.40% | -5.50% |
| `msmarco_bert_1M` | +165.52% | — | — | — |

There are gains across much of the panel, alongside high-recall losses on
SIFT, DEEP10M and GloVe100. The older k=10 comparison is documented separately
because its ParlayANN measurements were historical rather than paired reruns.

**SIFT1B, k=10, 2026-10-27:** Xeon Platinum 8358P, 8 physical cores, 3 trials,
Orion native-u8 with THP. The fastest measured points meeting R≥.95, .97 and .99
were respectively **+2.45%, +20.61% and +12.87%** versus ParlayANN. At R≥.90,
Orion was **9.82% slower**. Actual recalls differ; ParlayANN retains its original
tie-aware metric while Orion reports ID recall. This is a threshold comparison,
not a strict identical-recall claim.

**Graph quality matters:** the 2026-10-29 study runs the same Orion binary on
Rust-built and PA-built graphs for six datasets at k=10 and k=100. PA graphs
usually deliver higher QPS at comparable recall; GloVe25 is close. On raw
MS-MARCO the current Rust builder's L2 objective differs from the PA MIPS recipe,
so its gap is not an isolated builder comparison. See [graph-source results](docs/benchmarks.md#same-engine-different-graph-sources).

## Large datasets

SIFT10M/100M/1B presets select native-u8 storage. For SIFT1B with dimension 128,
R=64 and extra=16, the resident base is about **119.21 GiB**, the aligned graph
slab **357.63 GiB**, and reader counters **7.45 GiB**. Their sum, **484.29 GiB**,
is a lower bound, not peak RSS. Query buffers, scratch, runtime and construction
headroom must also fit.

A graph cache occupies about **384 GB logical disk space**, separately from the
128 GB byte base and source STAG export. Filesystem allocation/quota can differ
from logical sizes. [Large datasets](docs/large-datasets.md) covers memory,
block I/O, storage planning, and import-before-search workflows.

Linux large aligned allocations use `MADV_HUGEPAGE`; actual coverage depends on
the kernel and available memory and must be measured. This does not allocate
reserved 1 GiB hugetlb pages, and the hint is inactive on macOS.

## Reproducing a sweep

Pin graph source, hardware/thread placement, data order, k, L schedule, compiler
flags, and timing protocol. The recent local results used the following
prefetch overrides; these are **experimental settings, not compiled defaults**:

```bash
export ORION_DSTREAM_LA_Q=32
export ORION_DSTREAM_LA_TRUTH=32
export ORION_DSTREAM_SINK_BURST=32
export ORION_DSTREAM_BURST_TRIGGER_LINES=8
export ORION_DSTREAM_BURST_AHEAD_VECTORS=4
export ORION_RERANK_FACTOR=2

./target/release/orion-sweep gist --graph-source parlayann \
  --k 100 --threads 8 --trials 3 --print-config
./target/release/orion-sweep gist --graph-source parlayann \
  --k 100 --threads 8 --trials 3
```

Record the actual source and cache identity before comparing runs. Equal R,
construction L and alpha do not imply equal graph topology. Cache eviction is
best effort, and native engine timing/cache policies are not automatically
identical. Avoid interpreting changes between unpaired historical runs as a
causal speedup.

## Development and documentation

```bash
cargo test --locked -p orion-cli
cargo check --locked -p benchmark --bins
cargo test --locked -p vector --release
```

[Test coverage and CI](docs/testing.md) describes regression checks, coverage
measurement, and hardware execution in CI.

| Guide | Contents |
|---|---|
| [CLI and inputs](docs/cli.md) | Anonymous data, streams, optional GT, preflight |
| [Configuration](docs/configuration.md) | Presets, precedence, cascade, tuning units |
| [Graph sources](docs/graph-sources.md) | Auto selection, strict modes, cache provenance |
| [Large datasets](docs/large-datasets.md) | Native u8, block I/O, SIFT1B memory/storage |
| [Benchmarks](docs/benchmarks.md) | Dated results and comparison methods |
| [Historical figures](docs/historical-benchmarks.md) | Existing figures and data, unchanged |
| [Architecture](docs/architecture.md) | Graph layout and admission-aware search |
| [Optimizations](docs/optimizations.md) | Kernel/search implementation notes |
| [Ablation study](docs/ablation-study.md) | Cascade and adaptive-search diagnostics |
| [Platforms](docs/platforms.md) | SIMD builds, Linux THP, portability |
| [Testing](docs/testing.md) | Regression checks and CI scope |
| [Code structure](docs/code-structure.md) | Workspace crate responsibilities |

## License

[MIT](LICENSE). Copyright Chanchunhou; inherited Microsoft DiskANN files retain
their original Microsoft copyright notices.
