# Configuration

[← Back to main README](../README.md)


`benchmark/configs/sweep.yaml` is the **single source of truth** for every benchmark binary (`orion`, `benchmark`'s `--algorithms build-profile` / `memory-profile`). Per-dataset entries set the data paths, the base-graph source (`parlayann` import or `rust` in-process build), and the build params (`α / R / L_build / max_extra / window_size`). The CLI cascade (`--prefilter / --admission / --rerank`) is layered on top — see [Search-Phase Optimizations](optimizations.md#search-phase-optimizations) for how the three search-time axes are selected independently from the build-time params.

### Per-dataset PA-aligned build params

Every entry mirrors the recipe in `../ParlayANN/algorithms/vamana/scripts/<ds>` verbatim — the same `R / L_build / α / num_passes` PA's published results use. This keeps both engines on **bit-identical graph topology**, so the head-to-head numbers in the [Benchmark Results](#benchmark-results) section isolate search-side QPS differences from build-side graph-quality differences.

| Dataset           | Dim      | R   | L_build | α    | num_passes | PA reference script                                             |
| ----------------- | -------- | --- | ------- | ---- | ---------- | --------------------------------------------------------------- |
| `sift`            | 128      | 64  | 128     | 1.15 | 2          | `vamana/scripts/sift`                                           |
| `glove25`         | 32       | 100 | 200     | 1.0  | 2          | `vamana/scripts/glove25`                                        |
| `glove100`        | 100      | 100 | 200     | 1.0  | 2          | `vamana/scripts/glove100`                                       |
| `gist`            | 960      | 100 | 200     | 1.1  | 2          | `vamana/scripts/gist`                                           |
| `deep10m`         | 96 → 128 | 64  | 128     | 1.05 | 2          | `vamana/scripts/deep10M`                                        |
| `fashion-mnist`   | 784      | 40  | 80      | 1.1  | 2          | `vamana/scripts/fashion`                                        |
| `msmarco_bert_1M` | 768      | 64  | 128     | 1.0  | 1          | `vamana/scripts/msmarco_websearch`                              |
| `wiki_ada_1M`     | 1536     | 100 | 200     | 1.05 | 2          | high-D MIPS shape (no direct PA recipe — mirrors `OpenAIArXiv`) |

`max_extra = 16` is the hard cap on per-node `extras` post-partition (PhasedGraph slot stride = `HEADER(4) + max_degree + max_extra`, cache-line aligned). `window_size = 5` for convergence detection. Both are dataset-uniform; per-dataset overrides exist but none of the 8 datasets currently override them.

### File structure (abridged)

The top of `sweep.yaml` defines defaults; per-dataset blocks override only what differs. The `base_graph.staged_file` path resolves `${PA_ROOT}` against the `PA_ROOT` env var at load time (default `../ParlayANN`) so user-specific absolute paths stay out of the committed file.

```yaml
defaults:
  diskann:                              # vanilla DiskANN baseline (build-profile / memory-profile only)
    alpha: 2.0
    graph_degree: 64
    build_search_list_size: 100
  orion:                               # cross-dataset Orion defaults — overridden per dataset
    alpha: 1.2
    graph_degree: 64
    build_search_list_size: 100
    max_extra: 16
    window_size: 5
    metric: l2                          # legacy 4-way enum; see note below
  sweep:
    search_list_sizes: [16, 20, 24, 32, 40, 48, 56, 64, 80, 100, 128, 160, 200, 256, 512, 768, 1024]
    threads: 8
    trials: 1

datasets:
  sift:                                 # PA's vamana/scripts/sift verbatim
    dimension: 128
    paths: { base: "data/sift/sift_base.fvecs", query: "...", groundtruth: "..." }
    base_graph:
      source: parlayann                 # imports PA's .staged export
      staged_file: "${PA_ROOT}/data/sift1m/sift1m_ex${max_extra}.staged"
    orion:
      alpha: 1.15                       # PA's α for SIFT
      graph_degree: 64                  # PA's R
      build_search_list_size: 128       # PA's L_build
      metric: l2-q
  # ... gist, glove25, glove100, deep10m, fashion-mnist, msmarco_bert_1M, wiki_ada_1M ...
```

When `base_graph.source: parlayann`, the `orion.{R, α, L_build, max_extra}` params are **advisory** — they drive cache-filename conventions and the per-query calibrator, but the actual graph topology comes from PA's `.staged` export. The importer lives at `benchmark/src/runner/parlayann_bridge.rs`; first-run cost is ~3 s per dataset, cached to `cache/orion_parlayann/*.pgraph` thereafter. Set `source: rust` to build in-process via `build_diskann_index` using `orion.{α, R, L_build}` instead — used by the [Build Overhead](#build-overhead) + [Memory](#memory) profiles where we measure our own Vamana build time directly.

### `metric` field vs the cascade

The per-dataset `orion.metric` field carries the legacy 4-way enum (`l2 / l2-q / mips / mips-q`) consumed by the `build-profile` / `memory-profile` harnesses inside the `benchmark` binary. The production `orion` bin **does not read this field** — it picks the search-time triple from `Cascade::default_for_dataset` in `benchmark/src/bin/orion.rs`, which can be overridden per-invocation via `--prefilter / --admission / --rerank`. See [Per-dataset default cascade](#per-dataset-default-cascade) for the production mapping.

### Auto-Calibration

Convergence parameters (`threshold`, `early_exit_limit`) are automatically derived at build time via `calibrate()`. The calibrator runs warmup queries with full-graph search (no convergence, no early exit) and analyzes:

1. **Per-step admission rate curve** — the threshold is set at the inflection point where admission rate drops sharply.
2. **Gap distribution** between consecutive useful admissions — the early exit limit is set from the P90 tail gap.

This eliminates manual tuning and adapts to dataset characteristics automatically.
