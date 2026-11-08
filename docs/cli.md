<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Orion CLI and query inputs

[Back to the README](../README.md#contents)

`orion-cli` is the binary-only `orion` frontend. It uses the optional `cli`
feature of `orion-runtime` for input readers, graph loading, preflight and
backend adapters. Flags and argument parsing live in `orion-cli`;
Python uses the same runtime. Neither frontend depends on `benchmark`. The `orion-sweep`
executable remains in `benchmark` for repeated experimental runs.

The CLI uses `--cache-dir` as its working graph cache. It loads or prepares
the graph for the resolved recipe and calibrates searches from the first query
batch. In Rust and Python, `Index.open` reopens a snapshot published by
`Index.save` and restores the saved calibration. Pass a published snapshot directory to search with its saved configuration
and calibration. Supply the original external base file and a calibrated k:

```bash
orion --snapshot cache/sift-snapshot --base data/base.fbin --query data/query.fbin \
  --k 10 --search-l 96 --threads 8
```

`--snapshot` opens the directory through the runtime’s snapshot checks. A
missing calibration is an error. Search never calibrates on the incoming
query stream in this mode. Thread count and memory budget default to the
saved values. Use `--cache-dir` separately to build or reuse a working graph.

## Anonymous datasets and configuration

The CLI accepts explicit file paths and uses the same `IndexOptions` defaults as
Rust and Python. Dataset shortcuts, YAML and legacy experiment environment
variables are exclusive to `benchmark`; the production CLI has no positional
dataset argument or `--config` option.

```bash
cargo build --release -p orion-ann-cli --bin orion
./target/release/orion --base /data/base.fbin --query /data/queries.fbin \
  --metric l2 --k 10 --search-l 100 --output results.jsonl
```

Metric defaults to L2; dimension comes from the base header. A supplied dimension
must agree with that header. Resident storage follows the input format: bvecs
and u8bin retain native bytes, while fvecs and fbin use f32. Use
`--vector-storage f32` to expand bytes explicitly. Native-u8 search requires L2
and a ParlayANN graph, and automatically selects exact byte admission without
reranking. The f32 cascade is selected from dimension and metric.

Graph defaults come from `BuildConfig`, cache files default to `cache/orion-ann`,
and explicit CLI options override the shared service defaults. Both cached and
new construction recipes use the same validation rules.

The default graph policy is `auto`: a compatible PA cache, then a PA STAG v3
export, then an in-memory ParlayANN build. Disabling the default `parlay` feature
retains the f32 Rust fallback. Explicit `--graph-source
parlayann` and explicit staged paths are strict; native u8 never falls back to
Rust. Existing invalid files and permission errors fail instead of changing the
source. See [Graph sources](graph-sources.md) for precedence and provenance.

```bash
# Preparation alone needs no query or ground-truth inputs.
./target/release/orion --base /data/base.fbin --metric l2 --prepare-only

# Reuse a ParlayANN graph, with a cache independent of query/GT inputs.
./target/release/orion --base /data/base.fbin --query /data/queries.fbin \
  --metric l2 --staged-file /data/graph.staged --cache-dir /data/orion-cache
```

## Optional evaluation

Ground truth is optional. With `--groundtruth truth.ivecs`, the CLI computes
recall alongside search timing and counters. Without it, the summary contains
`"recall": null`; QPS, distance counts, and visited counts remain available.
Omit `--groundtruth` to skip evaluation. An explicitly supplied but
missing or malformed GT file is an error. GT rows must align with query order,
contain at least k IDs, and match the loaded base prefix.

Results are JSONL, one record per query, in input order. They go to stdout or a
new file selected by `--output`. Existing output files are not overwritten.
The final summary goes to stdout. Logs and errors go to stderr. If a stream
fails halfway through, already emitted rows remain; no success summary is emitted.

## Streaming queries

`QuerySource` describes decoded vector batches rather than files. The
`VectorReader<R: Read>` implementation supports fvecs, bvecs, fbin, and u8bin
without requiring `Seek`, a known count, or replay. It works with a file, stdin,
or another blocking `Read` implementation. `count_hint()` is optional.

```bash
cat /data/queries.bvecs | ./target/release/orion \
  --base /data/base.bvecs --query - --query-format bvecs --metric l2 \
  --query-batch-size 1024 --search-l 100 --output results.jsonl
```

The initial calibration sample is retained and then searched exactly once. Later
queries are read in bounded batches. The buffer bound is determined by the larger
of calibration sample size and query batch size; it does not grow with total
stream length. Byte queries widen only the small query buffer, never the base.

Ordinary search consumes a source once. It uses one L (default `max(k, 64)`) and
one pass; `qps` times search calls, while `stream_seconds` includes evaluation,
output, and subsequent input reads after setup/calibration. It does not perform
benchmark warmups or cache flushing.

## Repeatable experiments

```bash
cargo build --release --target-dir target --manifest-path benchmark/Cargo.toml --bin orion-sweep
./target/release/orion-sweep sift --search-list-sizes 48,100,200 \
  --threads 8 --trials 3
```

`orion-sweep` loads a regular query file so each L/trial sees the same input.
It rejects stdin and pipes rather than buffering an unbounded stream. GT is
optional for a sweep; missing recall is printed as `n/a`. Adaptive-search and
matched-state diagnostics that measure GT discovery still require ground truth
and explain that requirement explicitly.

## Memory preflight

`--preflight` shows a comfy-table memory breakdown in a terminal without
loading the base or graph. Redirected stdout remains JSON for scripts. Use
`--preflight-format table` or `--preflight-format json` to choose explicitly. Exceeding `--memory-budget-gib` returns a nonzero exit
status but preserves the complete report. It includes:

- base, graph slab, and the deduplicated storage requirements of selected sidecars;
- bounded query buffers (or replayed query payload), result IDs, and optional GT;
- formulas, phase labels, configured budget, and excess over budget;
- explicit unknown entries for worker scratch, unestimated sidecars, construction
  temporary state, allocator/runtime overhead, and OS overhead.

Unknown entries are not zero. The accounted search lower bound is not a peak-RSS
prediction; construction/import and search have different memory lifetimes.
Streaming input counts can remain unknown, while their buffer sizes are bounded.

Known presets support offline planning before any files have been downloaded:

```bash
./target/release/orion sift1b --preflight
./target/release/orion sift1b --preflight --preflight-format table
./target/release/orion sift1b --preflight --preflight-format json > preflight.json
./target/release/orion-sweep sift1b --preflight --memory-budget-gib 650
```

YAML `num_points` and `num_queries` describe source-file row counts. `max_points`
is a separate loading limit: SIFT100M loads a prefix of the declared 1B-row
base; SIFT10M uses a separately prepared 10M-row file. An explicit `--base` or `--query` discards the
corresponding preset count; a missing custom base needs its own YAML declaration.

Available vector headers must agree with the declared counts and dimensions.
Existing cache/export headers determine graph layout; absent graph files use the
configured degree/extras. Missing files are allowed for planning, but corrupt
files or conflicting metadata fail. Ordinary search and preparation still
require their inputs; a successful estimate does not certify file readiness.

The report includes `metadata_sources` and per-component `source` labels. Without
`--memory-budget-gib` it only reports sizes. With a budget, known accounted bytes
above the limit fail; a lower estimate is not a guarantee that the run will fit.
If query count is unknown for a sweep, replay-buffer sizes are `null`, not zero.
Streaming search still estimates its bounded query batches.

The report also shows `graph_source_policy`, concrete `graph_source`,
`graph_selection_reason`, `graph_action`, and `cache_path`. Offline `PA input
required` reports estimate memory but do not indicate that execution can start.

Offline estimates need explicit shape information when the base is absent:

```bash
orion --base /data/base.u8bin --dimension 128 --max-points 1000000000 --preflight
```

This estimates a byte-backed L2 index without any dataset-specific shortcut.
Readable headers take precedence over the offline point limit, which only caps
how many rows execution uses.
