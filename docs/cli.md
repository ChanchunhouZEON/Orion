# Orion CLI and query inputs

[Back to the README](../README.md#table-of-contents)

`orion-cli` is an independent workspace crate. It depends on the Orion engine,
not on `benchmark`. It supplies the `orion` executable and shared configuration,
input readers, graph loading, preflight, and backend adapters. The `orion-sweep`
executable remains in `benchmark` for repeated experimental runs.

## Anonymous datasets and configuration

A preset is optional. Without one, the resolved dataset name is `anonymous`.
An explicit `anonymous` name has the same meaning. An unknown named preset is
an error, so misspellings do not silently change the experimental configuration.

```bash
cargo build --release -p orion-cli --bin orion
./target/release/orion --base /data/base.fbin --query /data/queries.fbin \
  --metric l2 --k 10 --search-l 100 --output results.jsonl
```

Metric must be supplied for an anonymous dataset; dimension is read from the
base header unless explicitly supplied. The metric/dimension selector supplies
the default cascade. Build parameters use the shared global defaults. Explicit
CLI values override preset values. Native-u8 currently requires 128-dimensional
L2 byte inputs and a ParlayANN graph; it is selected explicitly by the large
SIFT presets or by `--vector-storage u8 --admission native-l2-u8 --rerank none`.
Other inputs retain f32 resident storage by default.

Resolution produces one `SearchPlan`; serialized `vector_storage` and `cascade`
fields are derived from that plan, not stored as independently mutable copies.
The bundled YAML remains `benchmark/configs/sweep.yaml`, embedded into the CLI
at compile time. `--config` selects a custom YAML file. Rebuild after changing
bundled defaults, or pass the edited file with `--config`.

Without a graph-source override, anonymous runs build a Rust graph. A ParlayANN
request reuses a compatible cache first, or imports its STAG file on a cache
miss. Missing both is an error; there is no silent switch to Rust. Missing or
malformed data and incompatible options also produce errors.

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
Use `--no-groundtruth` to disable a preset's GT path. An explicitly supplied but
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
cargo build --release -p benchmark --bin orion-sweep
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

- base, graph slab, reader counters, and the known L2U8 admission sidecar;
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
is a separate loading limit: SIFT10M/100M share a declared 1B-row base but load
only their configured prefixes. An explicit `--base` or `--query` discards the
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
