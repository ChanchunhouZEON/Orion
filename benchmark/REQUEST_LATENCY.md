<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
# <img src="../orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Request latency

`latency-sweep` is an unpublished benchmark executable. It calls the public
`Index::search` on borrowed **single-row** queries. Production APIs, SIMD kernels,
prefetch scheduling and `SearchResult` are unchanged. The benchmark workspace
enables Orion's `benchmark` feature, including its existing search counters.

```bash
cargo build --release --manifest-path benchmark/Cargo.toml --bin latency-sweep
benchmark/target/release/latency-sweep --config experiment.json --output results/new-run
```

On hosts with an incompatible repository linker override, set `RUSTFLAGS` to
the required compiler flags and use the system linker. Record the flags with
the experiment. An example explicit configuration (paths relative to this JSON):

```json
{
  "dataset": "example",
  "base": "base.fbin",
  "query": "query.fbin",
  "groundtruth": "truth.ivecs",
  "index": {
    "threads": 8,
    "auto_pin": true,
    "cache_dir": "cache",
    "graph": {"source": "parlayann", "staged_file": "graph.staged"}
  },
  "k": 10,
  "beams": [32, 64, 128],
  "calibration_samples": 200,
  "calibration_l": 200,
  "protocol": {"concurrency": [1, 8], "rounds": 3}
}
```

Set graph capacities/recipe and cascade overrides to match the imported graph
and intended experiment. An existing `snapshot` can replace `staged_file`; the
snapshot's saved recipe wins, with thread, residency and budget overrides from
`index`. Actual effective settings are recorded in `results.json`. Calibration
is explicitly repeated using this configuration's sample count and L. Without a
snapshot, an explicit `parlayann` source and an existing STAG file are required;
automatic or Rust construction is rejected.

## Boundaries and protocol

- A monotonic `Instant` brackets each `Index::search`. This includes validation,
  pool waiting, preparation, scratch acquisition/return, traversal, optional
  scoring, result assembly and external ID mapping. Query decoding, warmup,
  calibration, loading, histogram updates and result destruction are outside.
- Index threads and client concurrency are independent. Clients share one Index
  and its dedicated Rayon pool. This is closed-loop load, not fixed-arrival-rate
  server latency; it does not measure HTTP, Python conversion or network overhead.
- Clients warm up before a common start barrier. Default warmup is at least
  1,000 requests in total and 0.2 s per client. Every round uses a seeded shuffled
  query order, shared by both conditions. A ticket outside the timer distributes
  the next query, preventing fast clients from oversampling an easier partition.
- Default measurement is at least 1 s and 10,000 requests (rounded up across
  clients), increased to cover the whole query set once. The 30 s ceiling is
  checked between requests: it cannot cancel an in-progress search. Insufficient
  samples/time produce `status=incomplete`, never a successful target claim.
- Cache policy is steady-state, without cache flushing. Default cooldown is 1 s
  between conditions. Condition order reverses every other round.
- Before timing, each unique query at each concurrency is checked against the
  ordinary batch API, including scores and external IDs when requested. Recall
  uses the first k ground-truth IDs, respects partial results and counts each
  unique query once. No tie expansion or interpolation is applied.
- Failures abort the run and produce `status=failed`; failed requests are not
  silently dropped. Existing output directories are rejected. Atomic checkpoints
  retain completed rounds if a process is interrupted.

## Reporting and interpretation

Each round records request count, wall time, instrumented QPS, mean, P50/P95/P99,
maximum and a histogram. Histograms have three significant digits and a fixed
one-hour upper bound; overflow is an error. Mean and maximum use raw elapsed
nanoseconds, while percentile bucket bounds have HDR quantization. Clock-zero
samples are counted explicitly. Tail outliers are retained.

The pooled distribution merges samples; it **does not average percentiles**.
The report also retains the per-round P99 range and the median instrumented QPS.
QPS includes harness work between calls and is not historical batch-sweep QPS.
Repeated corpus replay increases sample count, not independent-query count.

`checkout_git_head_at_run` and `rustc_at_run` describe the launch environment,
not the executable's build provenance. Retain the build revision, source diff,
compiler flags and executable SHA-256 separately when copying a binary or
switching branches while an experiment runs.

`protocol.overhead_check=true` adds an untimed closed-loop control per point.
Its throughput estimates instrumentation impact but is a separate run subject
to hardware drift; do not subtract that estimate from request latency. No P99.9
claim or confidence interval is inferred from these short repeated runs.

Tests run with `cargo test --manifest-path benchmark/Cargo.toml --lib --test latency`.
CI checks correctness and statistics, not machine-dependent latency thresholds.
