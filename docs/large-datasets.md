# Large datasets on the in-memory Orion path

The shared CLI used by `orion`, `adaptive`, `adaptive_ee_ablation`, and
`matched_state` now loads the base into one 64-byte-aligned f32 allocation
(with the existing SIMD tail padding). Index loading takes ownership of that
allocation. Cached loads and PA imports do not create a second f32 base.
Search scoring, calibration, and local/remote/extra selection are unchanged.

## Inputs and prerequisites

Supported base/query inputs (selected by suffix, or `--base-format` and
`--query-format`):

| Format | Layout, little endian |
| --- | --- |
| `fvecs` | repeated `[u32 dimension][f32 coordinates]` |
| `bvecs` | repeated `[u32 dimension][u8 coordinates]` |
| `fbin` | `[u32 count][u32 dimension][packed f32 coordinates]` |
| `u8bin` | `[u32 count][u32 dimension][packed u8 coordinates]` |

Extensionless inputs retain the historical fvecs default. Ground truth is
ivecs. Byte coordinates are converted to f32 directly in the final allocation;
this is not a new quantization/scoring scheme. File lengths, dimensions,
per-record headers, and GT IDs are validated. File addressing uses 64-bit sizes;
node IDs remain u32, with u32::MAX reserved as a sentinel.

Use an existing **STAG v3** ParlayANN export for the exact base order and count.
The shared importer streams one node at a time into the final PhasedGraph slab,
validating counts, neighbor IDs, entry range, truncation, and trailing bytes.
The export degree must match `--graph-degree`. Its max-extra determines the
allocated slab stride. Entry selection retains the old sampled-medoid rule.

This does **not** make the ParlayANN builder external-memory. The older
`benchmark` multi-algorithm harness and its legacy partition importer still
have their original memory behavior; use the shared standalone CLI for this
large-dataset path. The in-process Rust builder can still allocate a separate
working dataset and partitions. Measure construction separately.

## SIFT1B workflow

Run from the repository root. Build on the target machine with its supported
instruction set; do not copy a native binary between incompatible CPUs.

```bash
cargo build --release -p benchmark --bin orion

# The preset points to data/sift1b/{bigann_base.bvecs,bigann_query.bvecs,idx_1000M.ivecs}.
# Override each path as needed. The PA export must already exist on a cache miss.
./target/release/orion sift1b --staged-file /data/sift1b/graph.staged \
  --cache-dir /data/orion-cache --memory-budget-gib 1800 --preflight

# Import and save only: no calibration, sidecar materialization, or query timing.
./target/release/orion sift1b --staged-file /data/sift1b/graph.staged \
  --cache-dir /data/orion-cache --memory-budget-gib 1800 --prepare-only

# Subsequent calls reuse the graph; the original staged export is not required.
./target/release/orion sift1b --cache-dir /data/orion-cache \
  --memory-budget-gib 1800 --k 10 --threads 32 --trials 3
./target/release/orion sift1b --cache-dir /data/orion-cache \
  --memory-budget-gib 1800 --k 100 --threads 32 --trials 3
```

The 1800 GiB budget is an example for a 2 TiB machine, not a validated peak
requirement. `--preflight` reads headers and resolves graph provenance without
loading vectors. Its lower bound includes the f32 base, graph slab, node reader
counters, and L2U8 admission buffer when selected. Other sidecars, construction,
query scratch, allocator overhead, and OS memory need additional headroom.
`--memory-budget-gib` rejects a known lower bound exceeding the budget; it is
not an RSS limiter or assurance that a run will fit.

For 1B x 128, R=64, extra=16 on a 64-bit host, these known resident components
are about 1,032 GB (decimal), including 128 GB of L2U8 admission data. Before
this change an additional 512 GB base copy was possible; PA import also held
full-file and per-node partition intermediates, which are now avoided.
Actual exported max-extra and degree can change this estimate.

For smaller prefixes, use `--max-points` together with a matching graph and
**ground truth computed for that prefix**. A dataset shortcut alone is not proof
of 1B coverage: check `num_points` in preflight and retain resolved run settings.

## Reliability and reporting

Graph and metadata writes use temporary files, flush/sync, and rename, with
metadata published after the graph. The shared CLI writes the provenance
manifest last and rejects incomplete/inconsistent cache sets. There is no
multi-file transaction or mid-import checkpoint: after a failed publication,
inspect the incomplete cache and use a fresh `--cache-dir` for a retry. Do not
run multiple writers against the same cache. Complete caches can be reused.

`benchmark/scripts/run_large_dataset.sh` wraps preflight, preparation and
k=10/100 sweeps, collecting per-stage `/usr/bin/time` logs, resolved settings,
source revision/dirty state and executable SHA-256 in a new run directory.
It expects a prebuilt executable and existing input/export files. On Linux,
`time -v` records maximum RSS; it is not a measurement of a separate ParlayANN
build. Preserve and report that builder's command, code version, elapsed time,
peak RSS, and disk use separately.

No 1B run has been performed locally. Tests cover format parity, sparse
billion-record header inspection, zero-copy ownership transfer, old/new graph
partition equivalence, cache reloads, invalid staged files, and large integer
capacity calculations; full-machine peak memory and throughput remain to be
measured on the target server.
