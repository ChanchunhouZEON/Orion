# Large datasets on the in-memory Orion path

The standalone `orion` executable supports a native-u8 search path for SIFT.
The `sift10m`, `sift100m`, and `sift1b` presets store original byte coordinates
in one 64-byte-aligned allocation, transferred to the index without a copy.
Admission computes exact squared L2 directly from this buffer. No quantized
admission sidecar or final precision reranking is needed. Convergence and
local/extra candidate expansion remain active.

Other presets retain f32 storage. The shared f32 CLI path used by `orion`,
`adaptive`, `adaptive_ee_ablation`, and `matched_state` also transfers its base
allocation without copying. The latter three diagnostics currently require f32
storage and reject a native-u8 configuration before loading the dataset.

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
ivecs. With `--vector-storage f32`, byte coordinates are converted directly into
the final f32 allocation. With `--vector-storage u8`, both base and queries must
use bvecs or u8bin; the base stays in original byte coordinates. This path
currently requires dimension 128, L2, a ParlayANN graph, `--prefilter none`,
`--admission l2-u8`, and `--rerank none`. Queries use a small f32 interface buffer
and are checked before conversion to bytes; no full f32 base is materialized. File lengths, dimensions,
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

## SIFT presets and workflow

`benchmark/configs/sweep.yaml` defines these native-u8 defaults:

| Preset | Base prefix | Ground truth under `data/sift1b/` |
| --- | ---: | --- |
| `sift10m` | 10,000,000 | `idx_10M.ivecs` |
| `sift100m` | 100,000,000 | `idx_100M.ivecs` |
| `sift1b` | 1,000,000,000 | `idx_1000M.ivecs` |

All three read `bigann_base.bvecs` and `bigann_query.bvecs` from that directory.
Each requires its own graph matching the loaded prefix. Defaults are R=64,
extra=16, original-u8 admission, no prefilter, and no final rerank. CLI options
override preset values. To compare the previous f32 storage and reranking path,
pass both `--vector-storage f32 --rerank f32`; its memory requirements are higher.
Native-u8 and f32 use separate cache namespaces.

Run from the repository root. Build on the target machine with its supported
instruction set; do not copy a native binary between incompatible CPUs.

```bash
cargo build --release -p benchmark --bin orion

# The preset points to data/sift1b/{bigann_base.bvecs,bigann_query.bvecs,idx_1000M.ivecs}.
# Override each path as needed. The PA export must already exist on a cache miss.
./target/release/orion sift1b --staged-file /data/sift1b/graph.staged \
  --cache-dir /path/to/orion/cache/graphs/dir --memory-budget-gib 650 --preflight

# Import and save only: no calibration, sidecar materialization, or query timing.
./target/release/orion sift1b --staged-file /data/sift1b/graph.staged \
  --cache-dir /path/to/orion/cache/graphs/dir --memory-budget-gib 650 --prepare-only

# Subsequent calls reuse the graph; the original staged export is not required.
./target/release/orion sift1b --cache-dir /path/to/orion/cache/graphs/dir \
  --memory-budget-gib 650 --k 10 --threads 32 --trials 3
./target/release/orion sift1b --cache-dir /path/to/orion/cache/graphs/dir \
  --memory-budget-gib 650 --k 100 --threads 32 --trials 3
```

The 650 GiB budget is an example for a 768 GiB machine, not a validated peak
requirement. `--preflight` reads headers and resolves graph provenance without
loading vectors. Its lower bound includes the resident base, graph slab, node
reader counters, and any separate L2U8 admission buffer. Native-u8 reports
`base_f32_bytes=0` and `l2_u8_admission_bytes=0`; its byte base is counted once.
Construction, query scratch, allocator overhead, and OS memory need additional
headroom. `--memory-budget-gib` rejects a known lower bound exceeding the budget;
it is not an RSS limiter or assurance that a run will fit.

For dimension 128, R=64, extra=16 on a 64-bit host, known native-u8 resident
components are approximately:

| Points | Byte base (decimal GB) | Total lower bound (decimal GB) | Total lower bound (GiB) |
| ---: | ---: | ---: | ---: |
| 10M | 1.28 | 5.2 | 4.84 |
| 100M | 12.8 | 52 | 48.43 |
| 1B | 128 | 520 | 484.29 |

Totals include a 384-byte graph slot and an 8-byte reader counter per node, plus
64 bytes of base tail padding. Actual exported max-extra and degree can change
this estimate. Only the base payload falls to one quarter of f32 size; the graph
is unchanged. The former f32 + L2U8-sidecar path needs about 1,032 GB at 1B.

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

## ParlayANN preparation and tests

`benchmark/scripts/parlayann_convert.py` now converts bvecs/fvecs and binary
inputs in bounded chunks, retains byte coordinates and validates GT against the
loaded prefix. Existing `--base-fvecs` / `--query-fvecs` calls remain valid.
Use `benchmark/scripts/prepare_parlayann_large.py --help` for the build-only
pipeline with graph + extras checkpoint, STAG, complete logs and resource reports.
The ParlayANN sibling repository documents its CTest suite in `tests/README.md`
and its construction/re-export workflow in `docs/large-datasets.md`.

The byte `FullPrecisionDistance` implementation computes original-coordinate
squared L2. Tests cover byte storage, scalar/SIMD distance parity, native admission
cutoffs, calibration and search parity with exact f32 on the same graph, prefix
configuration, preflight accounting, and cache ownership transfer. The optional
executable smoke tests verify preparation, cache reload without the STAG source,
search recall, zero f32 search comparisons, and absence of admission sidecars:

```bash
cargo test -p vector --offline
cargo test -p orion --offline
cargo test -p benchmark --bin orion --offline
cargo build -p benchmark --bin orion --offline
ORION_BINARY="$PWD/target/debug/orion" python3 -m unittest discover \
  -s benchmark/scripts/tests -v
```

These are small-fixture correctness tests; no SIFT10M, SIFT100M, or SIFT1B search
has been run as part of this integration.
