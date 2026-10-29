# Large datasets on the in-memory Orion path

The standalone `orion` executable (from the `orion-cli` crate) supports a native-u8 search path for SIFT.
The `sift10m`, `sift100m`, and `sift1b` presets store original byte coordinates
in one 64-byte-aligned allocation, transferred to the index without a copy.
Admission computes exact squared L2 directly from this buffer. No quantized
admission sidecar or final precision reranking is needed. Convergence and
local/extra candidate expansion remain active.

Other presets retain f32 storage. The shared f32 CLI path used by `orion`,
`adaptive`, `adaptive_ee_ablation`, and `matched_state` also transfers its base
allocation without copying. The latter three diagnostics currently require f32
storage and reject a native-u8 configuration before loading the dataset.

See [CLI and Query Inputs](cli.md) for anonymous datasets, optional GT, query streams,
and detailed memory reports. Repeated experiments use `orion-sweep`; ordinary
`orion` searches each query once.

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
`--admission native-l2-u8`, and `--rerank none`. Queries use a small f32 interface buffer
and are checked before conversion to bytes; no full f32 base is materialized. File lengths, dimensions,
per-record headers, and GT IDs are validated. File addressing uses 64-bit sizes;
node IDs remain u32, with u32::MAX reserved as a sentinel.

Native-u8 `.u8bin` base payloads load directly into the final aligned allocation
in requests of up to 8 MiB, without a per-vector staging buffer. Prefix limits
and the zeroed SIMD tail are preserved. Other formats retain their decoding and
per-record validation paths; this changes neither file formats nor threading.

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

| Preset | Loaded points | Default base | Ground truth |
| --- | ---: | --- | --- |
| `sift10m` | 10,000,000 | `data/sift10m/base.10M.bvecs` | `data/sift10m/groundtruth.10M.ivecs` |
| `sift100m` | 100,000,000 | `data/sift1b/bigann_base.bvecs` | `data/sift1b/idx_100M.ivecs` |
| `sift1b` | 1,000,000,000 | `data/sift1b/bigann_base.bvecs` | `data/sift1b/idx_1000M.ivecs` |

SIFT10M uses `data/sift10m/query.public.10K.bvecs`; larger presets use
`data/sift1b/bigann_query.bvecs`. Override these paths for remote deployments.
Each requires its own graph matching the loaded prefix. Defaults are R=64,
extra=16, original-u8 admission, no prefilter, and no final rerank. CLI options
override preset values. To compare the previous f32 storage and reranking path,
pass both `--vector-storage f32 --admission l2-u8 --rerank f32`; its memory requirements are higher.
Native-u8 and f32 use separate cache namespaces.

The admission names are explicit: `native-l2-u8` scores original bytes exactly,
whereas `l2-u8` scores an affine-quantized sidecar of an f32 base. Existing custom
native-u8 configurations must change their admission to `native-l2-u8`.
Configuration resolution produces a validated `SearchPlan` before loading:
`orion-cli/src/cli/plan.rs` owns compatibility and storage accounting, and `orion-cli/src/cli/execution.rs`
owns the standalone sweep's typed index/admission adapters. Both representations
share calibration, the search loop, timing, and reporting.


Run from the repository root. Build on the target machine with its supported
instruction set; do not copy a native binary between incompatible CPUs.

```bash
cargo build --release -p orion-cli --bin orion
cargo build --release -p benchmark --bin orion-sweep

# The preset points to data/sift1b/{bigann_base.bvecs,bigann_query.bvecs,idx_1000M.ivecs}.
# Override each path as needed. The PA export must already exist on a cache miss.
./target/release/orion sift1b --staged-file /data/sift1b/graph.staged \
  --cache-dir /path/to/orion/cache/graphs/dir --memory-budget-gib 650 --preflight

# Import and save only: no calibration, sidecar materialization, or query timing.
./target/release/orion sift1b --staged-file /data/sift1b/graph.staged \
  --cache-dir /path/to/orion/cache/graphs/dir --memory-budget-gib 650 --prepare-only

# Subsequent calls reuse the graph; the original staged export is not required.
./target/release/orion-sweep sift1b --cache-dir /path/to/orion/cache/graphs/dir \
  --memory-budget-gib 650 --k 10 --threads 32 --trials 3
./target/release/orion-sweep sift1b --cache-dir /path/to/orion/cache/graphs/dir \
  --memory-budget-gib 650 --k 100 --threads 32 --trials 3
```

The 650 GiB budget is an example for a 768 GiB machine, not a validated peak
requirement. `orion sift1b --preflight` can estimate from YAML alone, before
downloading data or exporting a graph. Available headers validate the declared
counts and supply the actual graph layout; missing files use configuration
assumptions labeled in the report. Its lower bound includes the resident base, graph slab, node
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

`benchmark/scripts/run_large_dataset.sh` uses `orion-sweep` and wraps preflight, preparation and
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
cargo test -p orion-cli --offline
cargo build -p benchmark --bin orion-sweep --offline
ORION_BINARY="$PWD/target/debug/orion-sweep" python3 -m unittest discover \
  -s benchmark/scripts/tests -v
```

These are small-fixture correctness tests; no SIFT10M, SIFT100M, or SIFT1B search
has been run as part of this integration.

## DiskANN f32 baseline: bounded-memory input

`diskann_sweep` accepts `sift10m`, `sift100m`, and `sift1b`. These remain
**f32 DiskANN** experiments: bvecs/u8bin coordinates are widened during a
streaming conversion, preserving their exact values. The driver never holds
an entire base allocation. Conversion uses batches of 8,192 rows; queries
remain resident for repeated sweeps. Existing normalized dataset recipes are
unchanged.

```bash
./diskann_sweep sift1b \
  --base /data/bigann_base.bvecs \
  --query /data/bigann_query.bvecs \
  --ground-truth /data/idx_1B.ivecs \
  --cache-dir /data/cache/diskann-sift1b \
  --threads 32 --k 100 --trials 3 --search-list-sizes 100,128,256,512,1024
```

Use `--prepare-only` to build/cache without queries or GT. For SIFT100M,
select `sift100m` and supply `idx_100M.ivecs`; its default prefix limit is
100 million rows. Smaller `--max-points` runs need their own matching GT.
Input paths infer formats from `.fvecs`, `.bvecs`, `.fbin`, or `.u8bin`.

A complete fbin requiring no normalization is passed directly to the index.
Other inputs are converted before index allocation, into `--cache-dir` rather
than `/tmp` (which may be RAM-backed). After a successful build, the temporary
conversion is removed before saving the index; unwinding also attempts cleanup.
An existing source fbin is never deleted. Abrupt termination can leave a
`diskann-build-<pid>.fbin` file. Do not share a cache directory between
concurrent builds.

DiskANN reads packed binary coordinates directly into its dataset in requests
of up to 8 MiB when file dimension equals resident stride (including SIFT128).
Padded dimensions reuse one block buffer, copy coordinates into aligned rows,
and zero each row's padding. Both construction input and saved `.data` reloads
use this path; malformed lengths and destination bounds are checked before read.

For 1B × 128, removing the duplicate f32 base saves 476.84 GiB. The remaining
index still allocates vectors, graph adjacency, and (in this build) neighbor
distances. Graph growth, allocator overhead and construction scratch are not
bounded by this driver; 2 TB capacity is not a measured peak guarantee. Run
one engine/task at a time and record `/usr/bin/time -v` maximum RSS. No full
1B construction has been validated locally.

For a fresh SIFT1B build, conversion needs approximately 512 GB of temporary
fbin storage. Successful cleanup releases it before save writes approximately
512 GB of index vectors and up to 260 GB of degree-64 graph data. These phases
therefore need approximately max(512, 512 + 260) = 772 GB of additional disk
space, plus filesystem headroom, rather than their combined 1.3 TB. This excludes
source data, existing indices, other engines' outputs, and abandoned temporary
files. Cleanup failures are reported and can leave the conversion occupying
space. A direct fbin input avoids the temporary conversion altogether.

Cache hits inspect base shape and metadata but do not decode its payload.
A manifest records canonical source path, size, mtime, prefix count,
normalization and build recipe. Legacy caches without a manifest are rebuilt.
This is not a content checksum: in-place data replacement preserving metadata
requires cache removal. Keep immutable inputs and separate experiment caches.

Validation:

```bash
cargo test -p benchmark --bin diskann_sweep
DISKANN_BINARY=./target/debug/diskann_sweep \
  python3 benchmark/scripts/tests/check_diskann_streaming.py
```

### Linux x86 verification and SIMD alignment

The f32/f16 AVX2 distance API accepts ordinary array references, which are not
necessarily SIMD-aligned. Its loads must therefore be unaligned, and dimensions
not divisible by eight require a scalar tail. The fixed implementation includes
an explicitly misaligned regression fixture. This addresses a concrete source
of SIGSEGV in the f32 distance used by ADSampling tests; confirm the rebuilt
`vector-tests` on the allocated compute node before a large experiment.

The repository now pins Rust 1.92. The earlier Rust 1.86 limitation no longer
describes the current build: Rust 1.92 supports the AVX-512 intrinsics and
`target_feature` configuration used here. Selecting the toolchain alone does
not enable AVX-512; the x86 build must also enable the kernel's required target
features (including AVX512F/BW/DQ/VL and VPOPCNTDQ for the deployed bundle).
These compile-time-selected paths require compatible compute nodes. Check
generated assembly and run the vector regression tests on the target machine;
do not infer the active kernel solely from the toolchain version.

## Storage and large pages

The shared [platform block writer](../platform/src/buffered_io.rs) combines small
headers and payloads into 8 MiB writes. Orion graph caches and DiskANN save paths
use this utility without making Orion's writer depend on DiskANN's writer.
File formats remain unchanged. Large reads/writes describe request size, not
parallel I/O; do not infer additional reader/writer threads from block size.

For the standard SIFT1B graph, the `.pgraph` logical size is 384,000,000,016 bytes.
Reserve roughly 400 GB for this graph cache alone, in addition to base, STAG,
other engines' caches and temporary files. Actual filesystem allocation and user
quota can differ from file length and filesystem-wide free space. Block writes
fixed observed small-write allocation amplification on the tested Lustre system;
this is not a universal filesystem compression guarantee.

Large Linux allocations use best-effort THP advice through the shared allocator.
Check actual page coverage, not just whether the advice call succeeded. See
[Platforms](platforms.md) for details and [Benchmarks](benchmarks.md#sift1b-on-x86)
for the measured SIFT1B run. Native-u8 automatic graph selection still requires
PA; it never widens the base and falls back to Rust.
