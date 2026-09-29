# Benchmark scripts

Run wrappers from any working directory. Relative data, configuration and output
paths are resolved against the Orion repository root. `BUILD_DIR` (or
`CARGO_TARGET_DIR`) selects the Cargo target directory; `PY` selects Python.

Shared implementation:

- `_common.sh`: repository anchoring, Cargo binary paths/builds, and preset paths
  obtained from `orion <dataset> --print-config`.
- `_parlay.sh`: published ParlayANN recipes and checkout validation. Raw GloVe
  paths here intentionally differ from Orion's normalized preset paths.
- `benchmark_support.py`: resolved preset paths/metrics, bounded prefix reads,
  recall validation, and chunked exact L2/IP/cosine ground truth.
- `../../data/vector_io.py`: chunked little-endian fvecs/ivecs writers.

`ORION_CONFIG_BIN` can select an existing `orion` executable (including a debug
build). Shell wrappers build the default executable when needed. Standalone
Python baseline scripts expect it to be built first:

```bash
cargo build --release --bin orion
python3 benchmark/scripts/dbms_baselines.py --help
```

`run_large_dataset.sh` uses `orion-sweep`, with preflight using the largest run's
k and the actual thread count. `run_search_profile.sh` also uses `orion-sweep`
and waits for its readiness marker with a timeout; it requires macOS xctrace.
It writes a new output directory and retains logs when preparation fails.

The three-engine comparison supports full presets at k=10 and eight threads.
`MAX_POINTS` is rejected there because the DiskANN executable cannot accept
subset-specific ground truth. `prepare_parlayann_data.sh` still supports subsets:
it converts vectors without the full-set GT, then recomputes GT with the recipe's
metric. MS-MARCO participates in all three engines; this workflow assumes its
vectors have already been normalized.

`baseline_comparison.py` refreshes the maintained three-engine comparison, or
reads its JSON with `--no-rust`. It no longer calls the removed
`qps-recall-sweep` algorithm. `orion_py_bench.py` requires `ORION_CACHE_PATH`
instead of guessing a cache name from an obsolete naming scheme.

Validation (no full datasets required):

```bash
python3 -m unittest discover -s benchmark/scripts/tests -v
```

Set `ORION_BINARY` to `orion-sweep` and `PARLAY_NEIGHBORS` to an existing neighbors
executable to enable native-byte/ParlayANN integration fixtures. Wrapper tests
use temporary fake executables; they cover command arguments, paths containing
spaces, build-directory overrides, process failures and log parsing. On macOS,
resource-timing tests require permission for `/usr/bin/time -l` to read sysctl.
