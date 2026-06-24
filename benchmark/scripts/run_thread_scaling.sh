#!/usr/bin/env bash
# Thread-scaling sweep across SIFT 1M and GIST 1M at PA-aligned build
# params (R / L_build / α resolved from `benchmark/configs/sweep.yaml`).
#
# Both engines build at the SAME orion params (since `run_thread_sweep`
# pulls from `scfg` directly), so the QPS-vs-thread curves isolate
# per-thread search efficiency — not graph-topology differences.
#
# Output: `visualizations/thread_sweep_{sift,gist}.json`, rendered by
# `visualizations/plot_threads.py` into `thread_scaling.png`.

set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=./target/release/benchmark

paths_for() {
  case "$1" in
    sift)
      BASE=data/sift/sift_base.fvecs
      QRY=data/sift/sift_query.fvecs
      GT=data/sift/sift_groundtruth.ivecs ;;
    gist)
      BASE=data/gist/gist_base.fvecs
      QRY=data/gist/gist_query.fvecs
      GT=data/gist/gist_groundtruth.ivecs ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

for ds in sift gist; do
  paths_for "$ds"
  echo "═══ $ds : thread-sweep ═══"
  $BIN --base "$BASE" --query "$QRY" --groundtruth "$GT" \
       --algorithms thread-sweep
  echo "DONE: $ds thread-sweep"
done

echo "all done"
