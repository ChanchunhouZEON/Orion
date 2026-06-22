#!/usr/bin/env bash
# Run the full multi-algorithm SIFT 1M baseline panel at 8 threads.
#
# Order:
#   1. baseline_comparison.py --no-rust  →  HNSW (hnswlib) + reuse
#      DiskANN / Staged / ParlayANN series from the head-to-head sweep
#      JSON produced by `sweep_staged_vs_diskann_vs_parlayann.sh`.
#   2. additional_baselines.py            →  FAISS IVF-Flat, FAISS IVF-PQ,
#      Annoy. Reads + appends to `visualizations/baseline_sift.json`.
#
# Output: `visualizations/baseline_sift.json` with seven series.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

source "$SCRIPT_DIR/_env.sh"
setup_python_env || exit 1

# --max-points 0 — full SIFT 1M (no subset, no GT recomputation).
# --threads 8   — matches `sweep.yaml` defaults.sweep.threads, so the
#                 Rust series (Staged / DiskANN / PA) are head-to-head.
$PY benchmark/scripts/baseline_comparison.py \
    --dataset sift --max-points 0 --threads 8 --trials 5 --no-rust

$PY benchmark/scripts/additional_baselines.py \
    --dataset sift --max-points 0 --threads 8 --trials 5

echo "all done"
