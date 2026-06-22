#!/usr/bin/env bash
# Ablation study driver.
#
# Builds the `benchmark` binary, runs `--algorithms ablation` on every
# dataset the ablation figure displays (currently SIFT + GloVe-100),
# then re-renders `visualizations/ablation_study.png` via
# `plot_ablation.py`. Each per-dataset run emits
# `visualizations/ablation_<ds>.json` consumed by the plot script.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

source "$SCRIPT_DIR/_env.sh"
setup_python_env || exit 1
BIN=./target/release/benchmark

DATASETS="${DATASETS:-sift glove100}"

paths_for() {
  case "$1" in
    sift)
      BASE=data/sift/sift_base.fvecs
      QRY=data/sift/sift_query.fvecs
      GT=data/sift/sift_groundtruth.ivecs ;;
    glove25)
      BASE=data/glove25_norm/glove-25-angular_base.fvecs
      QRY=data/glove25_norm/glove-25-angular_query.fvecs
      GT=data/glove25_norm/glove-25-angular_groundtruth.ivecs ;;
    glove100)
      BASE=data/glove100_norm/glove-100-angular_base.fvecs
      QRY=data/glove100_norm/glove-100-angular_query.fvecs
      GT=data/glove100_norm/glove-100-angular_groundtruth.ivecs ;;
    gist)
      BASE=data/gist/gist_base.fvecs
      QRY=data/gist/gist_query.fvecs
      GT=data/gist/gist_groundtruth.ivecs ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

echo "── building benchmark binary ──"
cargo build --release --bin benchmark 2>&1 | tail -2

for ds in $DATASETS; do
  paths_for "$ds"
  echo "── ablation: $ds ──"
  $BIN --base "$BASE" --query "$QRY" --groundtruth "$GT" --algorithms ablation
done

echo "── rendering ablation_study.png ──"
$PY visualizations/plot_ablation.py

echo "all done"
