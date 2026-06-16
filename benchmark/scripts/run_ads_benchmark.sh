#!/usr/bin/env bash
# ADSampling 4-way comparison driver.
#
# Builds the `benchmark` binary, runs `--algorithms ads-comparison` on
# GIST 1M (the headline ADS-sensitive workload), then re-renders
# `visualizations/ads_comparison.png` via `plot_ads.py`. All four
# variants reuse the cascade cache (cache/diskann/*_ads.bin,
# cache/staged/*_ads.bin) — first invocation pays the full build
# cost (~6 min); subsequent runs of the same config hit the cache.

set -euo pipefail
cd "$(dirname "$0")/../.."

PY=/opt/anaconda3/envs/ray/bin/python
BIN=./target/release/benchmark

DATASET="${DATASET:-gist}"
case "$DATASET" in
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
  *) echo "unknown DATASET=$DATASET (use sift|glove25|glove100|gist)" >&2; exit 1 ;;
esac

echo "── building benchmark binary ──"
cargo build --release --bin benchmark 2>&1 | tail -2

echo "── ads-comparison: $DATASET ──"
$BIN --base "$BASE" --query "$QRY" --groundtruth "$GT" --algorithms ads-comparison

echo "── rendering ads_comparison.png ──"
$PY visualizations/plot_ads.py

echo "all done"
