#!/usr/bin/env bash
# ADSampling 4-way comparison driver.
#
# Builds the `benchmark` binary, runs `--algorithms ads-comparison` on
# GIST 1M (the headline ADS-sensitive workload), then re-renders
# `visualizations/ads_comparison.png` via `plot_ads.py`. All four
# variants reuse the cascade cache (cache/diskann/*_ads.bin,
# cache/orion/*_ads.bin) — first invocation pays the full build
# cost (~6 min); subsequent runs of the same config hit the cache.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup_python_env || exit 1
BIN="${BIN:-$(binary_path benchmark)}"

DATASET="${DATASET:-gist}"
case "$DATASET" in
  sift|glove25|glove100|gist) ;;
  *) echo "ADS panel supports sift, glove25, glove100 and gist" >&2; exit 2 ;;
esac
paths_for "$DATASET"

echo "── building benchmark binary ──"
build_binary benchmark

echo "── ads-comparison: $DATASET ──"
"$BIN" --base "$BASE" --query "$QRY" --groundtruth "$GT" --algorithms ads-comparison

echo "── rendering ads_comparison.png ──"
"$PY" visualizations/plot_ads.py

echo "all done"
