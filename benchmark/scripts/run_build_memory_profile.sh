#!/usr/bin/env bash
# α-matched build + memory profile across the four reference datasets.
#
# Both engines run at the dataset's PA-aligned `scfg.alpha` so the only
# build-time delta isolates the PhasedGraph extras pass (and on the
# memory side, the candidate-set materialisation). Results land in
# `visualizations/{build,memory}_profile_<dataset>.json` and are
# rendered by `visualizations/plot_{build,memory}.py`.
#
# Dataset paths are resolved by the shared Rust configuration loader.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BIN="${BIN:-$(binary_path benchmark)}"


[[ -x "$BIN" ]] || build_binary benchmark

for ds in sift glove25 glove100 gist; do
  paths_for "$ds"

  echo "═══ $ds : build-profile ═══"
  "$BIN" --base "$BASE" --query "$QRY" --groundtruth "$GT" \
       --algorithms build-profile
  echo "DONE: $ds build-profile"

  echo "═══ $ds : memory-profile ═══"
  "$BIN" --base "$BASE" --query "$QRY" --groundtruth "$GT" \
       --algorithms memory-profile
  echo "DONE: $ds memory-profile"
done

echo "all done"
