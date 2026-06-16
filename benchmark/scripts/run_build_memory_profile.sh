#!/usr/bin/env bash
# α-matched build + memory profile across the four reference datasets.
#
# Both engines run at the dataset's PA-aligned `scfg.alpha` so the only
# build-time delta isolates the PhasedGraph extras pass (and on the
# memory side, the candidate-set materialisation). Results land in
# `visualizations/{build,memory}_profile_<dataset>.json` and are
# rendered by `visualizations/plot_{build,memory}.py`.
#
# Plain indexed-array loop — macOS ships bash 3.2 which lacks
# `declare -A`, so the per-dataset paths are inlined per case below.

set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=./target/release/benchmark

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

for ds in sift glove25 glove100 gist; do
  paths_for "$ds"

  echo "═══ $ds : build-profile ═══"
  $BIN --base "$BASE" --query "$QRY" --groundtruth "$GT" \
       --algorithms build-profile
  echo "DONE: $ds build-profile"

  echo "═══ $ds : memory-profile ═══"
  $BIN --base "$BASE" --query "$QRY" --groundtruth "$GT" \
       --algorithms memory-profile
  echo "DONE: $ds memory-profile"
done

echo "all done"
