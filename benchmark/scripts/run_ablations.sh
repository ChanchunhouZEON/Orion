#!/usr/bin/env bash
# Unified ablation driver — runs both ablation panels per dataset and
# renders the matching figures.
#
# Two panels per dataset, both QPS-vs-Recall@10 curves:
#
#   * `ablation`           — convergence-side ablation. Four series:
#       origin / no-extra / no-early-stop / full.
#       Writes `visualizations/ablation_<ds>.{json,png,pdf}`.
#
#   * `cascade-ablation`   — cascade-tier ablation. Four series:
#       admission-only / no-rerank / no-prefilter / full.
#       Writes `visualizations/cascade_ablation_<ds>.{json,png,pdf}`.
#
# Both ablations share the cache-first orion-graph load mirroring
# the `orion.rs` bin's PA-mode naming — first run pays the
# import/build cost, subsequent runs load from
# `cache/orion_parlayann/<ds>_…_pct60.pgraph` in 0.1-2 s.
#
# Override the dataset list with `DATASETS="…"`; default covers the
# five panel datasets the README references.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup_python_env || exit 1
BIN="${BIN:-$(binary_path benchmark)}"

DATASETS="${DATASETS:-sift gist glove25 glove100 msmarco_bert_1M}"
# Comma-separated `algorithms` to drive — defaults to both panels.
# Override with `ALGORITHMS=cascade-ablation` to skip the convergence
# panel (e.g. when only the cascade-tier numbers need refreshing).
ALGORITHMS="${ALGORITHMS:-ablation,cascade-ablation}"


echo "── building benchmark binary ──"
build_binary benchmark

# Run each algorithm × each dataset. Iterating algorithm-outermost
# lets the orion graph + sidecars stay warm in page-cache across
# the two ablations on the same dataset (only the orion graph is
# shared though — DiskANN baseline still rebuilds per ablation run).
IFS=',' read -ra ALG_LIST <<< "$ALGORITHMS"
for alg in "${ALG_LIST[@]}"; do
  for ds in $DATASETS; do
    paths_for "$ds"
    echo "── $alg: $ds ──"
    "$BIN" --base "$BASE" --query "$QRY" --groundtruth "$GT" \
      --algorithms "$alg"
  done
done

echo "── rendering panels ──"
for alg in "${ALG_LIST[@]}"; do
  case "$alg" in
    ablation)
      for ds in $DATASETS; do
        "$PY" visualizations/plot_ablation.py --dataset "$ds"
      done
      ;;
    cascade-ablation)
      for ds in $DATASETS; do
        "$PY" visualizations/plot_cascade_ablation.py --dataset "$ds"
      done
      ;;
    *) echo "[warn] no renderer wired for algorithm '$alg' — skipping plot" >&2 ;;
  esac
done

echo "all done"
