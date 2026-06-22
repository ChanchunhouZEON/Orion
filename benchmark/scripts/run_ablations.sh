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
# Both ablations share the cache-first staged-graph load mirroring
# the `staged_diskann.rs` bin's PA-mode naming — first run pays the
# import/build cost, subsequent runs load from
# `cache/staged_parlayann/<ds>_…_pct60.pgraph` in 0.1-2 s.
#
# Override the dataset list with `DATASETS="…"`; default covers the
# five panel datasets the README references.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

source "$SCRIPT_DIR/_env.sh"
setup_python_env || exit 1
BIN=./target/release/benchmark

DATASETS="${DATASETS:-sift gist glove25 glove100 msmarco_bert_1M}"
# Comma-separated `algorithms` to drive — defaults to both panels.
# Override with `ALGORITHMS=cascade-ablation` to skip the convergence
# panel (e.g. when only the cascade-tier numbers need refreshing).
ALGORITHMS="${ALGORITHMS:-ablation,cascade-ablation}"

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
    deep10m)
      BASE=data/deep10m/deep10m_base.fvecs
      QRY=data/deep10m/deep10m_query.fvecs
      GT=data/deep10m/deep10m_groundtruth.ivecs ;;
    msmarco_bert_1M)
      BASE=data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs
      QRY=data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs
      GT=data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs ;;
    wiki_ada_1M)
      BASE=data/wiki_ada_1M/wiki_ada_1M_base.fvecs
      QRY=data/wiki_ada_1M/wiki_ada_1M_query.fvecs
      GT=data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

echo "── building benchmark binary ──"
cargo build --release --bin benchmark 2>&1 | tail -2

# Run each algorithm × each dataset. Iterating algorithm-outermost
# lets the staged graph + sidecars stay warm in page-cache across
# the two ablations on the same dataset (only the staged graph is
# shared though — DiskANN baseline still rebuilds per ablation run).
IFS=',' read -ra ALG_LIST <<< "$ALGORITHMS"
for alg in "${ALG_LIST[@]}"; do
  for ds in $DATASETS; do
    paths_for "$ds"
    echo "── $alg: $ds ──"
    $BIN --base "$BASE" --query "$QRY" --groundtruth "$GT" \
      --algorithms "$alg"
  done
done

echo "── rendering panels ──"
for alg in "${ALG_LIST[@]}"; do
  case "$alg" in
    ablation)
      for ds in $DATASETS; do
        $PY visualizations/plot_ablation.py --dataset "$ds"
      done
      ;;
    cascade-ablation)
      for ds in $DATASETS; do
        $PY visualizations/plot_cascade_ablation.py --dataset "$ds"
      done
      ;;
    *) echo "[warn] no renderer wired for algorithm '$alg' — skipping plot" >&2 ;;
  esac
done

echo "all done"
