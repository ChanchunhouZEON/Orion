#!/usr/bin/env bash
# Cascade-stage ablation driver.
#
# Runs `benchmark --algorithms cascade-ablation` on the chosen datasets
# (default: SIFT only — the per-stage QPS contribution analysis was
# designed first as a SIFT-scope sanity check before extending across
# datasets). Each per-dataset run emits
# `visualizations/cascade_ablation_<ds>.json` with four series:
#   - full           (default per-dataset cascade)
#   - no_prefilter   (force PrefilterChoice::None)
#   - no_rerank      (force RerankChoice::None)
#   - admission_only (force both)
#
# Then renders `visualizations/cascade_ablation_<ds>.png/.pdf` via
# `plot_cascade_ablation.py`.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Resolve $PY via the shared env helper (probes user-set $PY first,
# then falls back to PATH `python3` / common conda envs). Validates
# that matplotlib + numpy are importable before continuing.
source "$SCRIPT_DIR/_env.sh"
setup_python_env || exit 1
BIN=./target/release/benchmark

DATASETS="${DATASETS:-sift}"

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

for ds in $DATASETS; do
  paths_for "$ds"
  echo "── cascade-ablation: $ds ──"
  $BIN --base "$BASE" --query "$QRY" --groundtruth "$GT" \
    --algorithms cascade-ablation
done

echo "── rendering cascade_ablation_<ds>.png ──"
for ds in $DATASETS; do
  $PY visualizations/plot_cascade_ablation.py --dataset "$ds"
done

echo "all done"
