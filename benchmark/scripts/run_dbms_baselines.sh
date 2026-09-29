#!/usr/bin/env bash
# In-process Milvus Lite + USearch panel across the headline datasets.
#
# Phase 1: For every dataset, run dbms_baselines.py which adds two
#          series to baseline_<ds>.json — `milvus_auto` (Milvus Lite
#          AUTOINDEX = IVF-class index family), `usearch_hnsw`
#          (Unum's in-process SIMD HNSW, M=16/efC=200) and `faiss hnsw + sq`.
#
# Phase 2: For datasets that don't have the older HNSW / FAISS / Annoy
#          rows in their baseline JSON yet, run baseline_comparison.py
#          --no-rust (adds hnswlib + reuses the Rust series from
#          sweep_orion_vs_parlayann_<ds>.json) + additional_baselines
#          (adds FAISS IVF-Flat / IVF-PQ + Annoy) so the panel is
#          complete.
#
# Phase 3: Re-render visualizations/baseline_panel_<ds>.png for each
#          dataset via plot_baseline_comparison.py --all.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup_python_env || exit 1

# All 7 headline datasets (matches what plot_dataset_all.py considers
# the public set; fashion-mnist is excluded from the panel by design).
DATASETS="${DATASETS:-sift glove25 glove100 gist deep10m msmarco_bert_1M wiki_ada_1M}"

# Datasets that need phase 2 to fill hnswlib + FAISS + Annoy rows into
# baseline_<ds>.json (phase 1 only writes usearch_hnsw + lancedb_hnsw).
# Default to every dataset so a fresh-from-scratch JSON gets the full
# series set; pre-existing JSONs are preserved by the merge logic in
# baseline_comparison.py.
NEED_BASELINE_FILL="${NEED_BASELINE_FILL:-$DATASETS}"

run_phase1() {
  local ds=$1
  echo "── phase 1 (dbms_baselines): $ds ──"
  "$PY" benchmark/scripts/dbms_baselines.py \
       --dataset "$ds" --max-points 0 --threads 8 --trials 3
}

run_phase2_baseline() {
  local ds=$1
  echo "── phase 2 (baseline_comparison + additional_baselines): $ds ──"
  "$PY" benchmark/scripts/baseline_comparison.py \
       --dataset "$ds" --max-points 0 --threads 8 --trials 3 --no-rust
  "$PY" benchmark/scripts/additional_baselines.py \
       --dataset "$ds" --max-points 0 --threads 8 --trials 3
}

# ── Phase 1: in-process Milvus + USearch + FAISS HNSW + SQ on every dataset ────────────────
for ds in $DATASETS; do
  run_phase1 "$ds"
done

# ── Phase 2: fill the Python-baseline panel where missing ─────────────────
for ds in $NEED_BASELINE_FILL; do
  if [[ " $DATASETS " == *" $ds "* ]]; then
    run_phase2_baseline "$ds"
  fi
done

# ── Phase 3: render every panel + a combined grid ────────────────────────
echo "── phase 3: render panels ──"
"$PY" visualizations/plot_baseline_comparison.py --all

echo "all done"
