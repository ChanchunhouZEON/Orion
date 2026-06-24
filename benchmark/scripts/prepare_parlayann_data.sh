#!/usr/bin/env bash
# One-time ParlayANN data prep for a dataset: convert rust-side fvecs
# to ParlayANN's fbin + gt format, then build the `.staged` export via
# ParlayANN's `./neighbors -staged_outfile`.
#
# For truncated datasets (e.g. GIST-100k sampled from GIST-1M), the
# original `.ivecs` ground truth is invalid against the truncated base,
# so we brute-force recompute gt via `compute_gt_brute.py`. Otherwise
# we pass the `.ivecs` straight through `parlayann_convert.py`.
#
# Usage:
#   DATASET=gist     bash benchmark/scripts/prepare_parlayann_data.sh
#   DATASET=glove25  bash benchmark/scripts/prepare_parlayann_data.sh
#   DATASET=glove100 bash benchmark/scripts/prepare_parlayann_data.sh
#   DATASET=sift     bash benchmark/scripts/prepare_parlayann_data.sh  # re-prep

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASET="${DATASET:-sift}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

source "$SCRIPT_DIR/_env.sh"
setup_python_env numpy || exit 1

LOCAL_PCT="${LOCAL_PCT:-60}"

case "$DATASET" in
    sift)
        BASE_FVECS="data/sift/sift_base.fvecs"
        QUERY_FVECS="data/sift/sift_query.fvecs"
        GT_IVECS="data/sift/sift_groundtruth.ivecs"
        MAX_POINTS=""           # full 1M
        PA_DIR_NAME="sift1m"
        # PA's published SIFT recipe (`-R 64 -L 128 -alpha 1.15
        # -num_passes 2 -quantize_bits 8 -verbose`). L=128 (was 100) +
        # α=1.15 (was 1.2) + the second build pass tighten the graph
        # vs our prior `-L 100 -α 1.2 -num_passes 1` config — visibly
        # higher recall per visited node at the same search L.
        PA_R=64; PA_L=128; PA_ALPHA=1.15; MAX_EXTRA="${MAX_EXTRA:-16}"
        PA_NUM_PASSES="${PA_NUM_PASSES:-2}"
        PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS:-8}"
        PA_VERBOSE="${PA_VERBOSE:-1}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-Euclidian}"
        ;;
    glove25)
        BASE_FVECS="data/glove25/glove-25-angular_base.fvecs"
        QUERY_FVECS="data/glove25/glove-25-angular_query.fvecs"
        GT_IVECS="data/glove25/glove-25-angular_groundtruth.ivecs"
        MAX_POINTS=""
        PA_DIR_NAME="glove25"
        PA_R=100; PA_L=200; PA_ALPHA=1; MAX_EXTRA="${MAX_EXTRA:-16}"
        PA_NUM_PASSES="${PA_NUM_PASSES:-2}"
        PA_NORMALIZE="${PA_NORMALIZE:-1}"
        PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS:-8}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-mips}"
        ;;
    glove100)
        BASE_FVECS="data/glove100/glove-100-angular_base.fvecs"
        QUERY_FVECS="data/glove100/glove-100-angular_query.fvecs"
        GT_IVECS="data/glove100/glove-100-angular_groundtruth.ivecs"
        MAX_POINTS=""
        PA_DIR_NAME="glove100"
        # Match our `glove100_aligned` build recipe (R=100 L=200 α=1.0 on
        # normalized vectors) *plus* PA's `-num_passes 2` — the second
        # pass re-inserts each vertex against the pass-1 graph so
        # neighbour selection is seeded with real connectivity instead
        # of the near-empty state pass 1 starts from. This is the single
        # biggest graph-quality lever PA uses on GloVe-100 that we were
        # missing.
        PA_R=100; PA_L=200; PA_ALPHA=1.0; MAX_EXTRA="${MAX_EXTRA:-16}"
        # `${VAR:-default}` so env overrides (e.g. `PA_NUM_PASSES=1` for
        # the 1-pass comparison run) take precedence over the defaults.
        PA_NUM_PASSES="${PA_NUM_PASSES:-2}"
        PA_NORMALIZE="${PA_NORMALIZE:-1}"
        PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS:-8}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-mips}"
        ;;
    gist)
        # GIST 1M — ParlayANN-aligned recipe from
        # `../ParlayANN/algorithms/vamana/scripts/gist`:
        #   BUILD_ARGS="-R 100 -L 200 -alpha 1.1 -num_passes 2"
        # the exported `.staged` and downstream PhasedGraph cache
        # match the public PA reference for apples-to-apples.
        BASE_FVECS="data/gist/gist_base.fvecs"
        QUERY_FVECS="data/gist/gist_query.fvecs"
        GT_IVECS="data/gist/gist_groundtruth.ivecs"
        MAX_POINTS=""           # full 1M
        PA_DIR_NAME="gist"
        PA_R=100; PA_L=200; PA_ALPHA=1.1; MAX_EXTRA="${MAX_EXTRA:-16}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-Euclidian}"
        ;;
    deep10m)
        # Yandex Deep10M (96-dim CNN features, L2 ground truth) — PA's
        # `vamana/scripts/deep10M` recipe verbatim:
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1.05 -num_passes 2 -quantize_bits 8"
        # The raw .fvecs must already be present at the paths below —
        # download from Yandex Deep1B (first 10M) or BigANN benchmark.
        # `convert.py` zero-pads D=96 → D=128 to match the rust-side
        # monomorphisation.
        BASE_FVECS="data/deep10m/deep10m_base.fvecs"
        QUERY_FVECS="data/deep10m/deep10m_query.fvecs"
        GT_IVECS="data/deep10m/deep10m_groundtruth.ivecs"
        MAX_POINTS=""           # full 10M
        PA_DIR_NAME="deep10M"
        PA_R=64; PA_L=128; PA_ALPHA=1.05; MAX_EXTRA="${MAX_EXTRA:-16}"
        PA_NUM_PASSES="${PA_NUM_PASSES:-2}"
        PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS:-8}"
        PA_VERBOSE="${PA_VERBOSE:-1}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-Euclidian}"
        ;;
    fashion-mnist)
        # Fashion-MNIST — 60K × 784-D image vectors (28×28 flattened),
        # L2. PA's recipe (`vamana/scripts/fashion` verbatim):
        #   BUILD_ARGS="-R 40 -L 80 -alpha 1.1 -num_passes 2 -quantize_bits 8"
        #   QUERY_ARGS="-quantize_bits 8"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/fashion-mnist/fashion-mnist-784-euclidean_base.fvecs"
        QUERY_FVECS="data/fashion-mnist/fashion-mnist-784-euclidean_query.fvecs"
        GT_IVECS="data/fashion-mnist/fashion-mnist-784-euclidean_groundtruth.ivecs"
        MAX_POINTS=""           # full 60K
        PA_DIR_NAME="fashion-mnist-784-euclidean"
        PA_R=40; PA_L=80; PA_ALPHA=1.1; MAX_EXTRA="${MAX_EXTRA:-16}"
        PA_NUM_PASSES="${PA_NUM_PASSES:-2}"
        PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS:-8}"
        PA_VERBOSE="${PA_VERBOSE:-1}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-Euclidian}"
        ;;
    msmarco_bert_1M)
        # MS-MARCO BERT 1M — 1M passages embedded with sentence-
        # transformers/msmarco-bert-base-dot-v5 (D=768, dot-product).
        # Vectors are emitted by `data/embed_msmarco_bert.py`. PA's
        # recipe (`vamana/scripts/msmarco_websearch`):
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1 -num_passes 1 -quantize_bits 8"
        #   TYPE_ARGS="-data_type float -dist_func mips -file_type bin"
        BASE_FVECS="data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs"
        QUERY_FVECS="data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs"
        GT_IVECS="data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs"
        MAX_POINTS=""           # full 1M
        PA_DIR_NAME="MSMarcoBert1M"
        PA_R=64; PA_L=128; PA_ALPHA=1.0; MAX_EXTRA="${MAX_EXTRA:-16}"
        PA_NUM_PASSES="${PA_NUM_PASSES:-1}"
        PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS:-8}"
        PA_VERBOSE="${PA_VERBOSE:-1}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-mips}"
        ;;
    wiki_ada_1M)
        # Wikipedia ada-002 1M — 1M passages × 1536-D OpenAI ada-002
        # embeddings, sourced from nlpkevinl/wikipedia_openai_embeddings
        # via `data/load_wiki_ada_1M.py`. ada-002 outputs are unit-norm
        # so MIPS == cosine natively. Build recipe: high-D shape
        # (R=100 L=200 α=1.05) with `-dist_func mips` since the metric
        # is dot-product, not L2.
        BASE_FVECS="data/wiki_ada_1M/wiki_ada_1M_base.fvecs"
        QUERY_FVECS="data/wiki_ada_1M/wiki_ada_1M_query.fvecs"
        GT_IVECS="data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs"
        MAX_POINTS=""           # full 1M
        PA_DIR_NAME="WikiAda1M"
        PA_R=100; PA_L=200; PA_ALPHA=1.05; MAX_EXTRA="${MAX_EXTRA:-16}"
        PA_NUM_PASSES="${PA_NUM_PASSES:-2}"
        PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS:-}"
        PA_VERBOSE="${PA_VERBOSE:-1}"
        PA_DIST_FUNC="${PA_DIST_FUNC:-mips}"
        ;;
    *)
        echo "Unknown DATASET=$DATASET" >&2
        exit 2
        ;;
esac

if [ -z "${PA_ROOT:-}" ]; then
    echo "ERROR: PA_ROOT env var must be set (path to your ParlayANN checkout)" >&2
    echo "  e.g. PA_ROOT=/path/to/ParlayANN DATASET=glove100 bash $0" >&2
    exit 2
fi
# Resolve PA_ROOT to an absolute path. The PA build invokes its
# `neighbors` binary via `cd "$PA_VAMANA" && ./neighbors ...`, so any
# relative paths derived from PA_ROOT (`PA_OUT_DIR/base.fbin`,
# `GRAPH_OUT`, ...) would break after the cd. Resolving once here
# means the caller can hand in either `../ParlayANN` or
# `/abs/path/to/ParlayANN` interchangeably.
if [ ! -d "$PA_ROOT" ]; then
    echo "ERROR: PA_ROOT='$PA_ROOT' is not a directory" >&2
    exit 2
fi
PA_ROOT="$(cd "$PA_ROOT" && pwd)"
PA_OUT_DIR="$PA_ROOT/data/$PA_DIR_NAME"
PA_VAMANA="$PA_ROOT/algorithms/vamana"
# `LOCAL_PCT` is the top-X% partition cutoff (% of per-node degree)
# fed to PA's `-local_pct`. Defaults to 60. Including it in the
# stub keeps generated artifacts disambiguated when sweeping the
# parameter (e.g. ex16_pct60 vs ex16_pct70).
LOCAL_PCT="${LOCAL_PCT:-60}"
ORION_STUB="${PA_DIR_NAME}_ex${MAX_EXTRA}_pct${LOCAL_PCT}"
ORION_OUT="$PA_OUT_DIR/${ORION_STUB}.staged"

# ParlayANN's per-dataset scripts (e.g. `algorithms/vamana/scripts/gist`)
# save the built graph as `graphs/graph_<R>_<alpha>` and pass it to
# downstream search via `-graph_path`. Match that naming so the same
# artifact can be reused by PA's own query benchmark without rebuilding
# (~9 min on GIST 1M). `%g` strips trailing zeros from alpha (1.10 →
# "1.1", 1.0 → "1", 1.15 → "1.15") to match PA's filename convention.
PA_ALPHA_TAG="$(printf '%g' "$PA_ALPHA")"
GRAPH_OUT_DIR="$PA_OUT_DIR/graphs"
GRAPH_OUT="$GRAPH_OUT_DIR/graph_${PA_R}_${PA_ALPHA_TAG}"

mkdir -p "$PA_OUT_DIR" "$GRAPH_OUT_DIR"

echo "═══ Step 1/3: convert fvecs → fbin ($DATASET) ═══"
CONVERT_ARGS=(
    --base-fvecs "$BASE_FVECS"
    --query-fvecs "$QUERY_FVECS"
    --gt-ivecs "$GT_IVECS"
    --out-dir "$PA_OUT_DIR"
)
if [ -n "$MAX_POINTS" ]; then
    CONVERT_ARGS+=(--max-base-points "$MAX_POINTS")
fi
"$PY" "$ROOT/benchmark/scripts/parlayann_convert.py" "${CONVERT_ARGS[@]}"

# When the base is truncated, the passed-through gt.bin uses IDs from
# the original larger set — invalid against our smaller base. Recompute
# brute-force to get correct top-k against the truncated base.
if [ -n "$MAX_POINTS" ]; then
    echo "═══ Step 2/3: brute-force recompute gt against truncated base ═══"
    "$PY" "$ROOT/benchmark/scripts/compute_gt_brute.py" \
        --base-fbin "$PA_OUT_DIR/base.fbin" \
        --query-fbin "$PA_OUT_DIR/query.fbin" \
        --out "$PA_OUT_DIR/gt.bin" \
        -k 100
else
    echo "═══ Step 2/3: gt passed through from original .ivecs ═══"
fi

PA_NUM_PASSES="${PA_NUM_PASSES:-1}"
# `PA_NORMALIZE=1` tells `./neighbors` to L2-normalize base+queries before
# building — needed for angular datasets where our `glove*_aligned`
# orion config reads pre-normalized fvecs.
PA_NORMALIZE_FLAG=""
if [ "${PA_NORMALIZE:-0}" = "1" ]; then
    PA_NORMALIZE_FLAG="-normalize"
fi
# `PA_QUANTIZE_BITS=N` adds `-quantize_bits N` (PA's u8/u16 quantized
# rerank sidecar). Empty = omit. Mirrors PA's published per-dataset
# recipe (e.g. SIFT uses `-quantize_bits 8`).
PA_QUANTIZE_FLAG=""
if [ -n "${PA_QUANTIZE_BITS:-}" ]; then
    PA_QUANTIZE_FLAG="-quantize_bits ${PA_QUANTIZE_BITS}"
fi
PA_VERBOSE_FLAG=""
if [ "${PA_VERBOSE:-0}" = "1" ]; then
    PA_VERBOSE_FLAG="-verbose"
fi

echo "═══ Step 3/3: ParlayANN build → $ORION_OUT  (R=$PA_R L=$PA_L α=$PA_ALPHA passes=$PA_NUM_PASSES normalize=${PA_NORMALIZE:-0} qbits=${PA_QUANTIZE_BITS:-none} ex=$MAX_EXTRA pct=$LOCAL_PCT) ═══"
echo "                                graph → $GRAPH_OUT (for PA's own search re-runs)"
(cd "$PA_VAMANA" && PARLAY_NUM_THREADS=8 ./neighbors \
    -R "$PA_R" -L "$PA_L" -alpha "$PA_ALPHA" -num_passes "$PA_NUM_PASSES" \
    -data_type float -dist_func "$PA_DIST_FUNC" $PA_NORMALIZE_FLAG \
    $PA_QUANTIZE_FLAG $PA_VERBOSE_FLAG \
    -max_extra "$MAX_EXTRA" -local_pct "$LOCAL_PCT" \
    -base_path "$PA_OUT_DIR/base.fbin" \
    -query_path "$PA_OUT_DIR/query.fbin" \
    -gt_path "$PA_OUT_DIR/gt.bin" \
    -res_path "/tmp/parlayann_${DATASET}_prep.csv" \
    -graph_outfile "$GRAPH_OUT" \
    -staged_outfile "$ORION_OUT" 2>&1 | tail -20)

echo
echo "Done. Artifacts in $PA_OUT_DIR:"
ls -lh "$PA_OUT_DIR"
echo
echo "Next: DATASET=$DATASET bash benchmark/scripts/sweep_orion_vs_diskann_vs_parlayann.sh"
