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

source "$SCRIPT_DIR/_common.sh"
setup_python_env numpy || exit 1

LOCAL_PCT="${LOCAL_PCT:-60}"

source "$SCRIPT_DIR/_parlay.sh"
load_parlay_recipe "$DATASET"
MAX_POINTS="${MAX_POINTS:-$DEFAULT_MAX_POINTS}"
MAX_EXTRA="${MAX_EXTRA:-$DEFAULT_MAX_EXTRA}"
PA_R="${PA_R:-$PA_BUILD_R}"
PA_L="${PA_L:-$PA_BUILD_L}"
PA_ALPHA="${PA_ALPHA:-$PA_BUILD_ALPHA}"
PA_NUM_PASSES="${PA_NUM_PASSES:-$PA_BUILD_PASSES}"
PA_QUANTIZE_BITS="${PA_QUANTIZE_BITS-$PA_BUILD_QBITS}"
PA_VERBOSE="${PA_VERBOSE:-1}"
PA_NORMALIZE="${PA_NORMALIZE:-${PA_NORMALIZE_FLAG:+1}}"


# ParlayANN checkout. Defaults to the recommended sibling clone
# (`../ParlayANN`); override on the command line for any other layout:
#   PA_ROOT=/path/to/ParlayANN DATASET=glove100 bash $0
resolve_parlay_root
PA_OUT_DIR="$PA_ROOT/data/$PA_DIR_NAME"
PA_VAMANA="$PA_ROOT/algorithms/vamana"
[[ -x "$PA_VAMANA/neighbors" ]] || { echo "Build $PA_VAMANA/neighbors first" >&2; exit 2; }
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
    --out-dir "$PA_OUT_DIR"
)
if [[ -n "$MAX_POINTS" && "$MAX_POINTS" != 0 ]]; then
    CONVERT_ARGS+=(--max-base-points "$MAX_POINTS" --skip-groundtruth)
else
    CONVERT_ARGS+=(--gt-ivecs "$GT_IVECS")
fi
"$PY" "$ROOT/benchmark/scripts/parlayann_convert.py" "${CONVERT_ARGS[@]}"

# When the base is truncated, the passed-through gt.bin uses IDs from
# the original larger set — invalid against our smaller base. Recompute
# brute-force to get correct top-k against the truncated base.
if [[ -n "$MAX_POINTS" && "$MAX_POINTS" != 0 ]]; then
    GT_METRIC=l2
    if [[ "$PA_DIST_FUNC" = mips ]]; then GT_METRIC=ip; fi
    if [[ "${PA_NORMALIZE:-0}" = 1 ]]; then GT_METRIC=cos; fi
    echo "═══ Step 2/3: brute-force recompute gt against truncated base ═══"
    "$PY" "$ROOT/benchmark/scripts/compute_gt_brute.py" \
        --base-fbin "$PA_OUT_DIR/base.fbin" \
        --query-fbin "$PA_OUT_DIR/query.fbin" \
        --out "$PA_OUT_DIR/gt.bin" \
        --metric "$GT_METRIC" -k 100
else
    echo "═══ Step 2/3: gt passed through from original .ivecs ═══"
fi

PA_NUM_PASSES="${PA_NUM_PASSES:-1}"
# `PA_NORMALIZE=1` tells `./neighbors` to L2-normalize base+queries before
# building — needed for angular datasets where our `glove*_aligned`
# orion config reads pre-normalized fvecs.
PA_NORMALIZE_FLAG=()
if [ "${PA_NORMALIZE:-0}" = "1" ]; then
    PA_NORMALIZE_FLAG=(-normalize)
fi
# `PA_QUANTIZE_BITS=N` adds `-quantize_bits N` (PA's u8/u16 quantized
# rerank sidecar). Empty = omit. Mirrors PA's published per-dataset
# recipe (e.g. SIFT uses `-quantize_bits 8`).
PA_QUANTIZE_FLAG=()
if [ -n "${PA_QUANTIZE_BITS:-}" ]; then
    PA_QUANTIZE_FLAG=(-quantize_bits "$PA_QUANTIZE_BITS")
fi
PA_VERBOSE_FLAG=()
if [ "${PA_VERBOSE:-0}" = "1" ]; then
    PA_VERBOSE_FLAG=(-verbose)
fi

echo "═══ Step 3/3: ParlayANN build → $ORION_OUT  (R=$PA_R L=$PA_L α=$PA_ALPHA passes=$PA_NUM_PASSES normalize=${PA_NORMALIZE:-0} qbits=${PA_QUANTIZE_BITS:-none} ex=$MAX_EXTRA pct=$LOCAL_PCT) ═══"
echo "                                graph → $GRAPH_OUT (for PA's own search re-runs)"
(cd "$PA_VAMANA" && PARLAY_NUM_THREADS=8 ./neighbors \
    -R "$PA_R" -L "$PA_L" -alpha "$PA_ALPHA" -num_passes "$PA_NUM_PASSES" \
    -data_type float -dist_func "$PA_DIST_FUNC" ${PA_NORMALIZE_FLAG[@]+"${PA_NORMALIZE_FLAG[@]}"} -file_type bin \
    ${PA_QUANTIZE_FLAG[@]+"${PA_QUANTIZE_FLAG[@]}"} ${PA_VERBOSE_FLAG[@]+"${PA_VERBOSE_FLAG[@]}"} \
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
