#!/usr/bin/env bash
# Back-to-back sweep of StagedDiskANN vs ParlayANN Vamana on the same
# ParlayANN-built base graph. Runs each implementation NUM_RUNS times
# with COOLDOWN_S seconds between runs, then parses the outputs into a
# single JSON with per-L medians.
#
# **PA invocation matches `../ParlayANN/algorithms/vamana/scripts/{sift,
# glove25,glove100,gist}` byte-for-byte** — same BUILD_ARGS, QUERY_ARGS,
# TYPE_ARGS as PA's published per-dataset recipes. The staged side loads
# the corresponding `.staged` export (produced by
# `prepare_parlayann_data.sh` with the same build params) so both sides
# see the same Vamana graph.
#
# Per-dataset metric assignments (fixed, no METRIC env override):
#   sift     → staged `l2-q`   (single-stage u8 L2 beam + f32 top-20 rerank)
#   glove25  → staged `mips`   (single-phase f32 IP, low-dim)
#   glove100 → staged `mips-q` (PA-style i8 beam + f32 rerank)
#   gist     → staged `l2-q`
#
# Default staged search env (overridable):
#   STAGED_GRAPH=pa
#   STAGED_USE_FILTER=0
#   STAGED_DSTREAM_LA_Q=32
#   STAGED_DSTREAM_LA_TRUTH=128
#
# Env knobs:
#   DATASET        sift | glove25 | glove100 | gist           (default sift)
#   NUM_RUNS       rounds per implementation                   (default 3)
#   COOLDOWN_S     seconds between rounds                      (default 60;
#                  use 180 for paper-grade thermal isolation)
#   MAX_POINTS     rust-side truncation                        (per-dataset default)
#   STAGED_FILE    override ParlayANN `.staged` export path    (auto-derived)
#   OUT_JSON       override output JSON path
#
# Usage:
#   PA_ROOT=/path/to/ParlayANN DATASET=sift     bash benchmark/scripts/sweep_staged_vs_parlayann.sh
#   PA_ROOT=/path/to/ParlayANN DATASET=glove100 bash benchmark/scripts/sweep_staged_vs_parlayann.sh

set -euo pipefail

DATASET="${DATASET:-sift}"
NUM_RUNS="${NUM_RUNS:-3}"
# 60s is the quick-iteration default — enough for steady-state runs
# but not for the first run after a hot build. When you need the
# tightest CV (for final JSON / plot numbers), use `COOLDOWN_S=180`:
# 180s is the empirically-tuned floor on M2 to drop inter-run CV from
# ~50% down to ~10-15%. `COOLDOWN_S` applies both before the first
# run and between consecutive runs.
COOLDOWN_S="${COOLDOWN_S:-60}"

# Default staged search env — applied unless the caller already exported them.
: "${STAGED_GRAPH:=pa}"
: "${STAGED_USE_FILTER:=0}"
: "${STAGED_DSTREAM_LA_Q:=10}"
: "${STAGED_DSTREAM_LA_TRUTH:=6}"
export STAGED_GRAPH STAGED_USE_FILTER STAGED_DSTREAM_LA_Q STAGED_DSTREAM_LA_TRUTH

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# ── Per-dataset resolution ────────────────────────────────────────────────
# PA_ROOT must come from env (no hardcoded user paths). The dataset
# branches set:
#   - PA_DIR_NAME              PA's `data/<dir>/` subdirectory
#   - DEFAULT_MAX_POINTS       rust-side truncation
#   - PA_BUILD_R/L/ALPHA/PASSES/QBITS  matches PA's vamana/scripts/<ds>
#   - PA_DIST_FUNC             "Euclidian" or "mips"
#   - PA_NORMALIZE_FLAG        "-normalize" or ""
#   - PA_QUERY_ARGS            QUERY_ARGS from vamana/scripts/<ds>
#   - STAGED_METRIC            staged_sweep --metric for this dataset
#   - DEFAULT_MAX_EXTRA        per-dataset PhasedGraph extras cap
if [ -z "${PA_ROOT:-}" ]; then
    echo "ERROR: PA_ROOT env var must be set (path to your ParlayANN checkout)" >&2
    echo "  e.g. PA_ROOT=/path/to/ParlayANN DATASET=$DATASET bash $0" >&2
    exit 2
fi
case "$DATASET" in
    sift)
        # `vamana/scripts/sift`:
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1.15 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 8 -verbose"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/sift/sift_base.fvecs"
        QUERY_FVECS="data/sift/sift_query.fvecs"
        GT_IVECS="data/sift/sift_groundtruth.ivecs"
        PA_DIR_NAME="sift1m"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=64
        PA_BUILD_L=128
        PA_BUILD_ALPHA=1.15
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 8 -verbose)
        STAGED_METRIC="l2-q"
        DEFAULT_MAX_EXTRA=16
        ;;
    glove25)
        # `vamana/scripts/glove25`:
        #   BUILD_ARGS="-R 100 -L 200 -alpha 1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func mips -normalize -file_type bin"
        BASE_FVECS="data/glove25/glove-25-angular_base.fvecs"
        QUERY_FVECS="data/glove25/glove-25-angular_query.fvecs"
        GT_IVECS="data/glove25/glove-25-angular_groundtruth.ivecs"
        PA_DIR_NAME="glove25"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG="-normalize"
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2)
        STAGED_METRIC="mips"
        DEFAULT_MAX_EXTRA=16
        ;;
    glove100)
        # `vamana/scripts/glove100`:
        #   BUILD_ARGS="-R 100 -L 200 -alpha 1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func mips -normalize -file_type bin"
        BASE_FVECS="data/glove100/glove-100-angular_base.fvecs"
        QUERY_FVECS="data/glove100/glove-100-angular_query.fvecs"
        GT_IVECS="data/glove100/glove-100-angular_groundtruth.ivecs"
        PA_DIR_NAME="glove100"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG="-normalize"
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2)
        STAGED_METRIC="mips-q"
        DEFAULT_MAX_EXTRA=16
        ;;
    gist)
        # `vamana/scripts/gist`:
        #   BUILD_ARGS="-R 100 -L 200 -alpha 1.1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 3 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/gist/gist_base.fvecs"
        QUERY_FVECS="data/gist/gist_query.fvecs"
        GT_IVECS="data/gist/gist_groundtruth.ivecs"
        PA_DIR_NAME="gist"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1.1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 3 -verbose -rerank_factor 2)
        STAGED_METRIC="l2-q"
        DEFAULT_MAX_EXTRA=16
        ;;
    *)
        echo "Unknown DATASET=$DATASET (expected: sift | glove25 | glove100 | gist)" >&2
        exit 2
        ;;
esac

MAX_POINTS="${MAX_POINTS:-$DEFAULT_MAX_POINTS}"
MAX_EXTRA="${MAX_EXTRA:-$DEFAULT_MAX_EXTRA}"

# Derive STAGED_FILE from MAX_EXTRA. The `_pct60` suffix matches PA's
# default `local_pct=60` in `staged_export.h`, mirrored by
# `STAGED_LOCAL_PCT=60` on the staged side.
PA_DATA_DIR="$PA_ROOT/data/$PA_DIR_NAME"
DEFAULT_STAGED_FILE="$PA_DATA_DIR/${PA_DIR_NAME}_ex${MAX_EXTRA}_pct60.staged"
STAGED_FILE="${STAGED_FILE:-$DEFAULT_STAGED_FILE}"
PA_BASE="${PA_BASE:-$PA_DATA_DIR/base.fbin}"
PA_QUERY="${PA_QUERY:-$PA_DATA_DIR/query.fbin}"
PA_GT="${PA_GT:-$PA_DATA_DIR/gt.bin}"

PA_DIR="${PA_DIR:-$PA_ROOT/algorithms/vamana}"

OUT_JSON="${OUT_JSON:-visualizations/sweep_staged_vs_parlayann_${DATASET}.json}"

REQUIRED_FILES=("$STAGED_FILE" "$PA_BASE" "$PA_QUERY" "$PA_GT")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$f" ]; then
        echo "Missing ParlayANN data for DATASET=$DATASET: $f" >&2
        echo "(Run prepare_parlayann_data.sh first to produce both the .staged export and the fbin/gt files.)" >&2
        exit 2
    fi
done

echo "═══ Build release ═══"
cargo build --release --bin staged_sweep 2>&1 | tail -1

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cat <<EOF
═══ Comparison spec ═══
  DATASET            = $DATASET
  STAGED_METRIC      = $STAGED_METRIC
  STAGED env         = GRAPH=$STAGED_GRAPH  USE_FILTER=$STAGED_USE_FILTER  LA_Q=$STAGED_DSTREAM_LA_Q  LA_T=$STAGED_DSTREAM_LA_TRUTH
  PA build           = -R $PA_BUILD_R -L $PA_BUILD_L -alpha $PA_BUILD_ALPHA -num_passes $PA_BUILD_PASSES -quantize_bits $PA_BUILD_QBITS -verbose
  PA query           = ${PA_QUERY_ARGS[*]}
  PA type            = -data_type float -dist_func $PA_DIST_FUNC $PA_NORMALIZE_FLAG -file_type bin
  STAGED_FILE        = $STAGED_FILE
  NUM_RUNS=$NUM_RUNS  COOLDOWN_S=${COOLDOWN_S}s  max_points=${MAX_POINTS:-all}
EOF

# ── Initial idle ──────────────────────────────────────────────────────────
# Avoid run-1 landing in a throttled window right after the release
# build — gives run 1 the same thermal starting point as runs 2-N.
echo "── initial idle ${COOLDOWN_S}s ──"
sleep "$COOLDOWN_S"

# ── Staged runs ────────────────────────────────────────────────────────────
for i in $(seq 1 "$NUM_RUNS"); do
    echo "── Staged run $i/$NUM_RUNS ──"
    STAGED_STAGED_FILE="$STAGED_FILE" \
        ./target/release/staged_sweep "${DATASET}" --metric "$STAGED_METRIC" 2>&1 \
        | tee "$TMPDIR/staged_$i.out" \
        | grep -E "^  L=|Calibrated|mode|═══"
    if [ "$i" -lt "$NUM_RUNS" ]; then
        echo "── cooldown ${COOLDOWN_S}s ──"
        sleep "$COOLDOWN_S"
    fi
done

echo "── cooldown ${COOLDOWN_S}s before ParlayANN ──"
sleep "$COOLDOWN_S"

# ── ParlayANN runs ─────────────────────────────────────────────────────────
# Build + query in one invocation per run, mirroring how PA's published
# `vamana/scripts/<ds>` invokes `./neighbors`. PA reports one
# "For 10@10 recall = ..." line per query Q value.
for i in $(seq 1 "$NUM_RUNS"); do
    echo "── ParlayANN run $i/$NUM_RUNS ──"
    (cd "$PA_DIR" && PARLAY_NUM_THREADS=8 ./neighbors \
        -R "$PA_BUILD_R" -L "$PA_BUILD_L" -alpha "$PA_BUILD_ALPHA" \
        -num_passes "$PA_BUILD_PASSES" -quantize_bits "$PA_BUILD_QBITS" -verbose \
        "${PA_QUERY_ARGS[@]}" \
        -data_type float -dist_func "$PA_DIST_FUNC" $PA_NORMALIZE_FLAG -file_type bin \
        -base_path "$PA_BASE" \
        -query_path "$PA_QUERY" \
        -gt_path "$PA_GT" \
        -res_path "$TMPDIR/pa_$i.csv" 2>&1) \
        | tee "$TMPDIR/pa_$i.out" \
        | grep -E "^For 10@10 recall"
    if [ "$i" -lt "$NUM_RUNS" ]; then
        echo "── cooldown ${COOLDOWN_S}s ──"
        sleep "$COOLDOWN_S"
    fi
done

# ── Parse + median → JSON ──────────────────────────────────────────────────
echo "═══ Parsing → $OUT_JSON ═══"
python3 "$ROOT/benchmark/scripts/collect_sweep_medians.py" \
    --tmpdir "$TMPDIR" \
    --num-runs "$NUM_RUNS" \
    --dataset "$DATASET" \
    --out "$OUT_JSON"

echo "Done. Result: $OUT_JSON"
