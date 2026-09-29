#!/usr/bin/env bash
# Back-to-back sweep of Orion vs ParlayANN Vamana on the same
# ParlayANN-built base graph. Runs each implementation NUM_RUNS times
# with COOLDOWN_S seconds between runs, then parses the outputs into a
# single JSON with per-L medians.
#
# **PA invocation matches `../ParlayANN/algorithms/vamana/scripts/{sift,
# glove25,glove100,gist}` byte-for-byte** — same BUILD_ARGS, QUERY_ARGS,
# TYPE_ARGS as PA's published per-dataset recipes. The orion side loads
# the corresponding `.staged` export (produced by
# `prepare_parlayann_data.sh` with the same build params) so both sides
# see the same Vamana graph.
#
# Orion cascade settings come directly from sweep.yaml.
#
# Default orion search env (overridable):
#   ORION_GRAPH=pa
#   ORION_DSTREAM_LA_Q=10
#   ORION_DSTREAM_LA_TRUTH=6
#   ORION_DSTREAM_SINK_BURST=12   (sink-time long-range prefetch burst)
#
# Env knobs:
#   DATASET        sift | glove25 | glove100 | gist           (default sift)
#   NUM_RUNS       rounds per implementation                   (default 3)
#   COOLDOWN_S     seconds between rounds                      (default 60;
#                  use 180 for paper-grade thermal isolation)
#   MAX_POINTS     rust-side truncation                        (per-dataset default)
#   ORION_FILE    override ParlayANN `.staged` export path    (auto-derived)
#   OUT_JSON       override output JSON path
#
# Usage:
#   PA_ROOT=/path/to/ParlayANN DATASET=sift     bash benchmark/scripts/sweep_orion_vs_diskann_vs_parlayann.sh
#   PA_ROOT=/path/to/ParlayANN DATASET=glove100 bash benchmark/scripts/sweep_orion_vs_diskann_vs_parlayann.sh

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

# Default orion search env — applied unless the caller already exported them.
: "${ORION_GRAPH:=pa}"
: "${ORION_DSTREAM_LA_Q:=10}"
: "${ORION_DSTREAM_LA_TRUTH:=6}"
: "${ORION_DSTREAM_SINK_BURST:=12}"
export ORION_GRAPH ORION_DSTREAM_LA_Q ORION_DSTREAM_LA_TRUTH ORION_DSTREAM_SINK_BURST

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$SCRIPT_DIR/_parlay.sh"
setup_python_env json

# ── Per-dataset resolution ────────────────────────────────────────────────
# PA_ROOT must come from env (no hardcoded user paths). The dataset
# branches set:
#   - PA_DIR_NAME              PA's `data/<dir>/` subdirectory
#   - DEFAULT_MAX_POINTS       rust-side truncation
#   - PA_BUILD_R/L/ALPHA/PASSES/QBITS  matches PA's vamana/scripts/<ds>
#   - PA_DIST_FUNC             "Euclidian" or "mips"
#   - PA_NORMALIZE_FLAG        "-normalize" or ""
#   - PA_QUERY_ARGS            QUERY_ARGS from vamana/scripts/<ds>
#   - DEFAULT_MAX_EXTRA        per-dataset PhasedGraph extras cap
# (Orion cascade triple is picked automatically by
# `orion`'s `Cascade::default_for_dataset` — no `--metric`
# / `--prefilter` / `--admission` / `--rerank` overrides needed here.)
# ParlayANN checkout. Defaults to the recommended sibling clone
# (`../ParlayANN`); override on the command line for any other layout:
#   PA_ROOT=/path/to/ParlayANN DATASET=$DATASET bash $0
resolve_parlay_root
# Explicit CLI flags take precedence over legacy environment variable names.
orion_run() {
    "$ORION_BIN" "$DATASET" --staged-file "$ORION_FILE" --max-extra "$MAX_EXTRA" --k 10 --threads 8
}

load_parlay_recipe "$DATASET"

MAX_POINTS="${MAX_POINTS:-$DEFAULT_MAX_POINTS}"
MAX_EXTRA="${MAX_EXTRA:-$DEFAULT_MAX_EXTRA}"
# diskann_sweep cannot accept subset-specific GT. Reject rather than compare
# prefix searches against the full-dataset truth or silently ignore MAX_POINTS.
if [[ -n "$MAX_POINTS" && "$MAX_POINTS" != 0 ]]; then
    echo "This comparison requires full presets; subset evaluation needs matching GT support in diskann_sweep" >&2
    exit 2
fi
[[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_RUNS must be positive' >&2; exit 2; }
LOCAL_PCT="${LOCAL_PCT:-60}"

# Derive ORION_FILE from MAX_EXTRA. The `_pct60` suffix matches PA's
# default `local_pct=60` in `staged_export.h`, mirrored by
# `ORION_LOCAL_PCT=60` on the staged side.
PA_DATA_DIR="$PA_ROOT/data/$PA_DIR_NAME"
DEFAULT_ORION_FILE="$PA_DATA_DIR/${PA_DIR_NAME}_ex${MAX_EXTRA}_pct${LOCAL_PCT}.staged"
ORION_FILE="${ORION_FILE:-$DEFAULT_ORION_FILE}"
PA_BASE="${PA_BASE:-$PA_DATA_DIR/base.fbin}"
PA_QUERY="${PA_QUERY:-$PA_DATA_DIR/query.fbin}"
PA_GT="${PA_GT:-$PA_DATA_DIR/gt.bin}"

PA_DIR="${PA_DIR:-$PA_ROOT/algorithms/vamana}"
PA_DIR="$(cd "$PA_DIR" && pwd)"
[[ -x "$PA_DIR/neighbors" ]] || { echo "Build $PA_DIR/neighbors first" >&2; exit 2; }

OUT_JSON="${OUT_JSON:-visualizations/sweep_orion_vs_parlayann_${DATASET}.json}"

REQUIRED_FILES=("$ORION_FILE" "$PA_BASE" "$PA_QUERY" "$PA_GT")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$f" ]; then
        echo "Missing ParlayANN data for DATASET=$DATASET: $f" >&2
        echo "(Run prepare_parlayann_data.sh first to produce both the .staged export and the fbin/gt files.)" >&2
        exit 2
    fi
done

PA_POINT_COUNT=$("$PY" -c 'import struct, sys; print(struct.unpack("<I", open(sys.argv[1], "rb").read(4))[0])' "$PA_BASE")

echo "═══ Build release ═══"
build_binary orion-sweep
build_binary diskann_sweep
ORION_BIN="$(binary_path orion-sweep)"
DISKANN_BIN="$(binary_path diskann_sweep)"

RUN_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$RUN_TMP_DIR"' EXIT

cat <<EOF
═══ Comparison spec ═══
  DATASET            = $DATASET
  ORION cascade     = resolved from sweep.yaml
  ORION env         = GRAPH=$ORION_GRAPH  LA_Q=$ORION_DSTREAM_LA_Q  LA_T=$ORION_DSTREAM_LA_TRUTH  SINK_BURST=$ORION_DSTREAM_SINK_BURST
  PA build           = -R $PA_BUILD_R -L $PA_BUILD_L -alpha $PA_BUILD_ALPHA -num_passes $PA_BUILD_PASSES ${PA_BUILD_QBITS:+-quantize_bits $PA_BUILD_QBITS} -verbose
  PA query           = ${PA_QUERY_ARGS[*]}
  PA type            = -data_type float -dist_func $PA_DIST_FUNC $PA_NORMALIZE_FLAG -file_type bin
  ORION_FILE        = $ORION_FILE
  NUM_RUNS=$NUM_RUNS  COOLDOWN_S=${COOLDOWN_S}s  max_points=${MAX_POINTS:-all}
EOF

# ── Initial idle ──────────────────────────────────────────────────────────
# Avoid run-1 landing in a throttled window right after the release
# build — gives run 1 the same thermal starting point as runs 2-N.
echo "── initial idle ${COOLDOWN_S}s ──"
sleep "$COOLDOWN_S"

# ── Orion runs ────────────────────────────────────────────────────────────
for i in $(seq 1 "$NUM_RUNS"); do
    echo "── Staged run $i/$NUM_RUNS ──"
    orion_run 2>&1 \
        | tee "$RUN_TMP_DIR/orion_$i.out" \
        | grep -E "L=|Calibrated|Cascade:|═══"
    if [ "$i" -lt "$NUM_RUNS" ]; then
        echo "── cooldown ${COOLDOWN_S}s ──"
        sleep "$COOLDOWN_S"
    fi
done

echo "── cooldown ${COOLDOWN_S}s before DiskANN ──"
sleep "$COOLDOWN_S"

# ── DiskANN runs ────────────────────────────────────────────────────────────
# Uses the in-process `DiskANNRunner` (diskann core crate) — same Vamana
# build params (R / L_build / α) as ParlayANN and Orion so the
# graph topology recipe is identical across all three engines. Metric
# defaults are picked inside `diskann_sweep` per the dataset's natural
# distance (Cosine for glove / msmarco / wiki_ada, L2 elsewhere).
# MS-MARCO inputs are normalized in this workflow, so cosine preserves IP ranking.
for i in $(seq 1 "$NUM_RUNS"); do
    echo "── DiskANN run $i/$NUM_RUNS ──"
    "$DISKANN_BIN" "$DATASET"  2>&1 \
        | tee "$RUN_TMP_DIR/diskann_$i.out" \
        | grep -E "^  L=|^DiskANN|^═══"
    if [ "$i" -lt "$NUM_RUNS" ]; then
        echo "── cooldown ${COOLDOWN_S}s ──"
        sleep "$COOLDOWN_S"
    fi
done


echo "── cooldown ${COOLDOWN_S}s before ParlayANN ──"
sleep "$COOLDOWN_S"

# ── ParlayANN: build graph once, reuse across runs ─────────────────────────
# Mirrors PA's published `vamana/scripts/<ds>` two-step recipe (build
# with `-graph_outfile`, then query with `-graph_path`). Graph filename
# follows PA's convention `graphs/graph_<R>_<alpha>` so the artifact
# can be reused by PA's own per-dataset query scripts unchanged.
# `printf '%g'` strips trailing zeros (1.10 → "1.1", 1.0 → "1") to
# match the convention.
PA_ALPHA_TAG="$(printf '%g' "$PA_BUILD_ALPHA")"
PA_GRAPH_OUT="$PA_DATA_DIR/graphs/graph_${PA_BUILD_R}_${PA_ALPHA_TAG}"
mkdir -p "$(dirname "$PA_GRAPH_OUT")"

if [ ! -f "$PA_GRAPH_OUT" ]; then
    echo "── Building ParlayANN graph once → $PA_GRAPH_OUT ──"
    # `-quantize_bits` is omitted on datasets where PA's published recipe
    # doesn't set it (e.g. wiki_ada_1M, where the build side relies on
    # the MIPS u8 admission tier alone). Empty `PA_BUILD_QBITS` → flag
    # drops out entirely.
    PA_QBITS_FLAG=()
    if [ -n "$PA_BUILD_QBITS" ]; then
        PA_QBITS_FLAG=(-quantize_bits "$PA_BUILD_QBITS")
    fi
    (cd "$PA_DIR" && PARLAY_NUM_THREADS=8 ./neighbors \
        -R "$PA_BUILD_R" -L "$PA_BUILD_L" -alpha "$PA_BUILD_ALPHA" \
        -num_passes "$PA_BUILD_PASSES" ${PA_QBITS_FLAG[@]+"${PA_QBITS_FLAG[@]}"} -verbose \
        -data_type float -dist_func "$PA_DIST_FUNC" $PA_NORMALIZE_FLAG -file_type bin \
        -base_path "$PA_BASE" \
        -graph_outfile "$PA_GRAPH_OUT" 2>&1) \
        | tee "$RUN_TMP_DIR/pa_build.out" \
        | tail -5
    echo "── cooldown ${COOLDOWN_S}s after build ──"
    sleep "$COOLDOWN_S"
else
    echo "── Reusing existing ParlayANN graph: $PA_GRAPH_OUT ──"
fi

# ── ParlayANN query-only runs ──────────────────────────────────────────────
# All NUM_RUNS runs share the prebuilt graph via `-graph_path`, so each
# round is just timed search — no graph rebuild between runs.
for i in $(seq 1 "$NUM_RUNS"); do
    echo "── ParlayANN run $i/$NUM_RUNS (query-only) ──"
    (cd "$PA_DIR" && PARLAY_NUM_THREADS=8 ./neighbors \
        "${PA_QUERY_ARGS[@]}" \
        -data_type float -dist_func "$PA_DIST_FUNC" $PA_NORMALIZE_FLAG -file_type bin \
        -base_path "$PA_BASE" \
        -query_path "$PA_QUERY" \
        -gt_path "$PA_GT" \
        -graph_path "$PA_GRAPH_OUT" \
        -res_path "$RUN_TMP_DIR/pa_$i.csv" 2>&1) \
        | tee "$RUN_TMP_DIR/pa_$i.out" \
        | grep -E "^For 10@10 recall"
    if [ "$i" -lt "$NUM_RUNS" ]; then
        echo "── cooldown ${COOLDOWN_S}s ──"
        sleep "$COOLDOWN_S"
    fi
done

# ── Parse + median → JSON ──────────────────────────────────────────────────
echo "═══ Parsing → $OUT_JSON ═══"
"$PY" "$ROOT/benchmark/scripts/collect_sweep_medians.py" \
    --tmpdir "$RUN_TMP_DIR" \
    --num-runs "$NUM_RUNS" \
    --dataset "$DATASET" \
    --num-points "$PA_POINT_COUNT" \
    --out "$OUT_JSON"

echo "Done. Result: $OUT_JSON"
