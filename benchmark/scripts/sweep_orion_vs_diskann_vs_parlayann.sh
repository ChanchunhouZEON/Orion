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
# Per-dataset cascade triples (set automatically by `orion`'s
# `Cascade::default_for_dataset`):
#   sift     → (none, l2-u8, f32)         direct u8 admission, no prefilter
#   glove25  → (none, mips-i8, ip-f32)    i8 sdot admission, MIPS
#   glove100 → (none, mips-i8, ip-f32)    i8 sdot admission, MIPS
#   gist     → (jl, l2-kt, f32)           JL prefilter + i8 kernel-trick L2
#   deep10m  → (none, l2-u8, f32)         L2 low-D, mirrors SIFT
#   fashion-mnist → (none, l2-kt, f32)    L2 D=784, 60K — kernel-trick admission
#   msmarco_bert_1M → (none, mips-i8, ip-f32)  MIPS high-D — sentence-BERT cosine
#   wiki_ada_1M  → (jl, mips-i8, ip-f32)  MIPS D=1536 — OpenAI ada-002 cosine
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
#   PA_ROOT=/path/to/ParlayANN DATASET=sift     bash benchmark/scripts/sweep_orion_vs_parlayann.sh
#   PA_ROOT=/path/to/ParlayANN DATASET=glove100 bash benchmark/scripts/sweep_orion_vs_parlayann.sh

set -euo pipefail

DATASET="${DATASET:-sift}"
NUM_RUNS="${NUM_RUNS:-3}"
# 60s is the quick-iteration default — enough for steady-state runs
# but not for the first run after a hot build. When you need the
# tightest CV (for final JSON / plot numbers), use `COOLDOWN_S=180`:
# 180s is the empirically-tuned floor on M2 to drop inter-run CV from
# ~50% down to ~10-15%. `COOLDOWN_S` applies both before the first
# run and between consecutive runs.
COOLDOWN_S="${COOLDOWN_S:-1}"

# Default orion search env — applied unless the caller already exported them.
: "${ORION_GRAPH:=pa}"
: "${ORION_DSTREAM_LA_Q:=10}"
: "${ORION_DSTREAM_LA_TRUTH:=6}"
: "${ORION_DSTREAM_SINK_BURST:=12}"
export ORION_GRAPH ORION_DSTREAM_LA_Q ORION_DSTREAM_LA_TRUTH ORION_DSTREAM_SINK_BURST

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
#   - DEFAULT_MAX_EXTRA        per-dataset PhasedGraph extras cap
# (Staged cascade triple is picked automatically by
# `orion`'s `Cascade::default_for_dataset` — no `--metric`
# / `--prefilter` / `--admission` / `--rerank` overrides needed here.)
# ParlayANN checkout. Defaults to the recommended sibling clone
# (`../ParlayANN`); override on the command line for any other layout:
#   PA_ROOT=/path/to/ParlayANN DATASET=$DATASET bash $0
PA_ROOT="${PA_ROOT:-../ParlayANN}"
# Resolve PA_ROOT to absolute so the per-dataset `cd "$PA_DIR" &&
# ./neighbors ...` invocations still see the right `$PA_OUT_DIR/*`
# paths (callers can hand in `../ParlayANN` interchangeably with
# `/abs/path`).
if [ ! -d "$PA_ROOT" ]; then
    echo "ERROR: PA_ROOT='$PA_ROOT' is not a directory" >&2
    echo "  set PA_ROOT to your ParlayANN checkout, e.g. PA_ROOT=/path/to/ParlayANN bash $0" >&2
    exit 2
fi
PA_ROOT="$(cd "$PA_ROOT" && pwd)"
# ── Per-dataset orion runners ────────────────────────────────────────────
# Each function spells out the cascade triple (`--prefilter`,
# `--admission`, `--rerank`) explicitly for that dataset and runs the
# orion binary inside a `( ... )` sub-shell so any env tweaks stay
# scoped — leakage between runs would silently bias QPS.
#
# All four cascades mirror what `Cascade::default_for_dataset` picks
# in `benchmark/src/bin/orion.rs`, but having them inline
# here makes per-dataset A/B tuning (e.g. swap GIST to `--admission
# l2-u16` for PA-bit-exact, or SIFT to `--prefilter jl` to probe the
# high-D-style cascade on low-D) a one-line change at the call site.

# SIFT 1M (D=128, L2, R=64 α=1.15):
#   none → l2-u8 → f32      direct u8 admission; D=128 fits 1 cache
#                           line so the JL prefilter is pure overhead
#                           (~-46% QPS in earlier A/B).
orion_run_sift() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion sift \
                --prefilter none --admission l2-u8 --rerank f32
    )
}

# GloVe-25 (D=25, MIPS-angular, R=100 α=1):
#   none → mips-i8 → ip-f32  i8 sdot on the unit-sphere base; D=25 is
#                            far below any prefilter break-even.
orion_run_glove25() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion glove25 \
                --prefilter none --admission mips-i8 --rerank ip-f32
    )
}

# GloVe-100 (D=100, MIPS-angular, R=100 α=1):
#   none → mips-i8 → ip-f32  same cascade as glove25; the i8 sdot is
#                            already DRAM-fed at this D.
orion_run_glove100() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion glove100 \
                --prefilter none --admission mips-i8 --rerank ip-f32
    )
}

# GIST (D=960, L2, R=100 α=1.1):
#   jl → l2-kt → f32         JL signature (1 cache line / vert) gates
#                            ~67% of candidates ahead of the i8 sdot
#                            admission; kernel-trick reconstructs L2
#                            via ‖q‖² + ‖x‖² − 2·⟨q,x⟩.
orion_run_gist() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion gist \
                --prefilter jl --admission l2-kt --rerank f32
    )
}

# Deep10M (native D=96 → padded D=128, L2, R=64 α=1.05):
#   none → l2-u8 → f32       L2 low-D. PA uses `-dist_func Euclidian`
#                            on the raw Deep10M; we mirror with direct
#                            u8 admission (lpv=1, no prefilter
#                            overhead). 10× the vertex count of SIFT
#                            so the admission slab is ~1.3 GB i8 —
#                            DRAM-streaming territory, sink-burst
#                            prefetch earns its keep here too.
orion_run_deep10m() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion deep10m \
                --prefilter none --admission l2-u8 --rerank f32
    )
}

# fashion-mnist (D=784, L2, R=40 α=1.1):
#   none → l2-kt → f32        60K too small for JL prefilter setup to
#                             amortise, but D=784 is wide enough that
#                             L2-Kt's per-vert ‖x‖² precompute (single
#                             dot product per compare instead of L2's
#                             two subtractions per dim) wins ~30% vs
#                             naive L2-U8.
orion_run_fashion_mnist() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion fashion-mnist \
                --prefilter none --admission l2-kt --rerank f32
    )
}

# msmarco_bert_1M (D=768, MIPS, R=64 α=1.0):
#   none → mips-i8 → ip-f32   No prefilter — empirical sweep showed JL
#                             actively hurts on this dataset (setup tax
#                             dominates at high L; filter drops some
#                             genuine top-K candidates). MIPS-i8 +
#                             f32-IP rerank without prefilter is the
#                             strict-pareto winner.
orion_run_msmarco_bert_1M() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion msmarco_bert_1M \
                --prefilter none --admission mips-i8 --rerank ip-f32
    )
}

# wiki_ada_1M (D=1536, MIPS, R=100 α=1.05):
#   jl → mips-i8 → ip-f32     Same shape as msmarco_bert_1M, just
#                             wider: 96 admission lines/vert instead
#                             of 48. ada-002 vectors are unit-norm
#                             by construction so no load-time renorm.
orion_run_wiki_ada_1M() {
    (
        ORION_STAGED_FILE="$ORION_FILE" \
            ./target/release/orion wiki_ada_1M \
                --prefilter jl --admission mips-i8 --rerank ip-f32
    )
}

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
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_sift
        ORION_CASCADE_LABEL="none → l2-u8 → f32"
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
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_glove25
        ORION_CASCADE_LABEL="none → mips-i8 → ip-f32"
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
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_glove100
        ORION_CASCADE_LABEL="none → mips-i8 → ip-f32"
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
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_gist
        ORION_CASCADE_LABEL="jl → l2-kt → f32"
        ;;
    deep10m)
        # `vamana/scripts/deep10M` (verbatim — PA's published recipe):
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1.05 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/deep10m/deep10m_base.fvecs"
        QUERY_FVECS="data/deep10m/deep10m_query.fvecs"
        GT_IVECS="data/deep10m/deep10m_groundtruth.ivecs"
        PA_DIR_NAME="deep10M"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=64
        PA_BUILD_L=128
        PA_BUILD_ALPHA=1.05
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 1 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_deep10m
        ORION_CASCADE_LABEL="none → l2-u8 → f32"
        ;;
    fashion-mnist)
        # `vamana/scripts/fashion` (verbatim — PA's published recipe):
        #   BUILD_ARGS="-R 40 -L 80 -alpha 1.1 -num_passes 2 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 8 -verbose"
        #   TYPE_ARGS="-data_type float -dist_func Euclidian -file_type bin"
        BASE_FVECS="data/fashion-mnist/fashion-mnist-784-euclidean_base.fvecs"
        QUERY_FVECS="data/fashion-mnist/fashion-mnist-784-euclidean_query.fvecs"
        GT_IVECS="data/fashion-mnist/fashion-mnist-784-euclidean_groundtruth.ivecs"
        PA_DIR_NAME="fashion-mnist-784-euclidean"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=40
        PA_BUILD_L=80
        PA_BUILD_ALPHA=1.1
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="Euclidian"
        PA_NORMALIZE_FLAG=""
        PA_QUERY_ARGS=(-quantize_bits 8 -verbose)
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_fashion_mnist
        ORION_CASCADE_LABEL="none → l2-kt → f32"
        ;;
    msmarco_bert_1M)
        # `vamana/scripts/msmarco_websearch` (PA's published MS-MARCO recipe):
        #   BUILD_ARGS="-R 64 -L 128 -alpha 1 -num_passes 1 -quantize_bits 8 -verbose"
        #   QUERY_ARGS="-quantize_bits 16 -quantize_mode 5 -verbose -rerank_factor 2"
        #   TYPE_ARGS="-data_type float -dist_func mips -file_type bin"
        BASE_FVECS="data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs"
        QUERY_FVECS="data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs"
        GT_IVECS="data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs"
        PA_DIR_NAME="MSMarcoBert1M"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=64
        PA_BUILD_L=128
        PA_BUILD_ALPHA=1.0
        PA_BUILD_PASSES=1
        PA_BUILD_QBITS=8
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG=""        # GT is brute-force MIPS on raw f32
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 5 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_msmarco_bert_1M
        ORION_CASCADE_LABEL="none → mips-i8 → ip-f32"
        ;;
    wiki_ada_1M)
        # OpenAI ada-002 + Wikipedia 1M (sourced from
        # nlpkevinl/wikipedia_openai_embeddings via load_wiki_ada_1M.py).
        # Build recipe: high-D shape (R=100 L=200 α=1.05 num_passes=2)
        # with `-dist_func mips` since ada-002 embeddings are
        # dot-product / cosine, not L2. ada-002 outputs are unit-norm
        # so MIPS == cosine ranking natively.
        BASE_FVECS="data/wiki_ada_1M/wiki_ada_1M_base.fvecs"
        QUERY_FVECS="data/wiki_ada_1M/wiki_ada_1M_query.fvecs"
        GT_IVECS="data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs"
        PA_DIR_NAME="WikiAda1M"
        DEFAULT_MAX_POINTS=""
        PA_BUILD_R=100
        PA_BUILD_L=200
        PA_BUILD_ALPHA=1.05
        PA_BUILD_PASSES=2
        PA_BUILD_QBITS=""
        PA_DIST_FUNC="mips"
        PA_NORMALIZE_FLAG=""        # ada-002 outputs are already unit-norm
        PA_QUERY_ARGS=(-quantize_bits 16 -quantize_mode 5 -verbose -rerank_factor 2)
        DEFAULT_MAX_EXTRA=16
        ORION_RUN=orion_run_wiki_ada_1M
        ORION_CASCADE_LABEL="jl → mips-i8 → ip-f32"
        ;;
    *)
        echo "Unknown DATASET=$DATASET (expected: sift | glove25 | glove100 | gist | deep10m | fashion-mnist | msmarco_bert_1M | wiki_ada_1M)" >&2
        exit 2
        ;;
esac

MAX_POINTS="${MAX_POINTS:-$DEFAULT_MAX_POINTS}"
MAX_EXTRA="${MAX_EXTRA:-$DEFAULT_MAX_EXTRA}"

# Derive ORION_FILE from MAX_EXTRA. The `_pct60` suffix matches PA's
# default `local_pct=60` in `staged_export.h`, mirrored by
# `ORION_LOCAL_PCT=60` on the staged side.
PA_DATA_DIR="$PA_ROOT/data/$PA_DIR_NAME"
DEFAULT_ORION_FILE="$PA_DATA_DIR/${PA_DIR_NAME}_ex${MAX_EXTRA}_pct60.staged"
ORION_FILE="${ORION_FILE:-$DEFAULT_ORION_FILE}"
PA_BASE="${PA_BASE:-$PA_DATA_DIR/base.fbin}"
PA_QUERY="${PA_QUERY:-$PA_DATA_DIR/query.fbin}"
PA_GT="${PA_GT:-$PA_DATA_DIR/gt.bin}"

PA_DIR="${PA_DIR:-$PA_ROOT/algorithms/vamana}"

OUT_JSON="${OUT_JSON:-visualizations/sweep_orion_vs_parlayann_${DATASET}.json}"

REQUIRED_FILES=("$ORION_FILE" "$PA_BASE" "$PA_QUERY" "$PA_GT")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$f" ]; then
        echo "Missing ParlayANN data for DATASET=$DATASET: $f" >&2
        echo "(Run prepare_parlayann_data.sh first to produce both the .staged export and the fbin/gt files.)" >&2
        exit 2
    fi
done

echo "═══ Build release ═══"
cargo build --release --bin orion --bin diskann_sweep 2>&1 | tail -1

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cat <<EOF
═══ Comparison spec ═══
  DATASET            = $DATASET
  ORION cascade     = $ORION_CASCADE_LABEL    (via $ORION_RUN)
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

# ── Staged runs ────────────────────────────────────────────────────────────
for i in $(seq 1 "$NUM_RUNS"); do
    echo "── Staged run $i/$NUM_RUNS ──"
    $ORION_RUN 2>&1 \
        | tee "$TMPDIR/orion_$i.out" \
        | grep -E "^  L=|Calibrated|Cascade:|═══"
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
for i in $(seq 1 "$NUM_RUNS"); do
    echo "── DiskANN run $i/$NUM_RUNS ──"
    ./target/release/diskann_sweep "$DATASET" 2>&1 \
        | tee "$TMPDIR/diskann_$i.out" \
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
        -num_passes "$PA_BUILD_PASSES" "${PA_QBITS_FLAG[@]}" -verbose \
        -data_type float -dist_func "$PA_DIST_FUNC" $PA_NORMALIZE_FLAG -file_type bin \
        -base_path "$PA_BASE" \
        -graph_outfile "$PA_GRAPH_OUT" 2>&1) \
        | tee "$TMPDIR/pa_build.out" \
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
