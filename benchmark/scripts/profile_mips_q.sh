#!/usr/bin/env bash
# Profile **only** the MIPS-Q search phase on Apple Silicon.
#
# The binary runs load → normalize → calibrate → build quantized sidecar,
# then writes `/tmp/staged_sweep_<PID>.ready` and sleeps 5 s. While it
# sleeps we attach `xctrace` to the live PID so the CPU-Profile trace
# excludes all the prep work — only the actual beam-search sweep ends up
# in the samples. (Mirrors `run_search_profile.sh` / `search_profile.rs`.)
#
# Usage:
#   bash benchmark/scripts/profile_mips_q.sh                    # glove100, L=[16], PF=4
#   bash benchmark/scripts/profile_mips_q.sh glove100
#   SEARCH_LIST_SIZES=16,32,64 bash benchmark/scripts/profile_mips_q.sh
#   STAGED_PF_BATCH=8 bash benchmark/scripts/profile_mips_q.sh   # try default PF
#
# Env knobs:
#   DATASET              dataset key (default glove100)
#   SEARCH_LIST_SIZES    comma-separated L schedule (default 16 — the
#                        smallest L so the hot beam loop dominates the
#                        profile rather than per-query overhead)
#   STAGED_PF_BATCH      prefetch lookahead, forwarded to the binary
#                        (default 4 — MSHR sweet spot on M2 for MIPS-Q)
#   STAGED_GRAPH         `pa` loads the ParlayANN-built PhasedGraph cache
#                        (default `pa`, our production comparison config)
#   METRIC               override the per-dataset default metric
#                        (usually unset → glove100 picks mips-q)
#   OUT_DIR              where the .trace bundle lands
#                        (default /tmp/staged_trace_<dataset>_<stamp>)
#
# Output:
#   <OUT_DIR>.trace  — open with Instruments.app (`open <path>.trace`)
#   <OUT_DIR>.log    — stdout/stderr from staged_sweep (sweep results)

set -euo pipefail

DATASET="${DATASET:-${1:-glove100}}"
SEARCH_LIST_SIZES="${SEARCH_LIST_SIZES:-16}"
export STAGED_PF_BATCH="${STAGED_PF_BATCH:-4}"
export STAGED_GRAPH="${STAGED_GRAPH:-pa}"
export STAGED_PROFILE_MARKER=1

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Use the workspace's `[profile.profiling]` (inherits release with
# `strip = false`, `debug = 2`, `lto = "thin"`, codegen-units = 4).
# The default `[profile.release]` has `strip = "symbols"` which wipes
# the symbol table after linking — Instruments then shows raw
# addresses with no function names. The `profiling` profile keeps
# full DWARF + symbols so xctrace's CPU Profile demangles every
# Rust frame, and `force-frame-pointers=yes` ensures the time
# profiler can unwind past leaf functions.
echo "── Build --profile profiling with frame pointers (no strip, full debuginfo) ──"
RUSTFLAGS="-C force-frame-pointers=yes" \
    cargo build --profile profiling --bin staged_sweep 2>&1 | tail -2

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/tmp/staged_trace_${DATASET}_${STAMP}}"
mkdir -p "$(dirname "$OUT_DIR")"
TRACE_OUT="${OUT_DIR}.trace"
LOG="${OUT_DIR}.log"

BIN="./target/profiling/staged_sweep"
METRIC_ARG=()
if [ -n "${METRIC:-}" ]; then
    METRIC_ARG=(--metric "$METRIC")
fi

echo "── 60s cooldown before sampling ──"
sleep 60

echo "── Config"
echo "    DATASET            = $DATASET"
echo "    SEARCH_LIST_SIZES  = $SEARCH_LIST_SIZES"
echo "    STAGED_PF_BATCH    = $STAGED_PF_BATCH"
echo "    STAGED_GRAPH       = $STAGED_GRAPH"
echo "    METRIC override    = ${METRIC:-<default-per-dataset>}"
echo "    Trace output       = $TRACE_OUT"
echo "    Stdout log         = $LOG"

# ── Step 1: launch staged_sweep in background, watch for ready marker ─────
# The binary does all prep (load/normalize/calibrate/quantize) and then
# writes the marker before sleeping. We attach xctrace during that 5 s
# window — `xctrace record --attach` auto-stops when the target exits.
echo ""
echo "── launching staged_sweep ──"
"$BIN" "$DATASET" \
    --search-list-sizes "$SEARCH_LIST_SIZES" \
    ${METRIC_ARG[@]+"${METRIC_ARG[@]}"} \
    > "$LOG" 2>&1 &
PID=$!

MARKER="/tmp/staged_sweep_${PID}.ready"
echo "    PID=$PID, waiting for marker $MARKER ..."
# Fail-safe: give prep at most ~5 min (load + build qdm8 on 1.18M × 100
# is usually 20-60 s).
for _ in $(seq 1 600); do
    if [ -f "$MARKER" ]; then
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "staged_sweep exited before writing marker — see $LOG" >&2
        tail -30 "$LOG" >&2 || true
        exit 1
    fi
    sleep 0.5
done
if [ ! -f "$MARKER" ]; then
    echo "timed out waiting for marker" >&2
    kill "$PID" 2>/dev/null || true
    exit 1
fi
echo "    marker seen — prep done, attaching xctrace"

# ── Step 2: attach xctrace to the running PID ─────────────────────────────
# `CPU Profile` template = time-profiler at 1 ms resolution; records
# user+kernel stacks. Trace stops when the target exits (staged_sweep
# finishes its sweep and the process terminates naturally).
xcrun xctrace record \
    --template 'CPU Profile' \
    --output "$TRACE_OUT" \
    --attach "$PID" &
XPID=$!

# Wait for staged_sweep to finish, then the trace recorder.
wait "$PID" 2>/dev/null || true
wait "$XPID" 2>/dev/null || true

echo ""
echo "── staged_sweep output ──"
tail -20 "$LOG"

echo ""
echo "Trace saved: $TRACE_OUT"
echo "Open with:   open '$TRACE_OUT'      # launches Instruments.app"
