#!/usr/bin/env bash
# macOS search-only CPU Counters capture. Preserve logs even on failure.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DATASET="${1:?Usage: $0 <dataset> [max_points]}"
MAX_POINTS="${2:-100000}"
BIN="${BIN:-$(binary_path orion-sweep)}"
command -v xctrace >/dev/null || { echo 'xctrace is required (Xcode on macOS)' >&2; exit 2; }
[[ -x "$BIN" ]] || { echo "Build orion-sweep first: $BIN" >&2; exit 2; }
if [[ -n "${RUN_DIR:-}" ]]; then
  mkdir "$RUN_DIR"
else
  RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/orion-profile.XXXXXX")"
fi
LOG="$RUN_DIR/search.log"
TRACE_OUT="$RUN_DIR/counters.trace"
PID=""; XPID=""; MARKER=""
cleanup() {
  # Only terminate children created by this invocation; retain user-visible output.
  if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; fi
  if [[ -n "$XPID" ]]; then kill "$XPID" 2>/dev/null || true; fi
  if [[ -n "$MARKER" ]]; then rm -f "$MARKER"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ORION_PROFILE_MARKER=1 "$BIN" "$DATASET" --max-points "$MAX_POINTS" \
  --trials "${TRIALS:-30}" >"$LOG" 2>&1 &
PID=$!
MARKER="/tmp/orion_sweep_${PID}.ready"
deadline=$((SECONDS + ${PROFILE_WAIT_SECONDS:-1800}))
while [[ ! -f "$MARKER" ]]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID" || status=$?
    PID=""
    cat "$LOG" >&2
    echo 'Search exited before the profiling marker' >&2
    exit "${status:-1}"
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for search preparation; see $LOG" >&2
    exit 1
  fi
  sleep 0.3
done
xctrace record --template "CPU Counters" --attach "$PID" \
  --time-limit "${PROFILE_DURATION:-30s}" --output "$TRACE_OUT" &
XPID=$!
wait "$XPID"
XPID=""
wait "$PID"
PID=""
xctrace export --input "$TRACE_OUT" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="counters-profile"]' \
  > "$RUN_DIR/counters.xml"
echo "Profile saved: $RUN_DIR"
