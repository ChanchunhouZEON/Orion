#!/bin/bash
set -e

DATASET="${1:?Usage: $0 <dataset> [max_points]}"
MAX_POINTS="${2:-100000}"
TRACE_OUT="/tmp/${DATASET}_counters.trace"
LOG="/tmp/cache_profile_${DATASET}.log"

rm -rf "$TRACE_OUT" "$LOG"

echo "=== Profiling $DATASET ($MAX_POINTS points) ==="

# Start profiler in background
./target/release/cache_profile "$DATASET" "$MAX_POINTS" 2>"$LOG" &
PID=$!

# Wait for marker file (build complete)
echo "Waiting for build (PID=$PID)..."
MARKER="/tmp/cache_profile_${PID}.ready"
while [ ! -f "$MARKER" ]; do sleep 0.3; done
echo "Build done. Attaching xctrace CPU Counters..."

# Attach xctrace — it will auto-stop when target exits
xctrace record --template "CPU Counters" --attach "$PID" --time-limit 30s --output "$TRACE_OUT" &
XPID=$!

# Wait for profiler to finish
wait $PID 2>/dev/null || true
wait $XPID 2>/dev/null || true

echo ""
echo "=== Profile Results ==="
cat "$LOG"

echo ""
echo "=== Exporting xctrace counters ==="
xctrace export --input "$TRACE_OUT" --xpath '/trace-toc/run[@number="1"]/data/table[@schema="counters-profile"]' \
  2>/dev/null | head -300 || echo "(xpath export failed, trying raw...)"

echo ""
echo "Trace saved: $TRACE_OUT"
