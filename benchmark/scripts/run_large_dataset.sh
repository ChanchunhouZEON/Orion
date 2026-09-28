#!/usr/bin/env bash
# Example:
# BIN=./target/release/orion RUN_DIR=/data/runs/sift1b-001 \
#   bash benchmark/scripts/run_large_dataset.sh sift1b \
#   --staged-file /data/graph.staged --cache-dir /data/cache --memory-budget-gib 1800
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to a new directory for logs}"
BIN="${BIN:-./target/release/orion}"
THREADS="${THREADS:-32}"
TRIALS="${TRIALS:-3}"
[[ -x "$BIN" ]] || { echo "Build the orion executable first: $BIN" >&2; exit 2; }
# Stage flags are controlled by this script; reject conflicting passthrough flags.
for arg in "$@"; do
  case "$arg" in
    --preflight|--prepare-only|--print-config|--k|--k=*|--threads|--threads=*|--trials|--trials=*)
      echo "Do not pass stage/k/thread/trial flags; use THREADS and TRIALS variables" >&2; exit 2 ;;
  esac
done
mkdir "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/revision.txt"
git status --short > "$RUN_DIR/worktree-status.txt"
uname -a > "$RUN_DIR/host.txt"
if command -v sha256sum >/dev/null; then
  sha256sum "$BIN" > "$RUN_DIR/executable.sha256"
else
  shasum -a 256 "$BIN" > "$RUN_DIR/executable.sha256"
fi
printf '%q ' "$BIN" "$@" > "$RUN_DIR/command.txt"
printf '\nTHREADS=%q TRIALS=%q\n' "$THREADS" "$TRIALS" >> "$RUN_DIR/command.txt"
case "$(uname -s)" in
  Linux) time_flags=(-v) ;;
  Darwin) time_flags=(-l) ;;
  *) echo "Unsupported timing platform" >&2; exit 2 ;;
esac
"$BIN" "$@" --preflight > "$RUN_DIR/preflight.json" 2> "$RUN_DIR/preflight.log"
/usr/bin/time "${time_flags[@]}" "$BIN" "$@" --prepare-only \
  > "$RUN_DIR/prepare.stdout" 2> "$RUN_DIR/prepare.log"
for k in 10 100; do
  /usr/bin/time "${time_flags[@]}" "$BIN" "$@" --k "$k" --threads "$THREADS" --trials "$TRIALS" \
    > "$RUN_DIR/k${k}.stdout" 2> "$RUN_DIR/k${k}.log"
done
