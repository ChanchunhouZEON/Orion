#!/usr/bin/env bash
# EE-disabled neighbor-policy analysis: benchmark -> JSON/logs -> PNG/PDF.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash benchmark/scripts/run_adaptive.sh [--dry-run | --plot-only]

Environment (paths are relative to the Orion repository root):
  DATASETS             Space-separated YAML shortcuts (default: "sift gist")
  K_VALUES             Space-separated k values (default: "10 100")
  THREADS              Query threads (default: 8)
  TRIALS               Timing trials per arm (default: 5)
  DIAGNOSTIC_QUERIES   Diagnostic sample count; 0 = all (default: 1000)
  GRAPH_SOURCE         parlayann or rust (default: parlayann)
  CONFIG               Optional YAML file; otherwise use normal config resolution
  SEARCH_LIST_SIZES    Optional comma-separated L override; otherwise per-k YAML
  RECALL_TARGETS       Comma-separated targets (default: 0.9,0.95,0.99)
  COOLDOWN_S           Idle before each benchmark (default: 30)
  OUT_DIR              Result directory (default: timestamped visualizations run)
  DETAIL_L             Optional detail-panel L; otherwise largest measured L
  PY                   Python interpreter (uses scripts/_env.sh)
  ADAPTIVE_BIN         Existing executable; skips compilation when supplied
  BUILD_DIR            Cargo target dir (default: CARGO_TARGET_DIR or target)

--dry-run prints commands without compiling, running, or writing files.
--plot-only requires OUT_DIR and renders existing JSONs for DATASETS x K_VALUES.
ParlayANN uses cache first; a cache miss needs a matching staged export.
A single SEARCH_LIST_SIZES override must be valid for every requested k.
Existing JSONs are never overwritten; use another OUT_DIR for a new run.
EOF
}

DRY_RUN=0
PLOT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --plot-only) PLOT_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [[ "$DRY_RUN" == 1 && "$PLOT_ONLY" == 1 ]]; then
  echo "--dry-run and --plot-only are mutually exclusive" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
DATASETS="${DATASETS:-sift gist}"
K_VALUES="${K_VALUES:-10 100}"
THREADS="${THREADS:-8}"
TRIALS="${TRIALS:-5}"
DIAGNOSTIC_QUERIES="${DIAGNOSTIC_QUERIES:-0}"
GRAPH_SOURCE="${GRAPH_SOURCE:-parlayann}"
RECALL_TARGETS="${RECALL_TARGETS:-0.9,0.95,0.99}"
COOLDOWN_S="${COOLDOWN_S:-1}"

if [[ "$PLOT_ONLY" == 1 && -z "${OUT_DIR:-}" ]]; then
  echo "--plot-only requires OUT_DIR" >&2; exit 2
fi

OUT_DIR="${OUT_DIR:-visualizations/adaptive_runs/$(date +%Y%m%d_%H%M%S)_$$}"
BUILD_DIR="${BUILD_DIR:-${CARGO_TARGET_DIR:-target}}"
BUILD=1

if [[ -n "${ADAPTIVE_BIN:-}" ]]; then
  BUILD=0
else
  ADAPTIVE_BIN="$BUILD_DIR/release/adaptive"
fi
[[ "$ADAPTIVE_BIN" = /* ]] || ADAPTIVE_BIN="$ROOT/$ADAPTIVE_BIN"
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$ROOT/$OUT_DIR"

read -r -a datasets <<< "$DATASETS"

read -r -a ks <<< "$K_VALUES"
if [[ ${#datasets[@]} == 0 || ${#ks[@]} == 0 ]]; then
  echo "DATASETS and K_VALUES must be nonempty" >&2; exit 2
fi

for ds in "${datasets[@]}"; do
  [[ "$ds" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid dataset shortcut: $ds" >&2; exit 2; }
done

for value in "$THREADS" "$TRIALS" "${ks[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive integer: $value" >&2; exit 2; }
done

for value in "$DIAGNOSTIC_QUERIES" "$COOLDOWN_S"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected nonnegative integer: $value" >&2; exit 2; }
done

case "$GRAPH_SOURCE" in
  parlayann|rust) ;;
  *) echo "GRAPH_SOURCE must be parlayann or rust" >&2; exit 2 ;;
esac

show() { printf '  '; printf '%q ' "$@"; printf '\n'; }
common=(--graph-source "$GRAPH_SOURCE" --threads "$THREADS" --trials "$TRIALS"
        --diagnostic-queries "$DIAGNOSTIC_QUERIES" --recall-targets "$RECALL_TARGETS")
[[ -z "${CONFIG:-}" ]] || common+=(--config "$CONFIG")
[[ -z "${SEARCH_LIST_SIZES:-}" ]] || common+=(--search-list-sizes "$SEARCH_LIST_SIZES")

if [[ "$DRY_RUN" == 0 ]]; then
  source "$SCRIPT_DIR/_env.sh"
  setup_python_env matplotlib numpy
else
  PY="${PY:-python3}"
fi

if [[ "$PLOT_ONLY" == 0 ]]; then
  # Check all destinations before starting any potentially long run.
  for ds in "${datasets[@]}"; do
    for k in "${ks[@]}"; do
      file="$OUT_DIR/adaptive_${ds}_k${k}.json"
      [[ ! -e "$file" ]] || { echo "Refusing to overwrite $file" >&2; exit 2; }
    done
  done
  if [[ "$BUILD" == 1 ]]; then
    show cargo build --release -p benchmark --bin adaptive --target-dir "$BUILD_DIR"
    if [[ "$DRY_RUN" == 0 ]]; then
      cargo build --release -p benchmark --bin adaptive --target-dir "$BUILD_DIR"
    fi
  fi
  if [[ "$DRY_RUN" == 0 ]]; then
    [[ -x "$ADAPTIVE_BIN" ]] || { echo "Not executable: $ADAPTIVE_BIN" >&2; exit 2; }
    mkdir -p "$OUT_DIR"
  fi
  # Resolve every configuration before running the first dataset.
  for ds in "${datasets[@]}"; do
    for k in "${ks[@]}"; do
      cmd=("$ADAPTIVE_BIN" "$ds" --k "$k" "${common[@]}" --print-config)
      show "${cmd[@]}"
      if [[ "$DRY_RUN" == 0 ]]; then
        "${cmd[@]}" > "$OUT_DIR/adaptive_${ds}_k${k}.config.json"
      fi
    done
  done
  for ds in "${datasets[@]}"; do
    for k in "${ks[@]}"; do
      stem="$OUT_DIR/adaptive_${ds}_k${k}"
      echo "Adaptive search: dataset=$ds k=$k"
      cmd=("$ADAPTIVE_BIN" "$ds" --k "$k" "${common[@]}" --output "$stem.json")
      show "${cmd[@]}"
      if [[ "$DRY_RUN" == 0 ]]; then
        sleep "$COOLDOWN_S"
        "${cmd[@]}" 2>&1 | tee "$stem.log"
        [[ -s "$stem.json" ]] || { echo "Missing result: $stem.json" >&2; exit 1; }
      fi
    done
  done
fi

# Plot only after all timings, so rendering cannot perturb later runs.
for ds in "${datasets[@]}"; do
  for k in "${ks[@]}"; do
    file="$OUT_DIR/adaptive_${ds}_k${k}.json"
    cmd=("$PY" -B "$ROOT/visualizations/plot_adaptive.py" "$file")
    [[ -z "${DETAIL_L:-}" ]] || cmd+=(--detail-l "$DETAIL_L")
    show "${cmd[@]}"
    if [[ "$DRY_RUN" == 0 ]]; then
      [[ -s "$file" ]] || { echo "Missing result: $file" >&2; exit 1; }
      "${cmd[@]}" 2>&1 | tee "$OUT_DIR/adaptive_${ds}_k${k}.plot.log"
    fi
  done
done
echo "Output directory: $OUT_DIR"
