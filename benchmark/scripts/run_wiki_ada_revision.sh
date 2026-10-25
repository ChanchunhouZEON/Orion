#!/usr/bin/env bash
# Full-size normalized ada-002 extension. No embedding API calls are made.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup_python_env numpy faiss
export PY
export DATASET=wiki_ada_1M
export NUM_RUNS="${NUM_RUNS:-3}"
export COOLDOWN_S="${COOLDOWN_S:-1}"
export MAX_POINTS=0
OUT_DIR="${OUT_DIR:-$ROOT/visualizations/wiki_ada_revision/$(date +%Y%m%d_%H%M%S)}"
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$ROOT/$OUT_DIR"
[[ ! -e "$OUT_DIR" ]] || { echo "Use a new OUT_DIR: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
git rev-parse HEAD > "$OUT_DIR/revision.txt"
git status --short > "$OUT_DIR/worktree-status.txt"
uname -a > "$OUT_DIR/host.txt"

if [[ ! -f data/wiki_ada_1M/manifest.json ]]; then
  "$PY" -u -B data/load_wiki_ada_1M.py --gt-backend faiss --threads 8 --num-shards 4 \
    2>&1 | tee "$OUT_DIR/preparation.log"
fi
"$PY" -c '
import json, pathlib
p = pathlib.Path("data/wiki_ada_1M")
m = json.loads((p / "manifest.json").read_text())
assert (m["n_base"], m["n_query"], m["dimension"]) == (1000000, 10000, 1536)
assert m["metric"] == "normalized-inner-product" and m["top_k"] >= 100
assert m["gt_backend"] == "faiss"
for kind, rows, dim in [("base", 1000000, 1536), ("query", 10000, 1536), ("groundtruth", 10000, m["top_k"])]:
    ext = "ivecs" if kind == "groundtruth" else "fvecs"
    assert (p / f"wiki_ada_1M_{kind}.{ext}").stat().st_size == rows * (dim + 1) * 4
print("Validated full-size normalized-IP dataset and exact ground truth")
' | tee "$OUT_DIR/input-validation.log"

# This is the graph preparation stage; do not run it beside query timings.
bash "$SCRIPT_DIR/prepare_parlayann_data.sh" 2>&1 | tee "$OUT_DIR/graph-preparation.log"
OUT_JSON="$OUT_DIR/three_engine.json" \
  bash "$SCRIPT_DIR/sweep_orion_vs_diskann_vs_parlayann.sh" \
  2>&1 | tee "$OUT_DIR/three-engine.log"

# The control changes only the optional prefilter on the same exported graph.
BIN="$(binary_path orion-sweep)"
"$BIN" "$DATASET" --k 10 --threads 8 --prefilter none --print-config \
  > "$OUT_DIR/prefilter-off.config.log" 2>&1
for ((i=1; i<=NUM_RUNS; i++)); do
  sleep "$COOLDOWN_S"
  "$BIN" "$DATASET" --k 10 --threads 8 --trials 1 --prefilter none \
    > "$OUT_DIR/prefilter_off_$i.log" 2>&1
done
"$PY" - "$OUT_DIR" "$NUM_RUNS" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmark/scripts")
from collect_sweep_medians import parse_orion, median_by_slot
out = Path(sys.argv[1])
runs = [parse_orion(out / f"prefilter_off_{i}.log") for i in range(1, int(sys.argv[2]) + 1)]
result = {"dataset": "wiki_ada_1M", "k": 10, "threads": 8,
          "num_queries": 10000, "prefilter": "none", "runs": runs,
          "points": median_by_slot(runs)}
(out / "prefilter_off.json").write_text(json.dumps(result, indent=2) + "\n")
PY
sleep "$COOLDOWN_S"
"$PY" -u -B "$SCRIPT_DIR/dbms_baselines.py" --dataset "$DATASET" \
  --engines faiss --max-points 0 --threads 8 --trials "$NUM_RUNS" --k 10 \
  --output "$OUT_DIR/faiss_hnsw_sq.json" 2>&1 | tee "$OUT_DIR/faiss.log"
echo "Completed: $OUT_DIR"
