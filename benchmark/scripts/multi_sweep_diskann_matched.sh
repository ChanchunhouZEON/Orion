#!/bin/bash
# Re-run qps-recall-sweep 3x per dataset and merge ONLY `diskann_matched`
# into visualizations/runs/qps_recall_{name}_run{i}.json, preserving existing
# `diskann` (alpha=2.0) and `staged` bands.
#
# Usage: bash benchmark/scripts/multi_sweep_diskann_matched.sh [num_runs] [max_points]

set -e

RUNS="${1:-3}"
MAX_POINTS="${2:-100000}"

DATASETS=(
    "sift:data/sift/sift_base.fvecs:data/sift/sift_query.fvecs:data/sift/sift_groundtruth.ivecs"
    "glove25:data/glove25/glove-25-angular_base.fvecs:data/glove25/glove-25-angular_query.fvecs:data/glove25/glove-25-angular_groundtruth.ivecs"
    "glove100:data/glove100/glove-100-angular_base.fvecs:data/glove100/glove-100-angular_query.fvecs:data/glove100/glove-100-angular_groundtruth.ivecs"
    "gist:data/gist/gist_base.fvecs:data/gist/gist_query.fvecs:data/gist/gist_groundtruth.ivecs"
)

cargo build --release --bin benchmark 2>&1 | tail -1
mkdir -p visualizations/runs

for ds_spec in "${DATASETS[@]}"; do
    IFS=: read -r name base query gt <<< "$ds_spec"
    echo ""
    echo "════════════════════════════════════════"
    echo "  Dataset: $name ($RUNS runs, extracting diskann_matched)"
    echo "════════════════════════════════════════"

    for run in $(seq 1 "$RUNS"); do
        echo ""
        echo "── $name run $run/$RUNS ──"
        cargo run --release --bin benchmark -- \
            --base "$base" --query "$query" --groundtruth "$gt" \
            --algorithms qps-recall-sweep --max-points "$MAX_POINTS" 2>&1 | \
            grep -E "^  L=|Calibrated|═══|Config" || true

        python3 - "$name" "$run" <<'PY'
import json, sys
name, run = sys.argv[1], sys.argv[2]
src = f"visualizations/qps_recall_{name}.json"
dst = f"visualizations/runs/qps_recall_{name}_run{run}.json"
with open(src) as f:
    new = json.load(f)
with open(dst) as f:
    old = json.load(f)
old["diskann_matched"] = new["diskann_matched"]
old["staged_alpha"] = new.get("staged_alpha", old.get("staged_alpha"))
with open(dst, "w") as f:
    json.dump(old, f)
print(f"  merged diskann_matched -> {dst}")
PY
    done
done

echo ""
echo "All diskann_matched runs merged into visualizations/runs/"
echo "Run: python3 visualizations/plot_qps_recall_bands.py"
