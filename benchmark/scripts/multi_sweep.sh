#!/bin/bash
# Run QPS-recall sweep N times per dataset, saving each run for band plotting.
# Usage: bash benchmark/scripts/multi_sweep.sh [num_runs] [max_points]

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
    echo "  Dataset: $name ($RUNS runs)"
    echo "════════════════════════════════════════"

    for run in $(seq 1 "$RUNS"); do
        echo ""
        echo "── $name run $run/$RUNS ──"
        cargo run --release --bin benchmark -- \
            --base "$base" --query "$query" --groundtruth "$gt" \
            --algorithms qps-recall-sweep --max-points "$MAX_POINTS" 2>&1 | \
            grep -E "^  L=|Calibrated|═══|Config"

        # Copy the generated JSON to runs directory
        cp "visualizations/qps_recall_${name}.json" \
           "visualizations/runs/qps_recall_${name}_run${run}.json"
    done
done

echo ""
echo "All runs complete. Results in visualizations/runs/"
echo "Run: python3 visualizations/plot_qps_recall_bands.py"
