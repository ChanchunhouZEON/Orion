#!/usr/bin/env bash
#SBATCH --job-name=orion-storage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
root="$1"
shift
cd "$root"
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# The step, not the login shell, must inspect the enforcing memory cgroup.
if command -v yhrun >/dev/null 2>&1; then
    exec yhrun -n 1 -c 8 python3 "$root/experiment.py" "$@" --root "$root"
else
    exec srun --ntasks=1 --cpus-per-task=8 python3 "$root/experiment.py" "$@" --root "$root"
fi
