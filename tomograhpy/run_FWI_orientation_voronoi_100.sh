#!/bin/bash
#SBATCH --job-name=fwi_orient_voronoi100
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=144
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ho21481@bristol.ac.uk
#SBATCH --signal=B:USR1@1800

# Run FWI_orientation_voronoi_100.py (100 grains, 30 x 30 mm) on 2 Isambard-AI GPUs (Salvus licence limit), one event per GPU at a time.
# Submit from this directory:
#   mkdir -p logs && sbatch run_FWI_orientation_voronoi_100.sh                          # resume (or start if no checkpoint)
#   mkdir -p logs && sbatch --export=ALL,FWI_FRESH_START=1 run_FWI_orientation_voronoi_100.sh   # new run
#
# The FWI saves a checkpoint after every L-BFGS attempt. 30 min before the 24 h limit Slurm sends
# USR1: this script then submits the next job (starting after this one ends), which resumes from
# the checkpoint. A finished run exits at once, so the chain stops by itself.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate salvus

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

nvidia-smi

# One Salvus site per GPU (isambard_oliver_gpu<g>, CUDA_VISIBLE_DEVICES = g). Initialising a site
# runs a short test simulation on its GPU, so it has to happen here, not on the login node.
for g in $(seq 0 $(( ${SLURM_GPUS_ON_NODE:-1} - 1 ))); do
    salvus-cli init-site "isambard_oliver_gpu$g"
done

# Time only the Python run; printed on exit, even if it fails
SECONDS=0
trap 'echo "Total simulation time: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m $((SECONDS % 60))s"' EXIT

# Resubmit on the time-limit warning; FWI_FRESH_START=0 so the next job never wipes the run
resubmit() {
    echo "Time limit near at $(date): submitting the next job (resumes from the checkpoint)"
    sbatch --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,FWI_FRESH_START=0 run_FWI_orientation_voronoi_100.sh
}
trap resubmit USR1

# Python in the background so the shell can handle USR1 while it runs; wait again after the
# trap (wait returns > 128 when interrupted by a signal) until Python itself exits
set +e
python FWI_orientation_voronoi_100.py &
PID=$!
wait "$PID"; STATUS=$?
while [ "$STATUS" -gt 128 ] && kill -0 "$PID" 2>/dev/null; do
    wait "$PID"; STATUS=$?
done
set -e
echo "Python exited with status $STATUS"

echo "Finished at $(date)"
