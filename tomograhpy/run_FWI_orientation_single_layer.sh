#!/bin/bash
#SBATCH --job-name=fwi_orient_sl
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ho21481@bristol.ac.uk

# Run FWI_orientation_single_layer.py on one Isambard-AI GPU.
# Submit from this directory:  mkdir -p logs && sbatch run_FWI_orientation_single_layer.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate salvus

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

nvidia-smi

# Time only the Python run; printed on exit, even if it fails
SECONDS=0
trap 'echo "Total simulation time: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m $((SECONDS % 60))s"' EXIT

python FWI_orientation_single_layer.py

echo "Finished at $(date)"
