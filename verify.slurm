#!/bin/bash
#SBATCH --job-name=taan_verify
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@nyu.edu

# ============================================================
# verify.slurm
# Run TAAN signal-quality verification on NYU HPC
# ============================================================

set -euo pipefail

echo "===== Job info ====="
echo "SLURM_JOB_ID      : $SLURM_JOB_ID"
echo "SLURM_NODELIST    : $SLURM_NODELIST"
echo "Start time        : $(date)"
echo ""

# ---- Environment -----------------------------------------------
module purge
module load python/3.11
module load cuda/12.1

# Activate virtual environment (adjust path as needed)
source ~/envs/taan/bin/activate

# Navigate to project root
PROJ_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJ_DIR}"

echo "Project directory : ${PROJ_DIR}"
echo "Python            : $(which python)"
echo "PyTorch           : $(python -c 'import torch; print(torch.__version__)')"
echo ""

# ---- Run verification ------------------------------------------
mkdir -p logs

python verify_taan.py \
    --seed 42 \
    --n-samples 150 \
    --batch-repeat 10

echo ""
echo "===== Verification complete ====="
echo "End time : $(date)"
