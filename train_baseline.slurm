#!/bin/bash
#SBATCH --job-name=taan_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@nyu.edu

# ============================================================
# train_baseline.slurm
# Train Qwen3-4B with standard GRPO (global normalization)
# on GSM8K + MATH (seen types only split).
# ============================================================

set -euo pipefail

echo "===== Job info ====="
echo "SLURM_JOB_ID      : $SLURM_JOB_ID"
echo "SLURM_NODELIST    : $SLURM_NODELIST"
echo "GPUs              : $SLURM_GPUS_ON_NODE"
echo "Start time        : $(date)"
echo ""

# ---- Environment -----------------------------------------------
module purge
module load python/3.11
module load cuda/12.1
module load nccl/2.18

source ~/envs/taan/bin/activate

PROJ_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJ_DIR}"

echo "Project directory : ${PROJ_DIR}"
echo "Python            : $(which python)"
echo "PyTorch           : $(python -c 'import torch; print(torch.__version__)')"
echo ""

# ---- Paths & hyperparameters -----------------------------------
MODEL_PATH="${HOME}/models/Qwen3-4B"
DATA_DIR="${PROJ_DIR}/data"
OUTPUT_DIR="${PROJ_DIR}/checkpoints/baseline_grpo_${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}" logs

# ---- Distributed setup -----------------------------------------
export MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE="${SLURM_NTASKS}"

# ---- Launch training -------------------------------------------
# Using OpenRLHF's GRPO trainer (unmodified — global normalization)
torchrun \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m openrlhf.cli.train_grpo \
        --pretrain "${MODEL_PATH}" \
        --reward_pretrain "${MODEL_PATH}" \
        --save_path "${OUTPUT_DIR}" \
        --dataset "${DATA_DIR}/gsm8k_math_seen.jsonl" \
        --input_key "problem" \
        --label_key "solution" \
        --train_batch_size 128 \
        --micro_train_batch_size 4 \
        --max_samples 50000 \
        --max_epochs 2 \
        --num_episodes 1 \
        --rollout_batch_size 128 \
        --n_samples_per_prompt 8 \
        --max_len 2048 \
        --generate_max_len 1024 \
        --advantage_estimator grpo \
        --actor_learning_rate 5e-7 \
        --critic_learning_rate 9e-6 \
        --init_kl_coef 0.01 \
        --kl_target 6 \
        --save_steps 200 \
        --logging_steps 10 \
        --eval_steps 200 \
        --bf16 \
        --flash_attn \
        --gradient_checkpointing \
        --use_wandb true \
        --wandb_project "taan-research" \
        --wandb_run_name "baseline_grpo_${SLURM_JOB_ID}"

echo ""
echo "===== Baseline training complete ====="
echo "Checkpoint : ${OUTPUT_DIR}"
echo "End time   : $(date)"
