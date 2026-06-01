#!/bin/bash
set -euo pipefail

DATASET="math_latex"
DESCRIPTION="tabdlm"
SAVE_DESCRIPTION="_synthetic_full"

# ---------- Training ----------
PYTHONPATH=. python main.py train \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --epochs 225 \
    --batch_size 16 \
    --batch_accum 16 \
    --warmup_ratio 0.03333333333333333 \
    --lora_r 32 \
    --lora_alpha 32 \
    --answer_len 36 \
    --loss_type dream_loss \
    --bf16

# ---------- Sampling ----------
PYTHONPATH=. python main.py sample \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --save_description "$SAVE_DESCRIPTION" \
    --do_sampling \
    --bf16 \
    --decoupled_steps \
    --gen_length 36 \
    --block_length 6 \
    --sample_step 64 \
    --text_steps 36 \
    --num_steps 50 \
    --temperature 0.8 \
    --sample_batch_size 32 \
    --eval_metrics match_score density c2st
