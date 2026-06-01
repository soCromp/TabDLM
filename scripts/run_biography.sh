#!/bin/bash
set -euo pipefail

DATASET="biography"
DESCRIPTION="tabdlm"
SAVE_DESCRIPTION="_synthetic_full"

# ---------- Training ----------
PYTHONPATH=. python main.py train \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --epochs 75 \
    --batch_size 8 \
    --batch_accum 16 \
    --lora_r 32 \
    --lora_alpha 32 \
    --answer_len 80 \
    --loss_type dream_loss \
    --bf16

# ---------- Sampling ----------
PYTHONPATH=. python main.py sample \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --save_description "$SAVE_DESCRIPTION" \
    --do_sampling \
    --bf16 \
    --gen_length 80 \
    --block_length 16 \
    --sample_step 80 \
    --temperature 0.8 \
    --sample_batch_size 16 \
    --seed 110 \
    --eval_metrics bio_match_score density
