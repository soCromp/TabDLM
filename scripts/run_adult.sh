#!/bin/bash
set -euo pipefail

DATASET="adult"
DESCRIPTION="tabdlm"
SAVE_DESCRIPTION="_synthetic_full"

# ---------- Training ----------
PYTHONPATH=. python main.py train \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --epochs 20 \
    --batch_size 4 \
    --batch_accum 32 \
    --lora_r 16 \
    --lora_alpha 32 \
    --answer_len 64 \
    --loss_type no_divide_pmask \
    --bf16

# ---------- Sampling ----------
PYTHONPATH=. python main.py sample \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --save_description "$SAVE_DESCRIPTION" \
    --do_sampling \
    --bf16 \
    --gen_length 64 \
    --block_length 64 \
    --sample_step 64 \
    --temperature 1.0 \
    --sample_batch_size 16 \
    --eval_metrics density c2st mle
