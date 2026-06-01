#!/bin/bash
set -euo pipefail

DATASET="beijing"
DESCRIPTION="tabdlm"
SAVE_DESCRIPTION="_synthetic_full"

# ---------- Training ----------
PYTHONPATH=. python main.py train \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --epochs 15 \
    --batch_size 16 \
    --batch_accum 16 \
    --lora_r 32 \
    --lora_alpha 32 \
    --answer_len 36 \
    --loss_type no_divide_pmask \
    --bf16

# ---------- Sampling ----------
PYTHONPATH=. python main.py sample \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --save_description "$SAVE_DESCRIPTION" \
    --do_sampling \
    --bf16 \
    --gen_length 36 \
    --block_length 36 \
    --sample_step 36 \
    --temperature 1.0 \
    --sample_batch_size 16 \
    --eval_metrics density c2st mle
