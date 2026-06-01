#!/bin/bash
set -euo pipefail

DATASET="rel_amazon"
DESCRIPTION="tabdlm"
SAVE_DESCRIPTION="_synthetic_full"

# ---------- Training ----------
PYTHONPATH=. python main.py train \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --epochs 75 \
    --batch_size 8 \
    --batch_accum 16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --answer_len 128 \
    --loss_type no_divide_pmask \
    --bf16

# ---------- Sampling ----------
PYTHONPATH=. python main.py sample \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --save_description "$SAVE_DESCRIPTION" \
    --do_sampling \
    --bf16 \
    --gen_length 128 \
    --block_length 128 \
    --sample_step 128 \
    --temperature 1.0 \
    --sample_batch_size 16 \
    --eval_metrics mle_wtext density
