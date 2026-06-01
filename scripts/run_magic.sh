#!/bin/bash
set -euo pipefail

DATASET="magic"
DESCRIPTION="tabdlm"
SAVE_DESCRIPTION="_synthetic_full"

# ---------- Training ----------
PYTHONPATH=. python main.py train \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --epochs 30 \
    --batch_size 16 \
    --batch_accum 32 \
    --lora_r 16 \
    --lora_alpha 32 \
    --answer_len 32 \
    --loss_type no_divide_pmask \
    --all_numerical \
    --bf16

# ---------- Sampling ----------
PYTHONPATH=. python main.py sample \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --save_description "$SAVE_DESCRIPTION" \
    --do_sampling \
    --bf16 \
    --all_numerical \
    --gen_length 64 \
    --block_length 64 \
    --sample_step 64 \
    --temperature 1.0 \
    --sample_batch_size 32 \
    --eval_metrics density c2st mle
