#!/bin/bash
set -euo pipefail

DATASET="rel_arxiv"
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
    --answer_len 160 \
    --loss_type no_divide_pmask \
    --bf16

# ---------- Sampling ----------
PYTHONPATH=. python main.py sample \
    --dataset_name "$DATASET" \
    --description "$DESCRIPTION" \
    --save_description "$SAVE_DESCRIPTION" \
    --do_sampling \
    --bf16 \
    --use_best_ckp \
    --gen_length 160 \
    --block_length 160 \
    --sample_step 160 \
    --temperature 1.0 \
    --sample_batch_size 8 \
    --seed 19 \
    --eval_metrics mle_wtext density
