#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/configs/paths.env"
: "${BASE_MODEL:?set BASE_MODEL}"
: "${DATA_ROOT:?set DATA_ROOT}"
swift sft \
  --model "${BASE_MODEL}" \
  --output_dir "${OUTPUT_ROOT}/sft" \
  --dataset "${SFT_DATASET:-${DATA_ROOT}/train.jsonl}" \
  --val_dataset "${SFT_VAL_DATASET:-${DATA_ROOT}/dev.jsonl}" \
  --tuner_type lora --torch_dtype bfloat16 \
  --num_train_epochs 10 --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 --learning_rate 2e-4 \
  --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --gradient_accumulation_steps 16 --eval_strategy steps --eval_steps 1000 \
  --save_strategy steps --save_steps 1000 --predict_with_generate true \
  --max_new_tokens 512 --max_length 2048 --eval_metric triple_f1 \
  --metric_for_best_model f1 --greater_is_better true --load_best_model_at_end true
