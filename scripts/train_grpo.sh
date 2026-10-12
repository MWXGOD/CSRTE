#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/configs/paths.env"
: "${BASE_MODEL:?set BASE_MODEL}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${SFT_ADAPTER:?set SFT_ADAPTER to an SFT LoRA checkpoint}"
: "${GRPO_DATASET:?set GRPO_DATASET to a JSONL containing solution references}"
swift rlhf --rlhf_type grpo \
  --model "${BASE_MODEL}" --adapters "${SFT_ADAPTER}" --ref_adapters "${SFT_ADAPTER}" \
  --output_dir "${OUTPUT_ROOT}/grpo" --external_plugins "${ROOT}/acre/reward.py" \
  --dataset "${GRPO_DATASET}" \
  --reward_funcs speechre_acre_reward --tuner_type lora --torch_dtype bfloat16 \
  --num_train_epochs 5 --per_device_train_batch_size 4 --num_generations 4 \
  --gradient_accumulation_steps 16 --dynamic_sample false --learning_rate 5e-6 \
  --temperature 1 --top_p 0.95 --max_new_tokens 512 --max_length 2048 --beta 0.04 \
  --eval_strategy no --save_strategy steps --save_steps 500 --use_vllm true \
  --vllm_mode colocate --vllm_gpu_memory_utilization 0.6 --vllm_max_model_len 2304
