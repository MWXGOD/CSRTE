#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/configs/paths.env"
CKPT="${1:?usage: $0 CHECKPOINT [dev|test]}"
SPLIT="${2:-test}"
DATASET="${EVAL_DATASET:-${DATA_ROOT}/${SPLIT}.jsonl}"
RESULT="${OUTPUT_ROOT}/eval/${SPLIT}_$(basename "${CKPT}").jsonl"
swift infer --model "${BASE_MODEL}" --adapters "${CKPT}" --val_dataset "${DATASET}" \
  --result_path "${RESULT}" --metric triple_f1 --infer_backend transformers \
  --torch_dtype bfloat16 --max_batch_size 1 --max_new_tokens 128 \
  --temperature 0.0 --top_k 1 --stop_words "<|endoftext|>"
python "${ROOT}/acre/evaluate.py" --result_path "${RESULT}" --label_path "${DATASET}"
