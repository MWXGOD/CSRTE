# Configuration

Copy `paths.env.example` to `paths.env` and edit the local paths before using the release scripts.

```bash
cp configs/paths.env.example configs/paths.env
source configs/paths.env
```

The main variables are:

| Variable | Meaning |
| --- | --- |
| `BASE_MODEL` | Local model path or model hub identifier for Qwen2-Audio-7B |
| `DATA_ROOT` | Directory containing `train.jsonl`, `dev.jsonl`, `test.jsonl`, and audio |
| `OUTPUT_ROOT` | Directory for SFT, GRPO, and evaluation outputs |
| `SFT_ADAPTER` | SFT LoRA checkpoint used as the GRPO starting point |
| `GRPO_DATASET` | GRPO JSONL with a top-level `solution` field |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

Optional dataset overrides:

```bash
export SFT_DATASET="${DATA_ROOT}/train.jsonl"
export SFT_VAL_DATASET="${DATA_ROOT}/dev.jsonl"
export GRPO_DATASET="/path/to/train_grpo.jsonl"
export EVAL_DATASET="${DATA_ROOT}/test.jsonl"
```

The release scripts use these variables and do not require absolute paths from the original experiment.
