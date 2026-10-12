# Release scripts

These are the entry points for the compact ACRE release. They use `configs/paths.env`.

```bash
bash scripts/train_sft.sh
export SFT_ADAPTER=/path/to/sft/checkpoint
export GRPO_DATASET=/path/to/train_grpo.jsonl
bash scripts/train_grpo.sh
bash scripts/evaluate.sh /path/to/checkpoint test
```

Before training, install MS-Swift, apply the Qwen2-Audio patch, activate the Python environment, and make sure the audio paths in the JSONL files are available.

The evaluation command writes inference JSONL and a metrics JSON file using `acre/evaluate.py`.
