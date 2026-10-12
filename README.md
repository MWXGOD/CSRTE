# ACRE

## An Acoustic-Cue and Contextual-Knowledge Reasoning Enhancement Framework for Speech Relation Extraction

This repository contains the compact release of **ACRE**, the CSRTE speech relation extraction system. ACRE fine-tunes Qwen2-Audio with LoRA SFT and GRPO, and uses pronunciation, pause, entity, relation, and triplet signals during optimization.

The repository contains the project-specific code, the Qwen2-Audio MS-Swift patch, the CSRTE JSONL data splits, and runnable training/evaluation scripts. The MS-Swift framework and audio files are not included.

## Overview

Given speech and an extraction instruction, the model generates relation triples in the form:

```text
head##relation##tail$$head##relation##tail
```

The ACRE reasoning target can additionally contain an acoustic reasoning block:

```text
<think>{"pronunciation": "...", "pause": "..."}</think>
<answer>head##relation##tail</answer>
```

The complete workflow is:

```text
CSRTE data -> LoRA SFT -> SFT checkpoint -> GRPO -> evaluation
```

## Repository Structure

```text
release/
├── acre/
│   ├── reward.py                 # ACRE GRPO reward
│   ├── evaluate.py               # Strict and PAE evaluation
│   └── __init__.py
├── configs/
│   ├── paths.env.example         # Local path configuration
│   └── README.md
├── data/
│   ├── train.jsonl               # Training split
│   ├── dev.jsonl                 # Development split
│   ├── test.jsonl                # Test split
│   └── audio/                    # Audio files should be placed here later
├── patches/
│   └── ms-swift-qwen2-audio.patch
├── requirements/
│   └── environment.txt
├── scripts/
│   ├── train_sft.sh
│   ├── train_grpo.sh
│   └── evaluate.sh
├── LICENSE
└── README.md
```

## Data

The included JSONL files are the CSRTE no-relations splits:

| File | Samples | Use |
| --- | ---: | --- |
| `data/train.jsonl` | 15,000 | SFT training |
| `data/dev.jsonl` | 2,000 | Development/evaluation |
| `data/test.jsonl` | 2,855 | Final evaluation |

Each row uses the MS-Swift multimodal conversation format:

```json
{
  "messages": [
    {"role": "user", "content": "<audio> ..."},
    {"role": "assistant", "content": "head##relation##tail<|endoftext|>"}
  ],
  "audios": ["/path/to/audio/example.wav"]
}
```

The `audio/` directory is intentionally empty in this release. Update the `audios` paths in the JSONL files, or place the corresponding audio files at the referenced paths before training.

The three included files are ordinary conversation/SFT data. GRPO requires a separate JSONL version with a top-level `solution` field; pass that file through `GRPO_DATASET` when running `scripts/train_grpo.sh`.

## Environment

Create an environment and install MS-Swift separately:

```bash
conda create -n acre python=3.11
conda activate acre
pip install -r requirements/environment.txt
```

The project was validated with MS-Swift `4.2.0.dev0` and the revision recorded in `requirements/ms-swift-version.txt`. This repository does not redistribute MS-Swift.

Apply the Qwen2-Audio compatibility patch:

```bash
git -C /path/to/ms-swift apply /path/to/release/patches/ms-swift-qwen2-audio.patch
```

Configure paths:

```bash
cp configs/paths.env.example configs/paths.env
${EDITOR:-vi} configs/paths.env
source configs/paths.env
```

At minimum, set `BASE_MODEL`, `DATA_ROOT`, and `OUTPUT_ROOT`. For this release, `DATA_ROOT` can point to `release/data`.

## Training

### SFT

Run LoRA supervised fine-tuning with the included train and development files:

```bash
bash scripts/train_sft.sh
```

The default paths are `${DATA_ROOT}/train.jsonl` and `${DATA_ROOT}/dev.jsonl`. The resulting adapter is saved under `${OUTPUT_ROOT}/sft`.

### GRPO

GRPO starts from an SFT adapter and needs data containing `solution` references:

```bash
export SFT_ADAPTER=/path/to/sft/checkpoint
export GRPO_DATASET=/path/to/train_grpo.jsonl
bash scripts/train_grpo.sh
```

The release reward is registered as `speechre_acre_reward`. Dynamic sampling is disabled in the provided command.

## Evaluation

Evaluate a saved adapter or checkpoint on the included test split:

```bash
bash scripts/evaluate.sh /path/to/checkpoint test
```

For the development split:

```bash
bash scripts/evaluate.sh /path/to/checkpoint dev
```

The evaluator reports strict entity, relation, and triplet precision/recall/F1, together with pronunciation-aware entity matching (PAE). Results are written below `${OUTPUT_ROOT}/eval`.

## Main Results

The final ACRE-Qwen2-Audio-7B results reported in the paper are:

| Dataset | Entity F1 | Relation F1 | Triplet F1 |
| --- | ---: | ---: | ---: |
| CoNLL04 | 58.09 | 87.14 | 42.07 |
| Common Voice | 53.66 | 69.58 | 30.41 |
| CSRTE | 53.96 | 78.25 | 34.90 |

On CSRTE, PAE improves entity F1 to 62.45 and triplet F1 to 42.26.

## Citation

If you use this code or the ACRE method, please cite the accompanying paper:

```text
ACRE: An Acoustic-Cue and Contextual-Knowledge Reasoning Enhancement
Framework for Speech Relation Extraction.
```
