# ACRE

## An Acoustic-Cue and Contextual-Knowledge Reasoning Enhancement Framework for Speech Relation Extraction

ACRE fine-tunes Qwen2-Audio with LoRA SFT and GRPO, and uses pronunciation, pause, entity, relation, and triplet signals during optimization.

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
│   └── audio/                    # Audio files
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
