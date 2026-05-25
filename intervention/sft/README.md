# SFT (Supervised Fine-Tuning) Training

This directory contains code and data for supervised fine-tuning.

## Files

- `prepare_sft_data.py`: Script to prepare SFT training data
- `train_sft_trl_tools.py`: Main training script using TRL library
- `run_train_sft_trl_tools_4gpu.sh`: 4-GPU training launch script
- `data/sft_train.jsonl`: SFT training data (1,994 samples)

## Training Data Format

Training data is in JSONL format, with each line containing one training sample. See `data/sft_train.jsonl` for the specific format.

## Usage

### 1. Prepare Training Data

```bash
python prepare_sft_data.py
```

### 2. Start Training

```bash
bash run_train_sft_trl_tools_4gpu.sh
```

## Training Configuration

The training script uses the TRL (Transformer Reinforcement Learning) library and supports:
- LoRA fine-tuning
- Multi-GPU training
- Tool call formatting
- Custom data loading

See `train_sft_trl_tools.py` and the launch script for specific configuration parameters.

## Dependencies

Main dependencies include:
- transformers
- trl
- peft (for LoRA)
- torch

See `requirements.txt` in the project root for detailed dependencies.
