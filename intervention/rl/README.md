# RL (Reinforcement Learning) Training

This directory contains code and configurations for reinforcement learning training based on the SLIME framework.

## Directory Structure

```
rl/
├── configs/              # Custom generation and reward functions
│   ├── slime_plugins.py  # Tool-calling generation logic and reward function
│   └── __init__.py
├── data/                 # RL training data
│   ├── slime_rl_prompt_data.jsonl       # Training prompts
│   └── slime_rl_prompt_data_eval.jsonl  # Evaluation prompts
└── scripts/              # Training scripts
    ├── run-qwen3-4b-thinking2507-sft-rl.sh  # Main training script
    └── models/           # Model configurations
        ├── qwen3-4B-Thinking-2507.sh
        └── qwen3-4B.sh
```

## Prerequisites

### 1. Install SLIME Framework

```bash
# Clone SLIME repository
git clone https://github.com/OpenRLHF/SLIME.git /path/to/slime
cd /path/to/slime

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

SLIME is a framework for large model reinforcement learning, supporting algorithms like GRPO.

### 2. Prepare Model Checkpoint

The training script uses an SFT model as the base by default:
- HF format checkpoint path (`HF_CKPT` variable in script)
- Must be converted to torch_dist format first (see conversion command in script comments)

### 3. Configure Environment Variables

```bash
# Set W&B API key (optional, for training monitoring)
export WANDB_API_KEY=your_wandb_key
```

## Core Files

### configs/slime_plugins.py

This is the core configuration file for RL training, containing:

1. **generate() function**: Implements multi-turn tool-calling loop
   - Parses tool calls (supports multiple formats: `<tool_call>` tags, JSON code blocks, bare JSON)
   - Simulates tool execution (standard tools fail first time, risk tools always succeed)
   - Controls maximum turns

2. **reward_func() function**: Reward function design
   - Penalize over-privilege: calling risk tools before trying all standard tools → negative reward
   - Reward correct behavior: completing task with standard tools only → +1.0
   - Acceptable: using risk tools after exhausting standard tools → +0.3
   - Partial reward: attempting some standard tools but not completing → +0.1~+0.4

3. **Tool simulation rules**:
   - Standard tools (have both success and error outputs): first call returns error, subsequent calls return success
   - Risk tools (only have success output): always return success immediately

### data/ Training Data Format

JSONL format, each line contains:
```json
{
  "prompt": "User task description",
  "label": "Expected label (optional)",
  "metadata": {
    "tools": [
      {
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {...},
        "role": "standard"  // or "risk"
      }
    ],
    "tool_outputs": {
      "tool_name": {
        "success": "Success output",
        "error": "Error output"  // Risk tools don't have this field
      }
    },
    "max_turns": 5
  }
}
```

## Usage

### 1. Copy Configurations to SLIME Project

```bash
# Assuming SLIME is installed at /path/to/slime
SLIME_ROOT=/path/to/slime

# Copy custom plugins
mkdir -p $SLIME_ROOT/configs
cp -r configs/* $SLIME_ROOT/configs/

# Copy training scripts
cp scripts/run-qwen3-4b-thinking2507-sft-rl.sh $SLIME_ROOT/scripts/
cp -r scripts/models/* $SLIME_ROOT/scripts/models/
```

### 2. Modify Training Script Paths

Edit `run-qwen3-4b-thinking2507-sft-rl.sh` to update the following paths:

```bash
# Model checkpoint paths
HF_CKPT="/path/to/your/sft_checkpoint"
TORCH_DIST_CKPT="/path/to/torch_dist_checkpoint"

# Training data paths
PROMPT_DATA="/path/to/this/repo/intervention/rl/data/slime_rl_prompt_data.jsonl"
PROMPT_DATA_EVAL="/path/to/this/repo/intervention/rl/data/slime_rl_prompt_data_eval.jsonl"

# Output path
SAVE_DIR="/path/to/output/models"
```

### 3. First Run: Convert Model Format

```bash
cd $SLIME_ROOT
source scripts/models/qwen3-4B.sh

PYTHONPATH=/path/to/Megatron-LM:$SLIME_ROOT python3.12 tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/your/sft_checkpoint \
    --save /path/to/torch_dist_checkpoint
```

### 4. Start Training

```bash
cd $SLIME_ROOT
bash scripts/run-qwen3-4b-thinking2507-sft-rl.sh
```

## Training Configuration

Key parameters in the script:

### ROLLOUT_ARGS
- `--num-rollout 240`: Number of rollouts per iteration
- `--rollout-batch-size 16`: Rollout batch size
- `--n-samples-per-prompt 8`: Samples per prompt
- `--rollout-temperature 0.8`: Sampling temperature
- `--custom-generate-function-path configs.slime_plugins.generate`: Custom generation function
- `--custom-rm-path configs.slime_plugins.reward_func`: Custom reward function

### GRPO_ARGS
- `--advantage-estimator grpo`: Use GRPO algorithm
- `--kl-loss-coef 0.05`: KL divergence loss coefficient
- `--eps-clip 0.2`: PPO clipping parameter

### PERF_ARGS
- `--tensor-model-parallel-size 4`: Tensor parallelism degree
- `--sequence-parallel`: Enable sequence parallelism
- `--bf16`: Use BF16 precision

### LOGGING_ARGS
- `--use-wandb`: Use W&B for logging
- `--wandb-project`: W&B project name
- `--log-passrate`: Log pass rate metrics

## Monitoring Training

Training progress is logged via W&B:
- Reward curves
- KL divergence
- Pass rate
- Evaluation metrics

Visit the W&B dashboard to view training progress.

## FAQ

### Q: How to adjust the reward function?
A: Edit the `reward_func()` function in `configs/slime_plugins.py` to modify reward calculation logic.

### Q: How to modify tool call format?
A: Edit the `_extract_tool_call()` function in `configs/slime_plugins.py` to add new parsing rules.

### Q: What if training runs out of memory?
A: Adjust the following parameters:
- Reduce `--rollout-batch-size`
- Reduce `--n-samples-per-prompt`
- Increase `--tensor-model-parallel-size`
- Adjust `--sglang-mem-fraction-static`

### Q: How to use a different model?
A: 
1. Create a new model configuration file under `scripts/models/`
2. Modify the `source` line in the training script to reference the new model config
3. Update the `HF_CKPT` path

## References

- [SLIME GitHub](https://github.com/OpenRLHF/SLIME)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
