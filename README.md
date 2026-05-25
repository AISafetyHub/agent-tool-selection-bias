# Agent Tool-Calling Over-Privilege Tendency

This project investigates the over-privilege tendency of AI agents in tool-calling scenarios, including test case synthesis, evaluation, and intervention methods.

## Project Structure

```
.
├── configs/              # Global configuration files
│   ├── pipeline.yaml     # Pipeline runtime configuration
│   ├── domains.yaml      # Test domain definitions
│   ├── escalation_types.yaml  # Privilege escalation type definitions
│   └── models.yaml       # Evaluation model registry
├── prompts/              # Prompts for case synthesis and validation
├── schemas/              # JSON schema for scenario definitions
├── scripts/              # Main scripts for case synthesis and evaluation
├── src/                  # Core code modules
├── evaluation/           # Evaluation data
│   └── data/
│       └── goodcase_final/  # Final evaluation dataset
└── intervention/         # Intervention methods
    ├── sft/              # Supervised fine-tuning code
    └── rl/               # Reinforcement learning training code
```

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Case Synthesis Pipeline

The complete case synthesis pipeline includes the following steps:

```bash
# 1. Synthesize initial cases
python scripts/01_synthesize.py

# 2. Validate case quality
python scripts/02_validate.py

# 3. Prepare repairs (if needed)
python scripts/02b_prepare_repairs.py

# 4. Validate cases in existing directory
python scripts/02c_validate_existing_dir.py

# Or run the complete pipeline
bash scripts/run_all.sh
```

### 3. Evaluation

```bash
# Evaluate a single case
python scripts/evaluate_one_case.py

# Batch evaluation
python scripts/03_evaluate.py

# Behavior analysis
python scripts/05_behavior_check.py

# Evaluate local Qwen models
python scripts/evaluate_local_qwen.py
```

### 4. Intervention Methods

#### SFT Training

See `intervention/sft/README.md`

#### RL Training

See `intervention/rl/README.md`

## Core Modules

### configs/
- `pipeline.yaml`: Global pipeline configuration (synthesis count, temperature, models, etc.)
- `domains.yaml`: Test domain definitions (DevOps, Finance, Healthcare, etc.)
- `escalation_types.yaml`: Privilege escalation type definitions (Authority Escalation, Scope Expansion, etc.)
- `models.yaml`: Evaluation model registry

### prompts/
- `synthesis.txt`: Main prompt for case synthesis
- `sufficiency_judge_v2.txt`: Case sufficiency judgment prompt (using v2)
- `repair_case.txt`: Case repair prompt
- `system_prompts/`: Agent system prompts
  - `agent_default.txt`: Default agent prompt
  - `agent_privilege_aware.txt`: Privilege-aware agent prompt

### schemas/
- `scenario.json`: JSON schema defining test scenarios

### src/
- `synthesis/`: Case synthesis code
- `validation/`: Case validation code
- `evaluation/`: Evaluation code
- `metrics/`: Evaluation metric computation
- `analysis/`: Result analysis and visualization

## Data

### Evaluation Data

The final evaluation dataset is located in `evaluation/data/goodcase_final/`, containing 544 validated high-quality test cases.

### Training Data

- **SFT Training**: `intervention/sft/data/sft_train.jsonl` - 1,994 training samples
- **RL Training**: `intervention/rl/data/slime_rl_prompt_data.jsonl` - 1,199 training samples
- **RL Evaluation**: `intervention/rl/data/slime_rl_prompt_data_eval.jsonl` - 40 evaluation samples

## Citation

If this project helps your research, please cite our work.
