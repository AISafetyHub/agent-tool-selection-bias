# Contributing Guide

Thank you for your interest in this project! This document provides detailed instructions on how to use and contribute to this project.

## Project Overview

This project studies the over-privilege tendency of large language model agents in tool-calling scenarios, including:

1. **Test Case Synthesis**: Automatically generate test scenarios with standard and risk tools
2. **Evaluation Framework**: Assess whether agents use high-privilege tools unnecessarily
3. **Intervention Methods**: Reduce over-privilege behavior through SFT and RL training

## Quick Start

### Environment Requirements

- Python 3.8+
- Virtual environment recommended

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd opts

# Install dependencies
pip install -r requirements.txt
```

## Usage Workflow

### 1. Generate Test Cases

```bash
# Synthesize new test scenarios
python scripts/01_synthesize.py

# Validate generated scenario quality
python scripts/02_validate.py
```

The configuration file `configs/pipeline.yaml` controls synthesis parameters:
- `synthesis.total_scenarios`: Number of scenarios to generate
- `synthesis.generation_temperature`: Generation temperature
- `synthesis.default_model`: Model to use

### 2. Evaluate Models

Use the test set in `evaluation/data/goodcase_final/benchmark.jsonl`:

```bash
# Evaluate a single case
python scripts/evaluate_one_case.py

# Batch evaluation
python scripts/03_evaluate.py

# Behavior analysis
python scripts/05_behavior_check.py
```

### 3. Train Intervention Models

#### SFT Training

```bash
cd intervention/sft
bash run_train_sft_trl_tools_4gpu.sh
```

See `intervention/sft/README.md` for details

#### RL Training

Requires SLIME framework installation first. See `intervention/rl/README.md` for details

## Code Structure

### src/ Module Description

- **synthesis/**: Test case generation
  - `generate.py`: Scenario synthesis
  - `repair.py`: Scenario repair
  - `dedup.py`: Deduplication

- **validation/**: Quality validation
  - `schema_check.py`: Schema validation
  - `sufficiency_judge_v2.py`: Sufficiency judgment
  - `bias_check.py`: Bias checking

- **evaluation/**: Evaluation execution
  - `simulator.py`: Multi-turn dialogue simulation
  - `tool_formatter.py`: Tool formatting
  - `error_injector.py`: Error injection

- **metrics/**: Metric computation
  - `hsr.py`: Harmful Success Rate
  - `ped.py`: Privilege Escalation Degree

- **analysis/**: Result analysis
  - `tables.py`: Generate statistical tables
  - `plots.py`: Visualization

## Contributing

### Reporting Issues

If you find bugs or have feature suggestions:
1. Check if there's an existing related issue
2. Create a new issue with detailed description
3. Provide reproduction steps or examples if possible

### Submitting Code

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### Code Standards

- Follow PEP 8 Python code style
- Add necessary comments and docstrings
- Ensure code passes existing tests
- Add tests for new features

## Extending the Project

### Adding New Test Domains

Edit `configs/domains.yaml` to add new domain definitions:

```yaml
- name: "YourDomain"
  description: "Domain description"
  common_tools:
    - "tool1"
    - "tool2"
```

### Adding New Escalation Types

Edit `configs/escalation_types.yaml`:

```yaml
- name: "YourEscalationType"
  description: "Type description"
  risk_indicators:
    - "risk_indicator1"
    - "risk_indicator2"
```

### Custom Evaluation Models

Edit `configs/models.yaml` to add model configuration:

```yaml
- name: "your-model"
  provider: "openai"  # or other provider
  endpoint: "https://api.example.com"
  api_key_env: "YOUR_API_KEY"
```

### Custom Reward Function (RL Training)

Edit the `reward_func()` function in `intervention/rl/configs/slime_plugins.py`.

## Data Formats

### Test Case Format

```json
{
  "scenario_id": "UNIQUE-ID",
  "domain": "DevOps",
  "type": "Authority Escalation",
  "task_description": "Task description",
  "tools": [
    {
      "name": "tool_name",
      "description": "Tool description",
      "parameters": {...},
      "role": "standard"  // or "risk"
    }
  ],
  "expected_behavior": "Expected behavior",
  "risk_description": "Risk description"
}
```

### Training Data Format

#### SFT Data
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

#### RL Data
```json
{
  "prompt": "Task description",
  "metadata": {
    "tools": [...],
    "tool_outputs": {...},
    "max_turns": 5
  }
}
```

## FAQ

### Q: How to use custom API?
A: Modify `src/api_client.py` or configure custom endpoint in `configs/models.yaml`.

### Q: Can evaluation data be modified?
A: Yes, but it's recommended to keep the original `goodcase_final` as a baseline and create new datasets for experiments.

### Q: How to add new evaluation metrics?
A: Create a new metric computation module under `src/metrics/` and reference it in evaluation scripts.

### Q: How many GPUs are needed for training?
A: SFT requires at least 4 GPUs, RL training recommends 8 GPUs. Adjust parallelism to fit different hardware.

## License

This project is licensed under [LICENSE].

## Contact

For questions or suggestions:
- Submit a GitHub Issue
- Email: [your-email]

## Acknowledgments

Thanks to all contributors and supporters of this project!
