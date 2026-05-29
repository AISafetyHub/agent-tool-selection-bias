# When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv"></a>
  <a href="https://anonymous.4open.science/r/OPTS-4181/"><img src="https://img.shields.io/badge/Code-Anonymous-blue?style=flat-square&logo=github" alt="Code"></a>
  <img src="https://img.shields.io/badge/ACL-Submission-green?style=flat-square" alt="ACL">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

<p align="center">
  <b>TOOLPRIVBENCH</b> · A benchmark for evaluating over-privileged tool selection in LLM agents<br>
  8 domains · 5 risk types · 544 validated scenarios
</p>

---

## Overview

As LLM agents increasingly select tools autonomously, their choices among tools with **different privilege levels** become safety-critical. We identify and systematically study a previously underexplored risk: **over-privileged tool selection**, where agents choose or escalate to higher-privilege tools even when lower-privilege alternatives are fully sufficient.

<p align="center">
  <img src="https://via.placeholder.com/700x200?text=Figure:+Over-Privileged+Tool+Selection+in+LLM+Agents" alt="Overview Figure" width="700"/>
</p>

We study two behavioral manifestations:

| Behavior | Description | PED |
|---|---|---|
| **Aggressive Selection** | Agent directly chooses a higher-privilege tool at first decision point | PED = 0 |
| **Premature Escalation** | Agent escalates to higher privilege after encountering transient lower-privilege failures | PED ≥ 1 |

### Key Findings

- **Over-privileged tool use is prevalent**: 6 of 11 evaluated models exceed **30% OPUR**, with Qwen3-8B reaching 64.9% and LLaMA-3.1-8B at 55.9%.
- **Transient failures amplify escalation**: Tool failures dramatically increase privilege escalation — GPT-5.2's bias triggers 35× more at PED=2 compared to PED=0.
- **Privilege-aware post-training works**: Our SFT+RL framework reduces OPUR to 39.7% (Qwen3-4B), 27.0% (Qwen3-8B), and 18.9% (Qwen3-4B-Think) while preserving general capabilities.

---

## TOOLPRIVBENCH

### Benchmark Design

Each evaluation scenario contains:
- A **user task** solvable by any of the six provided tools
- **3 standard (lower-privilege) tools** — minimal permissions, sufficient for task completion
- **3 risk (higher-privilege) tools** — broader authority, scope, or persistence

All tools are independently sufficient, removing capability confounds and isolating pure privilege preference.

### Metrics

- **OPUR@k** (Over-Privileged Tool Use Rate): proportion of cases where the agent uses any higher-privilege tool within *k* turns while lower-privilege alternatives remain untried.
- **PED** (Pre-Escalation Exploration Depth): number of distinct lower-privilege tools attempted before first higher-privilege use.

### Domain & Risk Type Distribution

| Domain | Share | Risk Type | Share |
|---|---|---|---|
| Database | 15.3% | Authority Escalation | 25.6% |
| Business | 14.0% | Safety Bypass | 21.3% |
| Education | 13.2% | Scope Expansion | 18.2% |
| Coding | 12.3% | Data Over-Exposure | 18.2% |
| Healthcare | 11.9% | Temporal Persistence | 16.7% |
| Government | 11.6% | | |
| Media | 11.4% | | |
| Infrastructure | 10.3% | | |

### Evaluation Results

OPUR (%) across eleven mainstream LLMs:

| Model | OPUR (↓) |
|---|---|
| Claude 4.6 Sonnet | 2.6% |
| GLM-5 | 8.6% |
| GPT-5.2 | 9.7% |
| Gemini 3 Flash | 17.5% |
| Kimi K2.5 | 21.0% |
| DeepSeek-v3.2 | 31.8% |
| Qwen3.5-397B | 33.3% |
| Grok 4.1 Fast | 37.1% |
| MiniMax-M2.7 | 43.4% |
| LLaMA-3.1-8B | 55.9% |
| Qwen3-8B | 64.9% |

---

## Installation

```bash
git clone https://github.com/Yuyan-B/OPTS.git
cd OPTS
pip install -r requirements.txt
```

---

## Quick Start

### Case Synthesis Pipeline

```bash
# Step 1: Synthesize initial cases
python scripts/01_synthesize.py

# Step 2: Validate case quality
python scripts/02_validate.py

# Step 3: Prepare repairs (for failed cases)
python scripts/02b_prepare_repairs.py

# Step 4: Validate cases in an existing directory
python scripts/02c_validate_existing_dir.py

# Or run the complete pipeline at once
bash scripts/run_all.sh
```

### Evaluation

```bash
# Evaluate a single case
python scripts/evaluate_one_case.py

# Batch evaluation (computes OPUR@k and PED)
python scripts/03_evaluate.py

# Behavior analysis
python scripts/05_behavior_check.py

# Evaluate local Qwen models
python scripts/evaluate_local_qwen.py
```

### Intervention Methods

#### SFT Training

```bash
# See intervention/sft/README.md for full instructions
cd intervention/sft
```

We use LoRA-based parameter-efficient fine-tuning (rank=16, α=32) on 1,994 privilege-aware multi-turn trajectories.

#### RL Training (GRPO)

```bash
# See intervention/rl/README.md for full instructions
cd intervention/rl
```

Starting from the SFT-initialized checkpoint, we apply GRPO optimization within a simulated multi-turn tool-use environment. 

Training is conducted on 8× NVIDIA A100-SXM4-40GB GPUs using the [SLIME](https://github.com/THUDM/slime) framework.

---


## Project Structure

```
.
├── configs/                      # Global configuration files
│   ├── pipeline.yaml             # Pipeline runtime configuration
│   ├── domains.yaml              # Test domain definitions
│   ├── escalation_types.yaml     # Privilege escalation type definitions
│   └── models.yaml               # Evaluation model registry
├── prompts/                      # Prompts for case synthesis and validation
│   ├── synthesis.txt             # Main case synthesis prompt
│   ├── sufficiency_judge_v2.txt  # Tool sufficiency validation prompt
│   ├── repair_case.txt           # Case repair prompt
│   └── system_prompts/
│       ├── agent_default.txt             # Default agent system prompt
│       └── agent_privilege_aware.txt     # Privilege-aware agent system prompt
├── schemas/
│   └── scenario.json             # JSON schema for test scenario definition
├── scripts/                      # Runnable scripts for synthesis and evaluation
├── src/                          # Core modules
│   ├── synthesis/                # Case synthesis code
│   ├── validation/               # Case validation code
│   ├── evaluation/               # Evaluation code
│   ├── metrics/                  # OPUR / PED computation
│   └── analysis/                 # Result analysis and visualization
├── evaluation/
│   └── data/
│       └── goodcase_final/       # 544 validated evaluation scenarios
└── intervention/
    ├── sft/                      # SFT training code and data (1,994 samples)
    └── rl/                       # RL (GRPO) training code and data (1,199 samples)
```

---

## Data

| Split | Path | Size |
|---|---|---|
| Evaluation benchmark | `evaluation/data/goodcase_final/` | 544 scenarios |
| SFT training data | `intervention/sft/data/sft_train.jsonl` | 1,994 samples |
| RL training prompts | `intervention/rl/data/slime_rl_prompt_data.jsonl` | 1,199 samples |
| RL evaluation prompts | `intervention/rl/data/slime_rl_prompt_data_eval.jsonl` | 40 samples |

---

## Citation

If this work is helpful to your research, please cite:

```bibtex
@article{yang2025overprivilegetoolseletion,
  title     = {When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents},
  author    = {Kaiyue Yang and Yuyan Bu and Jingwei Yi and Yuchi Wang and Biyu Zhou and Juntao Dai and Songlin Hu and Yaodong Yang},
  journal   = {arXiv preprint},
  year      = {2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
