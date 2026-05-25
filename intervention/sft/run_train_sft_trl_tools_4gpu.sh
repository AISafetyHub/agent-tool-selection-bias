#!/usr/bin/env bash
# 4-GPU DDP: uses physical GPUs 0,1,2,3 (visible as cuda:0..3 in each process).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/train_sft_trl_tools.py" "$@"
