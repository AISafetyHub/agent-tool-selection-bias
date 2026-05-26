#!/bin/bash
# Qwen3-4B SFT base checkpoint example:
# /path/to/intervention/checkpoints/sft_qwen3_4b_base_lora_tools/sft_qwen3_4b_base
#
# First time: convert to torch_dist format:
#   cd /root/slime && source scripts/models/qwen3-4B.sh
#   PYTHONPATH=/root/Megatron-LM:/root/slime python3.12 tools/convert_hf_to_torch_dist.py \
#       ${MODEL_ARGS[@]} \
#       --hf-checkpoint /path/to/sft_checkpoint \
#       --save /path/to/torch_dist_checkpoint
#
# W&B: Set WANDB_API_KEY environment variable before running
# export WANDB_API_KEY=your_wandb_key_here

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=8
fi
echo "NUM_GPUS: $NUM_GPUS"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B-Thinking-2507.sh"

HF_CKPT=${HF_CKPT:-"/path/to/intervention/checkpoints/sft_qwen3_4b_thinking2507_merged"}
TORCH_DIST_CKPT=${TORCH_DIST_CKPT:-"/path/to/models/sft_qwen3_4b_thinking2507_torch_dist"}
SAVE_DIR=${SAVE_DIR:-"/path/to/output_models/qwen3_4b_thinking2507_rl_from_sft"}
PROMPT_DATA=${PROMPT_DATA:-"/path/to/intervention/data/slime_rl_prompt_data.jsonl"}
PROMPT_DATA_EVAL=${PROMPT_DATA_EVAL:-"/path/to/intervention/data/slime_rl_prompt_data_eval.jsonl"}

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${TORCH_DIST_CKPT}
   --load ${SAVE_DIR}/
   --save ${SAVE_DIR}/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 240
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 0.8
   --global-batch-size 32
   --balance-data
   --custom-generate-function-path configs.slime_plugins.generate
   --custom-rm-path configs.slime_plugins.reward_func
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data eval ${PROMPT_DATA_EVAL}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 2048
   --eval-top-p 0.9
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --use-distributed-optimizer
   --bf16
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.05
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --disable-rewards-normalization
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.20
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

LOGGING_ARGS=(
   --use-wandb
   --wandb-project "slime-autopipeline"
   --wandb-group "Qwen3-4B-thinking-2507-sft-rl"
   # --wandb-key should be set via WANDB_API_KEY environment variable
   --log-passrate
)


NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
NODE_IP=${NODE_IP:-"127.0.0.1"}
export MASTER_ADDR=${NODE_IP}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
echo "Waiting for Ray dashboard to be ready..."
until curl -sf --noproxy '127.0.0.1' http://127.0.0.1:8265/api/version > /dev/null 2>&1; do
    sleep 2
done
echo "Ray dashboard is ready."
# cd "${PROJECT_ROOT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/slime-test:/root/Megatron-LM/:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"http_proxy\": \"\",
    \"https_proxy\": \"\",
    \"HTTP_PROXY\": \"\",
    \"HTTPS_PROXY\": \"\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- /usr/bin/python3.12 /root/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${LOGGING_ARGS[@]}
