#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B 1xGPU GRPO quickstart — GSM8K train only (no eval).
#
# Colocate mode: actor and rollout time-share the same GPU.
#
# train_iters = NUM_ROLLOUT × ROLLOUT_BATCH_SIZE × N_SAMPLES / GLOBAL_BATCH_SIZE
#             = 100 × 4 × 4 / 16 = 100 steps
#
# Dataset: openai/gsm8k (7473 problems)
#   Script normalizes the `answer` field (strips CoT, keeps only the number after ####)
#   into $DATA_DIR/gsm8k/main/train_clean.parquet on first run.
#   Original parquet answer is "推导全文 + #### 72"; math rm-type needs bare "72".
#
# Usage:
#   bash contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh
#
# Key overridable env vars:
#   MODEL_DIR  - dir containing Qwen3-0.6B/
#   DATA_DIR   - dir containing gsm8k/main/train-00000-of-00001.parquet
#
# Metrics to watch in ClearML:
#   rollout/raw_reward   — accuracy 0/1 (expect ~0.5–0.7 initial on GSM8K, rising over training)
#   train/pg_loss        — policy gradient loss (non-zero and decreasing)
#   train/grad_norm      — gradient norm (stable, not exploding)
#   NOTE: rollout/rewards and rollout/advantages are GRPO-normalized and will
#         always hover near 0 — this is correct behavior, not a bug.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/beginner-task}"
MODEL_DIR="${MODEL_DIR:-/your/model}"
DATA_DIR="${DATA_DIR:-/your/data}"

GSM8K_RAW="${DATA_DIR}/gsm8k/main/train-00000-of-00001.parquet"
GSM8K_CLEAN="${DATA_DIR}/gsm8k/main/train_clean.parquet"
if [ ! -f "${GSM8K_CLEAN}" ]; then
    python3 - <<EOF
import pandas as pd
df = pd.read_parquet("${GSM8K_RAW}")
df["answer"] = df["answer"].str.split("####").str[-1].str.strip()
df.to_parquet("${GSM8K_CLEAN}", index=False)
print(f"Wrote {len(df)} rows to ${GSM8K_CLEAN}")
EOF
fi

NUM_ROLLOUT="${NUM_ROLLOUT:=100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=4}"
N_SAMPLES="${N_SAMPLES:=8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=16}"
# train_iters = 100 × 4 × 8 / 16 = 200

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
    --ref-load ${MODEL_DIR}/Qwen3-0.6B
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data ${GSM8K_CLEAN}
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle

    --rm-type math

    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
    --n-samples-per-prompt ${N_SAMPLES}
    --rollout-max-response-len 2048
    --rollout-temperature 1

    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --balance-data
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --log-probs-max-tokens-per-gpu 8192
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2

    --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    # ~55% for training; SGLang uses 45%
    --sglang-mem-fraction-static 0.45
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name qwen3-0.6b-GRPO-gsm8k-1xgpu-${now}
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-0.6b-GRPO-gsm8k-1xgpu-${now}.log
