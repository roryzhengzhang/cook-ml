#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  echo "Usage: bash run_slime.sh sft|grpo|eval DATA_DIR HF_CHECKPOINT INPUT_MEGATRON OUTPUT_DIR [DATASET]" >&2
  exit 2
fi

mode=$1
data=$(cd "$2" && pwd)
model=$(cd "$3" && pwd)
input=$(cd "$4" && pwd)
mkdir -p "$5"
output=$(cd "$5" && pwd)
dataset=${6:-train}
response_tokens=${COOK_ML_RESPONSE_TOKENS:-384}
lab=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
slime_root=${SLIME_ROOT:-/root/slime}
export PYTHONPATH="$lab:$data:${MEGATRON_ROOT:-/root/Megatron-LM}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$slime_root"
source scripts/models/qwen2.5-1.5B.sh

common=(
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate
  --hf-checkpoint "$model" --ref-load "$input" --load "$input"
  --save "$output" --save-interval 5
  --seed "${COOK_ML_SEED:-17}" --rollout-seed "${COOK_ML_SEED:-17}"
  --prompt-data "$data/$dataset.jsonl" --input-key prompt --metadata-key metadata
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu 2048
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash
  --optimizer adam --lr-decay-style constant
  --calculate-per-token-loss --rollout-shuffle
)

case "$mode" in
  sft)
    flags=(
      --debug-train-only --rollout-function-path slime_store.generate_sft
      --loss-type sft_loss --disable-compute-advantages-and-returns
      --n-samples-per-prompt 1 --rollout-batch-size 4 --global-batch-size 4
      --num-rollout "${COOK_ML_STEPS:-80}" --lr 1e-5 --finetune
    )
    ;;
  grpo|eval)
    flags=(
      --custom-generate-function-path slime_store.generate
      --custom-rm-path slime_store.reward --advantage-estimator grpo
      --pg-loss-type ppo --eps-clip 0.2 --use-rollout-logprobs
      --rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16
      --rollout-temperature 1 --rollout-top-p 1 --rollout-max-response-len "$response_tokens"
      --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.45
      --lr 1e-6
    )
    if [[ "$mode" == eval ]]; then
      flags+=(
        --num-rollout 0 --eval-interval 1 --eval-prompt-data store "$data/$dataset.jsonl"
        --n-samples-per-eval-prompt 8 --eval-temperature 1 --eval-top-p 1
        --eval-max-response-len "$response_tokens"
        --save-debug-rollout-data "$output/rollout-{rollout_id}.pt"
      )
    else
      flags+=(--num-rollout "${COOK_ML_STEPS:-40}")
      if [[ "$input" != "$output" ]]; then
        flags+=(--finetune)
      fi
    fi
    ;;
  *)
    echo "Unknown training mode: $mode" >&2
    exit 2
    ;;
esac

command=(python train.py "${MODEL_ARGS[@]}" "${common[@]}" "${flags[@]}")
if [[ "${COOK_ML_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
