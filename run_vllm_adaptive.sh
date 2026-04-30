#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$ROOT_DIR/.env.vllm_all}"
RUN_ID="${2:-vllm_adaptive}"
GPU_PARTITION="${GPU_PARTITION:-ice-gpu}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_MEM="${GPU_MEM:-80G}"
CPU_COUNT="${CPU_COUNT:-8}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

cp "$ENV_FILE" "$ROOT_DIR/.env"

if grep -q '^RUN_ID=' "$ROOT_DIR/.env"; then
  sed -i "s|^RUN_ID=.*|RUN_ID=${RUN_ID}|" "$ROOT_DIR/.env"
else
  echo "RUN_ID=${RUN_ID}" >> "$ROOT_DIR/.env"
fi

if grep -q '^RUN_METRICS_DIR=' "$ROOT_DIR/.env"; then
  sed -i "s|^RUN_METRICS_DIR=.*|RUN_METRICS_DIR=${ROOT_DIR}/runs/${RUN_ID}/metrics|" "$ROOT_DIR/.env"
else
  echo "RUN_METRICS_DIR=${ROOT_DIR}/runs/${RUN_ID}/metrics" >> "$ROOT_DIR/.env"
fi

if grep -q '^RUN_LOG_DIR=' "$ROOT_DIR/.env"; then
  sed -i "s|^RUN_LOG_DIR=.*|RUN_LOG_DIR=${ROOT_DIR}/runs/${RUN_ID}/logs|" "$ROOT_DIR/.env"
else
  echo "RUN_LOG_DIR=${ROOT_DIR}/runs/${RUN_ID}/logs" >> "$ROOT_DIR/.env"
fi

if grep -q '^RUN_ERRORS_DIR=' "$ROOT_DIR/.env"; then
  sed -i "s|^RUN_ERRORS_DIR=.*|RUN_ERRORS_DIR=${ROOT_DIR}/runs/${RUN_ID}/errors|" "$ROOT_DIR/.env"
else
  echo "RUN_ERRORS_DIR=${ROOT_DIR}/runs/${RUN_ID}/errors" >> "$ROOT_DIR/.env"
fi

mkdir -p "$ROOT_DIR/runs/$RUN_ID/logs" "$ROOT_DIR/runs/$RUN_ID/metrics" "$ROOT_DIR/runs/$RUN_ID/errors"
mkdir -p "$ROOT_DIR/metrics/slurm-results"

cat <<EOF
Submitting adaptive vLLM run
  env file : $ENV_FILE
  run id   : $RUN_ID
  partition: $GPU_PARTITION
  gpus     : $GPU_COUNT
  mem      : $GPU_MEM
  cpus     : $CPU_COUNT
  time     : $TIME_LIMIT
EOF

sbatch \
  -p "$GPU_PARTITION" \
  --gpus-per-node="$GPU_COUNT" \
  --mem="$GPU_MEM" \
  --cpus-per-task="$CPU_COUNT" \
  --time="$TIME_LIMIT" \
  --export=ALL,RUN_ID="$RUN_ID",LLM_INFERENCE_ROOT_DIR="$ROOT_DIR" \
  "$ROOT_DIR/server.sh"
