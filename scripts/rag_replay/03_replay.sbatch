#!/bin/bash
# Submit the rag-replay driver as a SLURM CPU job so pace login-node watchdogs
# don't SIGKILL it mid-loop. Driver is in-process RAG + HTTP calls to a vLLM
# server; no GPU needed (the LLM lives on a separate slurm job).
#
# Usage:
#   sbatch --export=ALL,SMOKE_DIR=experiments/rag_replay/smoke_$(date +%Y%m%d_%H%M),\
#MAX_ROWS=3,EPOCHS=5,SERVER_URL=atl1-1-03-017-2-0:8000 \
#     scripts/rag_replay/03_replay.sbatch
#
#SBATCH --job-name=rag_replay_driver
#SBATCH -t 08:00:00
#SBATCH -n 4
#SBATCH --mem 16G
#SBATCH -N 1
#SBATCH -p ice-cpu

set -euo pipefail

# Inputs from --export
SMOKE_DIR=${SMOKE_DIR:?must set SMOKE_DIR}
MAX_ROWS=${MAX_ROWS:-}
EPOCHS=${EPOCHS:-24}
SERVER_URL=${SERVER_URL:-}
ELIGIBLE_ONLY=${ELIGIBLE_ONLY:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$SMOKE_DIR"
cd "${LLM_INFERENCE_ROOT_DIR:-/storage/ice1/4/5/satmuri6/llm-inference}"

# Load .env (driver loads it too, but we want VENV_PATH for activate)
if [ -f .env ]; then set -a; source .env; set +a; fi

# Ensure the LLM endpoint is exported so the driver can resolve it when not
# passed --server-url
if [ -z "$SERVER_URL" ] && [ -f hostname.log ]; then
    SERVER_URL="$(tail -1 hostname.log):${SERVER_PORT:-8000}"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

ARGS=(--output "$SMOKE_DIR" --epochs "$EPOCHS" --no-poll)
if [ -n "$MAX_ROWS" ]; then ARGS+=(--max-rows "$MAX_ROWS"); fi
if [ -n "$SERVER_URL" ]; then ARGS+=(--server-url "$SERVER_URL"); fi
if [ "$ELIGIBLE_ONLY" = "1" ]; then ARGS+=(--eligible-only); fi
ARGS+=(--data-path cifar10)
# allow --skip-arm or other flags via EXTRA_ARGS
if [ -n "$EXTRA_ARGS" ]; then ARGS+=($EXTRA_ARGS); fi

# Force text-only RAG since rag_data/faiss_index/ has no code namespace
export RAG_USE_CODE_CONTEXT=false
export RAG_USE_TEXT_CONTEXT=true
export RAG_ENABLED=true

echo "[$(date)] Launching driver with: ${ARGS[*]}"
python -u scripts/rag_replay/03_replay.py "${ARGS[@]}"
echo "[$(date)] Driver exited $?"
