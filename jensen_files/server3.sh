#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "H100"
#SBATCH --mem 160G
#SBATCH -c 16

echo "Starting vLLM server..."

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
else
    echo "Warning: .env file not found. Using default values."
    # Set defaults if .env doesn't exist
    export LLM_INFERENCE_ROOT_DIR="$(pwd)"
    export VENV_PATH="$(pwd)/.venv"
    export SERVER_HOST="0.0.0.0"
    export SERVER_PORT="8000"
    export SERVER_WORKERS="1"
    export HOSTNAME_LOG_FILE="$(pwd)/hostname.log"
    export CUDA_VISIBLE_DEVICES="0"
    export MKL_THREADING_LAYER="GNU"
fi

# ========= REDIRECT ALL CACHES/CONFIGS OFF $HOME =========
# Base scratch cache dir
export SCRATCH_CACHE_BASE="/home/hice1/jzhang3318/scratch/llm_cache"

# Generic XDG cache/config (many libs respect these)
export XDG_CACHE_HOME="${SCRATCH_CACHE_BASE}/xdg_cache"
export XDG_CONFIG_HOME="${SCRATCH_CACHE_BASE}/xdg_config"

# HuggingFace / transformers
export HF_HOME="${SCRATCH_CACHE_BASE}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"

# vLLM-specific cache + config
export VLLM_CACHE_DIR="${SCRATCH_CACHE_BASE}/vllm"
# If this version respects it, this moves usage stats too:
export VLLM_CONFIG_DIR="${SCRATCH_CACHE_BASE}/vllm_config"
# And just in case telemetry is on:
export VLLM_DISABLE_USAGE_STATS=1

# flashinfer JIT workspace (by default it uses ~/.cache/flashinfer)
export FLASHINFER_WORKSPACE_DIR="${SCRATCH_CACHE_BASE}/flashinfer"

# Create dirs
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$HF_HOME" "$VLLM_CACHE_DIR" "$VLLM_CONFIG_DIR" \
         "$FLASHINFER_WORKSPACE_DIR"


# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

# Set root directory if not already set
export LLM_INFERENCE_ROOT_DIR=${LLM_INFERENCE_ROOT_DIR:-$(pwd)}

# Activate virtual environment
if [ -d "${VENV_PATH}/bin" ]; then
    echo "Activating virtual environment at: $VENV_PATH"
    source "${VENV_PATH}/bin/activate"
else
    echo "Warning: Virtual environment not found at: $VENV_PATH"
fi

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=${HOSTNAME_LOG_FILE:-"${LLM_INFERENCE_ROOT_DIR}/hostname.log"}

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME"

# Use environment variables for server configuration
SERVER_HOST=${SERVER_HOST:-$SERVER_HOSTNAME}
SERVER_PORT=${SERVER_PORT:-8000}
SERVER_WORKERS=${SERVER_WORKERS:-1}

python -m vllm.entrypoints.openai.api_server \
  --model /home/hice1/jzhang3318/scratch/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --port 8000 \
  --host 0.0.0.0 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5}'
