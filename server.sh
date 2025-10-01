#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem 160G
#SBATCH -c 16

echo "launching LLM Server"

hostname

module load cuda

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

python -m uvicorn server:app --host $SERVER_HOST --port $SERVER_PORT --workers $SERVER_WORKERS