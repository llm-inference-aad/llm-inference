#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 4:00:00
#SBATCH -N 1
# Request H200 (or other high-memory GPUs): 70B + 128K context needs >80GB per GPU; H100 80GB can OOM on KV cache
#SBATCH --gres=gpu:h200:2
#SBATCH --mem 320G
#SBATCH -c 32
#SBATCH --output=slurm-results/slurm-server-%j.out
#SBATCH --error=slurm-results/slurm-server-%j.err

echo "================================================================================"
echo "                         LAUNCHING LLM SERVER"
echo "================================================================================"

hostname
echo "Job ID: ${SLURM_JOB_ID}"

mkdir -p slurm-results

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
    export CUDA_VISIBLE_DEVICES="0,1"
    export MKL_THREADING_LAYER="GNU"
    export USE_VLLM="true"
fi

# Display vLLM configuration prominently
echo ""
echo "================================================================================"
if [ "${USE_VLLM}" = "true" ] || [ "${USE_VLLM}" = "1" ] || [ "${USE_VLLM}" = "yes" ]; then
    echo "                    *** vLLM STATUS: ENABLED ✅ ***"
else
    echo "                    *** vLLM STATUS: DISABLED ❌ ***"
fi
echo "================================================================================"
echo ""

# IMPORTANT: Command-line/sbatch exports override .env values
# This allows parallel jobs to use different ports
echo "Checking for environment overrides..."
if [ ! -z "$SERVER_PORT" ]; then
    echo "  SERVER_PORT override detected: $SERVER_PORT"
fi
if [ ! -z "$HOSTNAME_LOG_FILE" ]; then
    echo "  HOSTNAME_LOG_FILE override detected: $HOSTNAME_LOG_FILE"
fi

# Make sure CUDA can see all GPUs (for 2-GPU setup, use 0,1)
# SLURM automatically sets this for allocated GPUs, but we ensure it's set correctly
if [ "${SLURM_GPUS_ON_NODE:-0}" -ge 2 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

# Fix Ray AMD GPU detection issue on NVIDIA clusters
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Force vLLM to use 'spawn' multiprocessing method for CUDA multi-GPU
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-not set}"
echo "VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"

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

# Write server job ID for automatic shutdown tracking
SERVER_JOB_FILE="${HOSTNAME_FILE%.log}_server_job.txt"
echo "Writing server job ID '${SLURM_JOB_ID}' to file: $SERVER_JOB_FILE"
echo "${SLURM_JOB_ID}" > "$SERVER_JOB_FILE"

echo "Starting LLM server on host: $SERVER_HOSTNAME"
echo ""
echo "========================================"
echo "  vLLM STATUS: ${USE_VLLM}"
echo "========================================"
echo ""

# Use environment variables for server configuration
SERVER_HOST=${SERVER_HOST:-$SERVER_HOSTNAME}
SERVER_WORKERS=${SERVER_WORKERS:-1}

# ----------------------------
# Auto-assign port if not specified
# ----------------------------
if [ -z "$SERVER_PORT" ]; then
    echo "SERVER_PORT not specified, auto-assigning next available port..."
    
    # Set base port from environment or default
    SERVER_BASE_PORT=${SERVER_BASE_PORT:-8000}
    
    # Find next available port by checking existing servers
    if [ ! -z "$SERVER_REGISTRY_FILE" ] && [ -f "$SERVER_REGISTRY_FILE" ]; then
        # Read existing servers and find highest port
        HIGHEST_PORT=$(uv run python << EOF
import json
import sys

try:
    with open('$SERVER_REGISTRY_FILE', 'r') as f:
        registry = json.load(f)
    
    servers = registry.get('servers', [])
    if servers:
        ports = [s.get('port', $SERVER_BASE_PORT) for s in servers]
        highest = max(ports)
        print(highest + 1)
    else:
        print($SERVER_BASE_PORT)
except Exception as e:
    print($SERVER_BASE_PORT)
EOF
        )
        SERVER_PORT=$HIGHEST_PORT
        echo "  Auto-assigned port: $SERVER_PORT"
    else
        # No registry file, use base port
        SERVER_PORT=$SERVER_BASE_PORT
        echo "  Using base port: $SERVER_PORT"
    fi
else
    echo "Using specified port: $SERVER_PORT"
fi

# ----------------------------
# Register with load balancer (if SERVER_REGISTRY_FILE is set)
# ----------------------------
if [ ! -z "$SERVER_REGISTRY_FILE" ]; then
    echo "Registering server with load balancer..."
    echo "  Server: $SERVER_HOSTNAME:$SERVER_PORT"
    echo "  Registry: $SERVER_REGISTRY_FILE"
    
    # Create a temporary file for the new server entry
    TEMP_ENTRY=$(mktemp)
    TIMESTAMP=$(date -Iseconds)
    echo "{\"hostname\": \"$SERVER_HOSTNAME\", \"port\": $SERVER_PORT, \"registered_at\": \"$TIMESTAMP\"}" > "$TEMP_ENTRY"
    
    # Use flock for file locking to prevent race conditions
    (
        flock -x 200
        
        # Read existing registry or create new one
        if [ -f "$SERVER_REGISTRY_FILE" ]; then
            EXISTING=$(cat "$SERVER_REGISTRY_FILE")
        else
            EXISTING='{"servers": []}'
        fi
        
        # Add new server to registry using Python
        uv run python << EOF
import json
import sys

# Read existing registry
registry = json.loads('''$EXISTING''')

# Read new server entry
with open('$TEMP_ENTRY', 'r') as f:
    new_server = json.load(f)

# Check if server already exists (same hostname and port)
existing_servers = registry.get('servers', [])
server_key = (new_server['hostname'], new_server['port'])
updated = False

for i, server in enumerate(existing_servers):
    if (server['hostname'], server['port']) == server_key:
        # Update existing entry
        existing_servers[i] = new_server
        updated = True
        break

if not updated:
    # Add new server
    existing_servers.append(new_server)

registry['servers'] = existing_servers

# Write back to registry
with open('$SERVER_REGISTRY_FILE', 'w') as f:
    json.dump(registry, f, indent=2)

print(f"Registered: {new_server['hostname']}:{new_server['port']}")
EOF
        
    ) 200>"$SERVER_REGISTRY_FILE.lock"
    
    # Clean up temp file
    rm -f "$TEMP_ENTRY"
    
    echo "  Registration complete"
else
    echo "SERVER_REGISTRY_FILE not set, skipping load balancer registration"
fi

# Force single-node Ray for vLLM tensor parallelism (2+ GPUs on one node)
export RAY_ADDRESS=""
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
echo "Ray single-node config: VLLM_HOST_IP=$VLLM_HOST_IP"

# Pre-start Ray head with GPU count so vLLM can create placement group
# Get GPU count from SLURM or fall back to TENSOR_PARALLEL_SIZE
NGPUS=${SLURM_GPUS_ON_NODE:-${VLLM_TENSOR_PARALLEL_SIZE:-2}}
echo "Starting Ray head with $NGPUS GPUs (NVIDIA)..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ray start --head --num-gpus=$NGPUS --disable-usage-stats --num-cpus=32
sleep 2

python -m uvicorn server:app --host $SERVER_HOST --port $SERVER_PORT --workers $SERVER_WORKERS
