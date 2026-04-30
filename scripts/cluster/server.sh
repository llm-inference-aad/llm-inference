#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem 32G
#SBATCH -c 4
#SBATCH --output=slurm-results/slurm-server-%j.out
#SBATCH --error=slurm-results/slurm-server-%j.err

# Ensure we run from repo root (in case sbatch was called from elsewhere)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

echo "launching LLM Server"

hostname

mkdir -p slurm-results

module load cuda

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        # Preserve sbatch/command-line overrides.
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < .env
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

# IMPORTANT: Command-line/sbatch exports override .env values
# This allows parallel jobs to use different ports
echo "Checking for environment overrides..."
if [ ! -z "$SERVER_PORT" ]; then
    echo "  SERVER_PORT override detected: $SERVER_PORT"
fi
if [ ! -z "$HOSTNAME_LOG_FILE" ]; then
    echo "  HOSTNAME_LOG_FILE override detected: $HOSTNAME_LOG_FILE"
fi

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

# Set root directory if not already set
export LLM_INFERENCE_ROOT_DIR=${LLM_INFERENCE_ROOT_DIR:-$(pwd)}
export USE_VLLM=${USE_VLLM:-true}
export VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-true}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
export VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
echo "Inference backend config: USE_VLLM=$USE_VLLM, VLLM_ENABLE_PREFIX_CACHING=$VLLM_ENABLE_PREFIX_CACHING"
echo "vLLM runtime config: VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD, VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN, VLLM_DTYPE=$VLLM_DTYPE"

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
    
    # Find next available port by checking existing servers.
    if [ ! -z "$SERVER_REGISTRY_FILE" ] && [ -f "$SERVER_REGISTRY_FILE" ]; then
        # Read existing servers and find highest port
        HIGHEST_PORT=$(python - "$SERVER_REGISTRY_FILE" "$SERVER_BASE_PORT" << 'PYEOF'
import json, sys
try:
    reg_file = sys.argv[1]
    base = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    with open(reg_file, 'r') as f:
        registry = json.load(f)
    servers = registry.get('servers', [])
    if servers:
        ports = [int(s.get('port', base)) for s in servers if s.get('port') is not None]
        print(max(ports) + 1 if ports else base)
    else:
        print(base)
except Exception:
    print(8000)
PYEOF
        )
        SERVER_PORT=$(echo "$HIGHEST_PORT" | tr -d '[:space:]')
        SERVER_PORT=${SERVER_PORT:-$SERVER_BASE_PORT}
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
        python << EOF
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

# Fallback: ensure SERVER_PORT is set (port auto-assign may fail with corrupted registry)
SERVER_PORT=$(echo "${SERVER_PORT:-}" | tr -d '[:space:]')
SERVER_PORT=${SERVER_PORT:-8000}
echo "Starting uvicorn on port $SERVER_PORT"
python -m uvicorn app.server:app --host "$SERVER_HOST" --port "${SERVER_PORT}" --workers "$SERVER_WORKERS"
