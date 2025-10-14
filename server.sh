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

python -m uvicorn server:app --host $SERVER_HOST --port $SERVER_PORT --workers $SERVER_WORKERS