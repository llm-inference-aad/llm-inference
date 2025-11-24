#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=slurm-results/slurm-server-%j.out
#SBATCH --error=slurm-results/slurm-server-%j.err

echo "launching LLM Server"

hostname

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

# Write server job ID for automatic shutdown tracking
SERVER_JOB_FILE="${HOSTNAME_FILE%.log}_server_job.txt"
echo "Writing server job ID '${SLURM_JOB_ID}' to file: $SERVER_JOB_FILE"
echo "${SLURM_JOB_ID}" > "$SERVER_JOB_FILE"

echo "Starting LLM server on host: $SERVER_HOSTNAME"

# Use environment variables for server configuration
SERVER_HOST=${SERVER_HOST:-$SERVER_HOSTNAME}
SERVER_WORKERS=${SERVER_WORKERS:-1}

# ----------------------------
# Safe Port Assignment: Check availability and find free port if needed
# ----------------------------
function is_port_available() {
    local port=$1
    # Check if port is in use (returns 0 if available, 1 if taken)
    ! ss -tln | grep -q ":${port} "
}

function find_free_port() {
    local start_port=$1
    local max_attempts=50
    for ((i=0; i<max_attempts; i++)); do
        local candidate=$((start_port + i))
        if is_port_available $candidate; then
            echo $candidate
            return 0
        fi
    done
    echo "ERROR: Could not find free port in range ${start_port}-$((start_port + max_attempts))" >&2
    return 1
}

# Check if our previous server is still running (zombie prevention)
if [ -f "$SERVER_JOB_FILE" ]; then
    OLD_JOB_ID=$(cat "$SERVER_JOB_FILE" 2>/dev/null)
    if [ ! -z "$OLD_JOB_ID" ] && [ "$OLD_JOB_ID" != "${SLURM_JOB_ID}" ]; then
        # Check if old job is still running
        if squeue -j "$OLD_JOB_ID" &>/dev/null; then
            echo "WARNING: Previous server job $OLD_JOB_ID is still running!"
            echo "  This may cause port conflicts. Consider canceling it: scancel $OLD_JOB_ID"
        fi
    fi
fi

# ----------------------------
# Port availability check and cleanup
# ----------------------------
check_port_available() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

kill_port_process() {
    local port=$1
    echo "  Attempting to free port $port..."
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "  Found processes using port $port: $pids"
        # Only kill if it's a uvicorn/python server process (safety check)
        for pid in $pids; do
            local cmd=$(ps -p $pid -o comm= 2>/dev/null)
            if [[ "$cmd" == "python"* ]] || [[ "$cmd" == "uvicorn"* ]]; then
                echo "  Killing $cmd process (PID: $pid)"
                kill -9 $pid 2>/dev/null
                sleep 1
            else
                echo "  WARNING: Process $pid ($cmd) is not a Python/uvicorn server - skipping"
            fi
        done
    fi
}

# ----------------------------
# Smart Port Assignment: Use specified port or find free one
# ----------------------------
SERVER_BASE_PORT=${SERVER_BASE_PORT:-8000}

if [ -z "$SERVER_PORT" ]; then
    echo "SERVER_PORT not specified, using base port: $SERVER_BASE_PORT"
    SERVER_PORT=$SERVER_BASE_PORT
else
    echo "SERVER_PORT specified as: $SERVER_PORT"
fi

# Check if the target port is available
if ! is_port_available $SERVER_PORT; then
    echo "WARNING: Port $SERVER_PORT is already in use!"
    
    # Try to find what's using it (for debugging)
    echo "Process using port $SERVER_PORT:"
    ss -tlnp | grep ":${SERVER_PORT} " || echo "  (could not determine process)"
    
    # Auto-assign a free port
    echo "Auto-assigning next available port..."
    FREE_PORT=$(find_free_port $SERVER_BASE_PORT)
    if [ $? -eq 0 ]; then
        SERVER_PORT=$FREE_PORT
        echo "  ✓ Assigned free port: $SERVER_PORT"
    else
        echo "  ✗ FATAL: Could not find free port. Exiting."
        exit 1
    fi
else
    echo "✓ Port $SERVER_PORT is available"
fi

# Define port file path (will write after validation)
SERVER_PORT_FILE="${HOSTNAME_FILE%.log}_server_port.txt"

# ----------------------------
# Verify port is available and clean up if needed
# ----------------------------
echo "Checking port availability: $SERVER_PORT"
if ! check_port_available $SERVER_PORT; then
    echo "  Port $SERVER_PORT is already in use"
    kill_port_process $SERVER_PORT
    sleep 2
    
    # Verify port is now free
    if ! check_port_available $SERVER_PORT; then
        echo "  ERROR: Port $SERVER_PORT still in use after cleanup attempt"
        echo "  Trying to find next available port..."
        
        # Try up to 10 sequential ports
        for offset in {1..10}; do
            CANDIDATE_PORT=$((SERVER_PORT + offset))
            echo "  Checking port $CANDIDATE_PORT..."
            if check_port_available $CANDIDATE_PORT; then
                echo "  Found available port: $CANDIDATE_PORT"
                SERVER_PORT=$CANDIDATE_PORT
                break
            fi
        done
        
        # Final check
        if ! check_port_available $SERVER_PORT; then
            echo "  FATAL: Could not find available port in range $SERVER_PORT-$((SERVER_PORT+10))"
            exit 1
        fi
    else
        echo "  Port $SERVER_PORT is now available"
    fi
else
    echo "  Port $SERVER_PORT is available"
fi

# ----------------------------
# Write final port to file after all validation and reassignment
# ----------------------------
echo "Writing server port '$SERVER_PORT' to file: $SERVER_PORT_FILE"
echo "$SERVER_PORT" > "$SERVER_PORT_FILE"

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
