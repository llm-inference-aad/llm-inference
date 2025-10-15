#!/bin/bash

# Add Server to Existing Cluster
# Usage: ./add_server.sh [NUM_SERVERS]
#   NUM_SERVERS: Number of additional servers to add (default: 1)

set -e

NUM_SERVERS=${1:-1}

# Load environment variables
if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found in $(pwd)"
  exit 1
fi

set -a
source .env
set +a

export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$(pwd)}"
export SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE:-${LLM_INFERENCE_ROOT_DIR}/servers.json}"
export SERVER_BASE_PORT="${SERVER_BASE_PORT:-8000}"

echo "===== Adding $NUM_SERVERS Server(s) to Existing Cluster ====="
echo "Working directory: $LLM_INFERENCE_ROOT_DIR"
echo "Registry file: $SERVER_REGISTRY_FILE"
echo ""

# Check if registry file exists
if [[ ! -f "$SERVER_REGISTRY_FILE" ]]; then
    echo "ERROR: Server registry file not found: $SERVER_REGISTRY_FILE"
    echo "Make sure you have a running cluster first."
    exit 1
fi

# Check current server count
CURRENT_SERVERS=$(uv run python << EOF
import json
try:
    with open('$SERVER_REGISTRY_FILE', 'r') as f:
        registry = json.load(f)
    print(len(registry.get('servers', [])))
except:
    print(0)
EOF
)

echo "Current servers in cluster: $CURRENT_SERVERS"
echo ""

# Start additional servers
echo "Step 1: Starting $NUM_SERVERS additional server(s)..."
SERVER_JOB_IDS=()

for i in $(seq 1 $NUM_SERVERS); do
    echo "  Starting server $i/$NUM_SERVERS..."
    
    # Submit server job (port will be auto-assigned)
    SERVER_JOB_OUTPUT=$(SERVER_REGISTRY_FILE=$SERVER_REGISTRY_FILE sbatch server.sh)
    SERVER_JOB_ID=$(echo "$SERVER_JOB_OUTPUT" | awk '{print $NF}')
    SERVER_JOB_IDS+=("$SERVER_JOB_ID")
    
    echo "    Server job submitted: $SERVER_JOB_ID"
    
    # Small delay between submissions
    sleep 0.5
done

echo ""
echo "===== Additional Servers Started Successfully ====="
echo ""
echo "Job IDs:"
echo "  Additional Servers: ${SERVER_JOB_IDS[*]}"
echo ""
echo "Monitoring commands:"
echo "  Check job status:      squeue -u \$USER"
echo "  Monitor cluster:       ./monitor_cluster.sh"
echo "  View server registry:  cat $SERVER_REGISTRY_FILE | uv run python -m json.tool"
echo ""
echo "The servers will auto-assign ports starting from the next available port."
echo "Wait 2-3 minutes for model loading, then check status with ./monitor_cluster.sh"
echo ""
echo "To stop additional servers:"
echo "  scancel ${SERVER_JOB_IDS[*]}"
echo ""


