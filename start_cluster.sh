#!/bin/bash

# Start LLM Inference Cluster with Load Balancer
# Usage: ./start_cluster.sh [NUM_SERVERS]
#   NUM_SERVERS: Number of inference servers to start (default: 3)

set -e
NUM_SERVERS=${1:-3}

# ----------------------------
# Configuration
# -------------------------
# Load environment variables
if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found in $(pwd)"
  exit 1
fi

set -a
source .env
set +a

export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$(pwd)}"
if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID=$(AUTOMATED_CALL=true bash scripts/create_run.sh "cluster")
fi
export RUN_ID
export RUN_DIR="${RUN_DIR:-${LLM_INFERENCE_ROOT_DIR}/runs/${RUN_ID}}"
export RUN_LOG_DIR="${RUN_LOG_DIR:-${RUN_DIR}/logs}"
export RUN_METRICS_DIR="${RUN_METRICS_DIR:-${RUN_DIR}/metrics}"
export RUN_ERRORS_DIR="${RUN_ERRORS_DIR:-${RUN_DIR}/errors}"
export SERVER_BASE_PORT="${SERVER_BASE_PORT:-8000}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-9000}"
export LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${RUN_LOG_DIR}/loadbalancer.log}"
export SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE:-${RUN_LOG_DIR}/servers.json}"

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${RUN_ERRORS_DIR}"

echo "===== Starting LLM Inference Cluster ====="
echo "Working directory: $LLM_INFERENCE_ROOT_DIR"
echo "RUN_ID: $RUN_ID"
echo "RUN_DIR: $RUN_DIR"
echo "RUN_LOG_DIR: $RUN_LOG_DIR"
echo "RUN_METRICS_DIR: $RUN_METRICS_DIR"
echo "RUN_ERRORS_DIR: $RUN_ERRORS_DIR"
echo "Number of servers: $NUM_SERVERS"
echo "Server base port: $SERVER_BASE_PORT"
echo "Load balancer port: $LOAD_BALANCER_PORT"
echo "Registry file: $SERVER_REGISTRY_FILE"
echo ""

# ----------------------------
# Clean up old files
# ----------------------------
echo "Cleaning up old registry and log files..."
rm -f "$LOADBALANCER_LOG_FILE"
echo '{"servers": []}' > "$SERVER_REGISTRY_FILE"
echo "Registry file initialized"
echo ""

# ----------------------------
# Start load balancer
# ----------------------------
echo "Step 1: Starting load balancer..."
LB_JOB_OUTPUT=$(sbatch \
  --output "${RUN_LOG_DIR}/slurm-load-balancer-%j.out" \
  --error "${RUN_ERRORS_DIR}/slurm-load-balancer-%j.err" \
  --export=ALL,RUN_ID="${RUN_ID}",RUN_DIR="${RUN_DIR}",RUN_LOG_DIR="${RUN_LOG_DIR}",RUN_METRICS_DIR="${RUN_METRICS_DIR}",RUN_ERRORS_DIR="${RUN_ERRORS_DIR}",SLURM_LOG_DIR="${RUN_LOG_DIR}",SLURM_ERROR_DIR="${RUN_ERRORS_DIR}",LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE}",SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE}" \
  load_balancer.sh)
LB_JOB_ID=$(echo "$LB_JOB_OUTPUT" | awk '{print $NF}')
echo "  Load balancer job submitted: $LB_JOB_ID"
echo ""

# ----------------------------
# Wait for load balancer hostname
# ----------------------------
echo "Step 2: Waiting for load balancer to start..."
MAX_WAIT=120  # 2 minutes
WAIT_COUNT=0
while [[ ! -f "$LOADBALANCER_LOG_FILE" ]] && [[ $WAIT_COUNT -lt $MAX_WAIT ]]; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [[ $((WAIT_COUNT % 10)) -eq 0 ]]; then
        echo "  Still waiting... (${WAIT_COUNT}s)"
    fi
done

if [[ ! -f "$LOADBALANCER_LOG_FILE" ]]; then
    echo "ERROR: Load balancer did not start within $MAX_WAIT seconds"
    echo "Check the job output: squeue -j $LB_JOB_ID"
    exit 1
fi

LOADBALANCER_HOSTNAME=$(cat "$LOADBALANCER_LOG_FILE")
echo "  Load balancer started on: $LOADBALANCER_HOSTNAME:$LOAD_BALANCER_PORT"
echo ""

# ----------------------------
# Start inference servers
# ----------------------------
echo "Step 3: Starting $NUM_SERVERS inference servers..."
SERVER_JOB_IDS=()

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    SERVER_PORT=$((SERVER_BASE_PORT + i))
    
    echo "  Starting server $((i + 1))/$NUM_SERVERS on port $SERVER_PORT..."
    
    # Submit server job with custom port and registry file
    SERVER_JOB_OUTPUT=$(sbatch \
      --output "${RUN_LOG_DIR}/slurm-server-%j.out" \
      --error "${RUN_ERRORS_DIR}/slurm-server-%j.err" \
      --export=ALL,RUN_ID="${RUN_ID}",RUN_DIR="${RUN_DIR}",RUN_LOG_DIR="${RUN_LOG_DIR}",RUN_METRICS_DIR="${RUN_METRICS_DIR}",RUN_ERRORS_DIR="${RUN_ERRORS_DIR}",SLURM_LOG_DIR="${RUN_LOG_DIR}",SLURM_ERROR_DIR="${RUN_ERRORS_DIR}",SERVER_PORT="${SERVER_PORT}",SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE}",HOSTNAME_LOG_FILE="${RUN_LOG_DIR}/hostname-${SERVER_PORT}.log" \
      server.sh)
    SERVER_JOB_ID=$(echo "$SERVER_JOB_OUTPUT" | awk '{print $NF}')
    SERVER_JOB_IDS+=("$SERVER_JOB_ID")
    
    echo "    Server job submitted: $SERVER_JOB_ID (port $SERVER_PORT)"
    
    # Small delay between submissions
    sleep 0.5
done

echo ""
echo "===== Cluster Started Successfully ====="
echo ""
echo "Job IDs:"
echo "  Load Balancer: $LB_JOB_ID"
echo "  Servers: ${SERVER_JOB_IDS[*]}"
echo ""
echo "Load Balancer URL: http://$LOADBALANCER_HOSTNAME:$LOAD_BALANCER_PORT"
echo ""
echo "Monitoring commands:"
echo "  Check job status:      squeue -u \$USER"
echo "  Check server pool:     curl http://$LOADBALANCER_HOSTNAME:$LOAD_BALANCER_PORT/servers"
echo "  View load balancer log: tail -f ${RUN_LOG_DIR}/slurm-load-balancer-$LB_JOB_ID.out"
echo "  View load balancer err: tail -f ${RUN_ERRORS_DIR}/slurm-load-balancer-$LB_JOB_ID.err"
echo "  View server registry:  cat $SERVER_REGISTRY_FILE"
echo ""
echo "To submit client jobs (make sure USE_LOAD_BALANCER=true in .env):"
echo "  sbatch run.sh"
echo ""
echo "To stop the cluster:"
echo "  scancel $LB_JOB_ID ${SERVER_JOB_IDS[*]}"
echo ""
