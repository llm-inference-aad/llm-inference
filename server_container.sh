#!/bin/bash
# Container-aware server launch wrapper for Slurm
# Attempts to run the server inside an Apptainer/Singularity/Docker image
# Fallbacks to conda env `llm-vllm` or existing virtualenv when container is unavailable.

# SBATCH directives may be overridden on the sbatch command line
# Example: sbatch -p ice-gpu -C H100 --gpus-per-node=1 server_container.sh
# Defaults mirror server.sh where practical
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 08:00:00
#SBATCH -C "H100"
#SBATCH --gpus-per-node=1
#SBATCH -p ice-gpu
#SBATCH --mem 80G
#SBATCH -c 8
#SBATCH --output=metrics/slurm-results/slurm-server-%j.out
#SBATCH --error=metrics/slurm-results/slurm-server-%j.err

set -euo pipefail
echo "launching LLM Server (container-aware wrapper)"
hostname

mkdir -p metrics/slurm-results

# try to load modules if available (best-effort)
module load cuda 2>/dev/null || true

# Load environment variables from .env if present
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a
fi

export LLM_INFERENCE_ROOT_DIR=${LLM_INFERENCE_ROOT_DIR:-$(pwd)}
export RUN_ID=${RUN_ID:-server-only}
export RUN_DIR=${RUN_DIR:-${LLM_INFERENCE_ROOT_DIR}/runs/${RUN_ID}}
export RUN_LOG_DIR=${RUN_LOG_DIR:-${RUN_DIR}/logs}
export RUN_METRICS_DIR=${RUN_METRICS_DIR:-${RUN_DIR}/metrics}
export RUN_ERRORS_DIR=${RUN_ERRORS_DIR:-${RUN_DIR}/errors}
export SERVER_HOST=${SERVER_HOST:-0.0.0.0}
export SERVER_PORT=${SERVER_PORT:-}
export SERVER_WORKERS=${SERVER_WORKERS:-1}
export SERVER_REGISTRY_FILE=${SERVER_REGISTRY_FILE:-"${RUN_LOG_DIR}/servers.json"}

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${RUN_ERRORS_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/server-runtime-${SLURM_JOB_ID:-manual}.out") \
     2> >(tee -a "${RUN_ERRORS_DIR}/server-runtime-${SLURM_JOB_ID:-manual}.err" >&2)

echo "RUN_ID: ${RUN_ID}"
echo "RUN_LOG_DIR: ${RUN_LOG_DIR}"

export SERVER_HOSTNAME=$(hostname)
echo "$SERVER_HOSTNAME" > "${RUN_LOG_DIR}/hostname.log"
echo "${SLURM_JOB_ID}" > "${RUN_LOG_DIR}/hostname_server_job.txt"

# determine port if not provided (reuse logic from server.sh)
if [ -z "${SERVER_PORT}" ]; then
    SERVER_BASE_PORT=${SERVER_BASE_PORT:-8000}
    if [ -f "${SERVER_REGISTRY_FILE}" ]; then
        HIGHEST_PORT=$(python - <<'PY'
import json,os
reg=os.environ.get('SERVER_REGISTRY_FILE')
base=int(os.environ.get('SERVER_BASE_PORT'))
try:
    with open(reg) as f:
        r=json.load(f)
    ports=[int(s.get('port',base)) for s in r.get('servers',[])]
    print(max(ports)+1 if ports else base)
except Exception:
    print(base)
PY
)
        SERVER_PORT=$HIGHEST_PORT
    else
        SERVER_PORT=$SERVER_BASE_PORT
    fi
fi
echo "Using SERVER_PORT=$SERVER_PORT"

# Register server in registry
if [ ! -z "${SERVER_REGISTRY_FILE}" ]; then
    TIMESTAMP=$(date -Iseconds)
    TMP=$(mktemp)
    echo "{\"hostname\": \"${SERVER_HOSTNAME}\", \"port\": ${SERVER_PORT}, \"registered_at\": \"${TIMESTAMP}\"}" > "$TMP"
    (
        flock -x 200
        if [ -f "${SERVER_REGISTRY_FILE}" ]; then
            EXISTING=$(cat "${SERVER_REGISTRY_FILE}")
        else
            EXISTING='{"servers": []}'
        fi
        python - <<'PY'
import json,os
regfile=os.environ['SERVER_REGISTRY_FILE']
tmp=os.environ['TMP_ENTRY']
existing=json.loads(os.environ['EXISTING'])
with open(tmp) as f:
    new=json.load(f)
servs=existing.get('servers',[])
servs.append(new)
existing['servers']=servs
with open(regfile,'w') as f:
    json.dump(existing,f,indent=2)
print('Registered', new['hostname'], new['port'])
PY
    ) 200>"${SERVER_REGISTRY_FILE}.lock"
    rm -f "$TMP"
fi

# Container detection and command builder
VLLM_SIF=${VLLM_SIF:-"$(pwd)/vllm.sif"}
CONTAINER_CMD=""
if command -v apptainer >/dev/null 2>&1 && [ -f "$VLLM_SIF" ]; then
    echo "Using apptainer image: $VLLM_SIF"
    CONTAINER_CMD="apptainer exec --nv --bind $(pwd):/workspace $VLLM_SIF bash -lc 'cd /workspace; '"
elif command -v singularity >/dev/null 2>&1 && [ -f "$VLLM_SIF" ]; then
    echo "Using singularity image: $VLLM_SIF"
    CONTAINER_CMD="singularity exec --nv --bind $(pwd):/workspace $VLLM_SIF bash -lc 'cd /workspace; '
elif command -v docker >/dev/null 2>&1; then
    echo "Docker available; will attempt to run vllm docker image if accessible"
    CONTAINER_CMD="docker run --rm --gpus all -v $(pwd):/workspace -w /workspace vllm/vllm:latest bash -lc 'cd /workspace; '
fi

# If container command built and image present, run server inside it
if [ ! -z "$CONTAINER_CMD" ]; then
    echo "Launching server inside container"
    if [ "${VLLM_BACKEND:-true}" = "true" ]; then
        CON_CMD="$CONTAINER_CMD python -m uvicorn server_vllm:app --host ${SERVER_HOST} --port ${SERVER_PORT} --workers 1'"
    else
        CON_CMD="$CONTAINER_CMD python -m uvicorn server:app --host ${SERVER_HOST} --port ${SERVER_PORT} --workers ${SERVER_WORKERS}'"
    fi
    echo "Container command: $CON_CMD"
    eval $CON_CMD
    exit $?
fi

# Fallback: try to use conda env llm-vllm, then virtualenv, then system python
PYTHON_EXEC=""
if command -v conda >/dev/null 2>&1 && conda env list | grep -q '^llm-vllm'; then
    echo "Activating conda env llm-vllm"
    eval "$(conda shell.bash hook)"
    conda activate llm-vllm || true
    PYTHON_EXEC=$(command -v python || true)
fi

if [ -z "$PYTHON_EXEC" ]; then
    if [ -x "${VENV_PATH:-$(pwd)/.venv}/bin/python" ]; then
        PYTHON_EXEC="${VENV_PATH:-$(pwd)/.venv}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_EXEC=$(command -v python3)
    else
        PYTHON_EXEC=python
    fi
fi

echo "Using python executable: $PYTHON_EXEC"

if [ "${VLLM_BACKEND:-true}" = "true" ]; then
    echo "Starting vLLM backend (fallback non-container)"
    exec $PYTHON_EXEC -m uvicorn server_vllm:app --host ${SERVER_HOST} --port ${SERVER_PORT} --workers 1
else
    echo "Starting legacy HuggingFace backend (fallback non-container)"
    exec $PYTHON_EXEC -m uvicorn server:app --host ${SERVER_HOST} --port ${SERVER_PORT} --workers ${SERVER_WORKERS}
fi
