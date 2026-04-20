#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 16:00:00                # Runtime in D-HH:MM
#SBATCH --mem-per-gpu=16G
#SBATCH -n 1                      # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
# Removed GPU constraint - ice-gpu partition uses different naming (a100, v100, etc.)
#SBATCH --output=metrics/slurm-results/slurm-main-%j.out
#SBATCH --error=metrics/slurm-results/slurm-main-%j.err

set -Eeuo pipefail

# ==============================================================================
# Determine Repository Root
# ==============================================================================
# Use SLURM_SUBMIT_DIR if running under SLURM, otherwise use script location
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Change to repository root to ensure all paths work correctly
cd "${REPO_ROOT}"

# ==============================================================================
# Automatic Run Directory Setup
# ==============================================================================
# If RUN_ID is not set, create a new run directory automatically
if [[ -z "${RUN_ID:-}" ]]; then
  echo "No RUN_ID provided. Creating new run directory..."
  RUN_ID=$(AUTOMATED_CALL=true bash scripts/create_run.sh "auto")
  echo "Created RUN_ID: ${RUN_ID}"
fi

# Set run-scoped paths (absolute paths avoid subprocess working directory issues)
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
RUN_LOG_DIR="${RUN_DIR}/logs"
RUN_METRICS_DIR="${RUN_DIR}/metrics"
RUN_ERRORS_DIR="${RUN_DIR}/errors"

# Validate run directory exists
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: Run directory ${RUN_DIR} does not exist!"
  echo "Expected path: ${RUN_DIR}"
  echo "REPO_ROOT: ${REPO_ROOT}"
  echo "RUN_ID: ${RUN_ID}"
  echo "Create it first with: bash scripts/create_run.sh [optional_name]"
  exit 1
fi

export RUN_ID
export RUN_DIR
export RUN_LOG_DIR
export RUN_METRICS_DIR
export RUN_ERRORS_DIR
mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${RUN_ERRORS_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/run-main-${SLURM_JOB_ID:-manual}.out") \
     2> >(tee -a "${RUN_ERRORS_DIR}/run-main-${SLURM_JOB_ID:-manual}.err" >&2)
echo "Using run directory: ${RUN_DIR}"
# ==============================================================================

echo "=== Launching LLM Guided Evolution ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
date

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${RUN_ERRORS_DIR}" metrics/slurm-results

# ----------------------------
# Load modules / CUDA / Python
# ----------------------------
module load cuda
module load anaconda3 || true     # Load Python 3.12.5 which is compatible with our requirements
export CUDA_VISIBLE_DEVICES=0

# ----------------------------
# Ensure uv on PATH
# ----------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH. Install with: pipx install uv  (or per your cluster setup)"
  exit 1
fi

echo "UV version: $(uv --version)"
echo "Python via uv: $(uv run python --version)"

	# ----------------------------
	# Load environment variables
	# ----------------------------
	if [[ ! -f .env ]]; then
	  echo "ERROR: .env file not found in $(pwd)"
	  exit 1
	fi

	# Preserve any experiment overrides passed via `sbatch --export` (or manually exported).
	# `.env` is treated as defaults; overrides should win.
	OVERRIDE_RAG_ENABLED="${RAG_ENABLED-}"
	OVERRIDE_RAG_USE_CODE_CONTEXT="${RAG_USE_CODE_CONTEXT-}"
	OVERRIDE_RAG_USE_TEXT_CONTEXT="${RAG_USE_TEXT_CONTEXT-}"
	OVERRIDE_RAG_RERANKER_ENABLED="${RAG_RERANKER_ENABLED-}"
	OVERRIDE_NUM_GENERATIONS="${NUM_GENERATIONS-}"
	OVERRIDE_POPULATION_SIZE="${POPULATION_SIZE-}"
	OVERRIDE_START_POPULATION_SIZE="${START_POPULATION_SIZE-}"
	OVERRIDE_EXPERIMENT_SEED="${EXPERIMENT_SEED-}"

	# Auto-export all keys defined in .env into this shell's environment
	set -a
	source .env
	set +a

	# Re-apply preserved overrides (if set).
	[[ -n "${OVERRIDE_RAG_ENABLED}" ]] && export RAG_ENABLED="${OVERRIDE_RAG_ENABLED}"
	[[ -n "${OVERRIDE_RAG_USE_CODE_CONTEXT}" ]] && export RAG_USE_CODE_CONTEXT="${OVERRIDE_RAG_USE_CODE_CONTEXT}"
	[[ -n "${OVERRIDE_RAG_USE_TEXT_CONTEXT}" ]] && export RAG_USE_TEXT_CONTEXT="${OVERRIDE_RAG_USE_TEXT_CONTEXT}"
	[[ -n "${OVERRIDE_RAG_RERANKER_ENABLED}" ]] && export RAG_RERANKER_ENABLED="${OVERRIDE_RAG_RERANKER_ENABLED}"
	[[ -n "${OVERRIDE_NUM_GENERATIONS}" ]] && export NUM_GENERATIONS="${OVERRIDE_NUM_GENERATIONS}"
	[[ -n "${OVERRIDE_POPULATION_SIZE}" ]] && export POPULATION_SIZE="${OVERRIDE_POPULATION_SIZE}"
	[[ -n "${OVERRIDE_START_POPULATION_SIZE}" ]] && export START_POPULATION_SIZE="${OVERRIDE_START_POPULATION_SIZE}"
	[[ -n "${OVERRIDE_EXPERIMENT_SEED}" ]] && export EXPERIMENT_SEED="${OVERRIDE_EXPERIMENT_SEED}"

# Quick masked sanity checks (show only prefixes)


: "${LLM_INFERENCE_ROOT_DIR:=/home/hice1/satmuri6/scratch/llm-inference}"
export LLM_INFERENCE_ROOT_DIR
echo "LLM_INFERENCE_ROOT_DIR: $LLM_INFERENCE_ROOT_DIR"

# ------------------------------------------------------------------------------
# Enforce run-scoped paths for all runtime logs and metrics
# ------------------------------------------------------------------------------
export RUN_LOG_DIR="${RUN_LOG_DIR}"
export RUN_METRICS_DIR="${RUN_METRICS_DIR}"
export RUN_ERRORS_DIR="${RUN_ERRORS_DIR}"
export SLURM_LOG_DIR="${RUN_LOG_DIR}"
export SLURM_ERROR_DIR="${RUN_ERRORS_DIR}"
export METRICS_PATH="${RUN_METRICS_DIR}"
export HOSTNAME_LOG_FILE="${RUN_LOG_DIR}/hostname.log"
export LOADBALANCER_LOG_FILE="${RUN_LOG_DIR}/loadbalancer.log"
export SERVER_REGISTRY_FILE="${RUN_LOG_DIR}/servers.json"

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${RUN_ERRORS_DIR}"
echo "RUN_LOG_DIR: ${RUN_LOG_DIR}"
echo "RUN_METRICS_DIR: ${RUN_METRICS_DIR}"
echo "RUN_ERRORS_DIR: ${RUN_ERRORS_DIR}"

# ----------------------------
# CUDA libs for uv environment (best-effort)
# ----------------------------
# Some clusters need nvjitlink visible to the Python env used by uv.
UV_SITE_PKGS="$(uv run python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export LD_LIBRARY_PATH="$UV_SITE_PKGS/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH updated for nvjitlink (best-effort)."

# ----------------------------
# Final info dump
# ----------------------------
echo "UV Python path: $(uv run which python)"
nvidia-smi || true

# =============================================================================
# cleanup_server — called by trap on EXIT/INT/TERM
#
# Responsible for:
#   1. Cancelling the nested server job (if tracking file exists).
#   2. Moving any SLURM logs that haven't already been relocated.
#   3. Updating run_metadata.json with terminal status ("completed" on clean
#      exit, "cancelled" when the process exits with a non-zero code or is
#      interrupted).
#
# The function inspects the exit code of the main Python process (captured in
# $EVOLUTION_EXIT_CODE) rather than $? inside the trap, because traps reset $?
# to their own return value.
#
# NOTE: the body of this function is mirrored in tests/test_run_teardown.sh so
# the teardown logic can be exercised without loading SLURM/CUDA modules.  Keep
# the two in sync when editing.
# =============================================================================
cleanup_server() {
  local exit_code="${EVOLUTION_EXIT_CODE:-1}"

  echo "=== cleanup_server: exit_code=${exit_code} ==="

  # The cleanup is ordered so that state-changing operations required by the
  # PR 0 acceptance criterion (status flip, tracking-file removal, server
  # scancel) run FIRST — before any sleep or log-moving.  SLURM SIGKILLs the
  # script after a short grace period when the main job is scancelled, so any
  # sleep in the critical path will cause the trap to be cut short.

  local server_job_file="${HOSTNAME_LOG_FILE%.log}_server_job.txt"
  local server_job_id=""

  if [[ -f "${server_job_file}" ]]; then
    server_job_id=$(cat "${server_job_file}" 2>/dev/null || true)
  fi

  # -- 1. Flip run_metadata.json.status (critical) -----------------------------
  if [[ -f "${RUN_DIR}/run_metadata.json" ]]; then
    local status
    if [[ "${exit_code}" -eq 0 ]]; then
      status="completed"
    else
      status="cancelled"
    fi

    python3 - <<PYEOF
import json, sys

try:
    with open('${RUN_DIR}/run_metadata.json', 'r') as fh:
        metadata = json.load(fh)
    metadata['status'] = '${status}'
    metadata['completed_at'] = '$(date -Iseconds)'
    metadata['slurm_job_id'] = '${SLURM_JOB_ID:-}'
    with open('${RUN_DIR}/run_metadata.json', 'w') as fh:
        json.dump(metadata, fh, indent=2)
    print(f"run_metadata.json updated: status={metadata['status']}")
except Exception as exc:
    print(f"WARNING: could not update run_metadata.json: {exc}", file=sys.stderr)
PYEOF
  fi

  # -- 2. Cancel the nested server job (critical) ------------------------------
  if [[ -n "${server_job_id}" && "${server_job_id}" != "null" ]]; then
    echo "Cancelling server job: ${server_job_id}"
    scancel "${server_job_id}" 2>/dev/null || \
      echo "Warning: scancel ${server_job_id} failed (may have already finished)"
  elif [[ ! -f "${server_job_file}" ]]; then
    echo "Warning: Server job tracking file not found: ${server_job_file}"
    echo "  Server may need to be shut down manually"
  else
    echo "Warning: No valid server job ID in ${server_job_file}"
  fi

  # -- 3. Remove tracking files so nothing is stranded (critical) --------------
  if [[ -f "${server_job_file}" ]]; then
    rm -f "${server_job_file}"
    rm -f "${HOSTNAME_LOG_FILE}"
    echo "Removed server tracking files"
  fi

  # -- 4. Best-effort log flush + migration (may be cut short by SIGKILL) ------
  if [[ -n "${server_job_id}" && "${server_job_id}" != "null" ]]; then
    # Give the server a brief moment to flush buffered stdout before SLURM
    # yanks everything.  Kept short so the trap completes inside the default
    # scancel grace window.
    sleep 3

    local legacy_out="${REPO_ROOT}/metrics/slurm-results/slurm-server-${server_job_id}.out"
    local legacy_err="${REPO_ROOT}/metrics/slurm-results/slurm-server-${server_job_id}.err"
    [[ -f "${legacy_out}" ]] && mv "${legacy_out}" "${RUN_LOG_DIR}/" && \
      echo "Moved legacy server .out log"
    [[ -f "${legacy_err}" ]] && mv "${legacy_err}" "${RUN_ERRORS_DIR}/" && \
      echo "Moved legacy server .err log"
  fi

  if [[ -d "${REPO_ROOT}/metrics/slurm-results" ]]; then
    mkdir -p "${RUN_LOG_DIR}" "${RUN_ERRORS_DIR}"

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
      local main_out="${REPO_ROOT}/metrics/slurm-results/slurm-main-${SLURM_JOB_ID}.out"
      local main_err="${REPO_ROOT}/metrics/slurm-results/slurm-main-${SLURM_JOB_ID}.err"
      [[ -f "${main_out}" ]] && mv "${main_out}" "${RUN_LOG_DIR}/"  && echo "Moved main job .out log"
      [[ -f "${main_err}" ]] && mv "${main_err}" "${RUN_ERRORS_DIR}/" && echo "Moved main job .err log"
    fi

    find "${REPO_ROOT}/metrics/slurm-results" -type f \
      \( -name "eval-*.out" -o -name "llm-*.out" -o -name "slurm-server-*.out" \) \
      -newer "${RUN_DIR}/run_metadata.json" -exec mv {} "${RUN_LOG_DIR}/" \; 2>/dev/null || true
    find "${REPO_ROOT}/metrics/slurm-results" -type f \
      \( -name "eval-*.err" -o -name "llm-*.err" -o -name "slurm-server-*.err" \) \
      -newer "${RUN_DIR}/run_metadata.json" -exec mv {} "${RUN_ERRORS_DIR}/" \; 2>/dev/null || true
  fi

  echo "=== cleanup_server complete ==="
}

# Install the trap BEFORE the nested sbatch so that any signal from this point
# forward (including SIGTERM from `scancel <main_job>`) triggers cleanup.
# EXIT fires even on clean return from the script, which is what we want.
trap cleanup_server EXIT INT TERM

# ----------------------------
# Run your job
# ----------------------------
echo "=== Launching LLM Server ==="
SERVER_SBATCH_ARGS=(--parsable)
SERVER_PARTITION="${SERVER_PARTITION:-}"
SERVER_CONSTRAINT="${SERVER_CONSTRAINT:-}"

if [[ -z "${SERVER_PARTITION}" && -n "${SLURM_JOB_ID:-}" ]]; then
  SERVER_PARTITION="$(squeue -h -j "${SLURM_JOB_ID}" -o "%P" | awk 'NR==1 {print $1}')"
fi

# Keep the nested server on the same queue family as the run job unless the
# caller explicitly overrides it. This avoids the server drifting to server.sh's
# baked-in ice-gpu default when the main run was launched elsewhere.
if [[ -n "${SERVER_PARTITION}" ]]; then
  SERVER_SBATCH_ARGS+=(-p "${SERVER_PARTITION}")
fi

# Broad NVIDIA placement avoids pinning to H100-only unless the caller asks for it.
if [[ -z "${SERVER_CONSTRAINT}" ]]; then
  SERVER_CONSTRAINT="nvidia-gpu"
fi
SERVER_SBATCH_ARGS+=(-C "${SERVER_CONSTRAINT}")

if [[ -n "${SERVER_GPUS_PER_NODE:-}" ]]; then
  SERVER_SBATCH_ARGS+=("--gpus-per-node=${SERVER_GPUS_PER_NODE}")
fi

SERVER_SBATCH_ARGS+=("--output=${RUN_LOG_DIR}/slurm-server-%j.out" "--error=${RUN_LOG_DIR}/slurm-server-%j.err")

echo "Server submission args: ${SERVER_SBATCH_ARGS[*]} server.sh"
SERVER_JOB_ID=$(sbatch "${SERVER_SBATCH_ARGS[@]}" server.sh)
echo "Server job submitted with ID: ${SERVER_JOB_ID}"

# Persist the server job ID so cleanup_server (the EXIT/INT/TERM trap) can
# scancel it even if the main process is SIGKILL'd before cleanup runs —
# and even if the main shell env is lost.  cleanup_server reads this file.
echo "${SERVER_JOB_ID}" > "${HOSTNAME_LOG_FILE%.log}_server_job.txt"

echo "=== Running: uv run python run_improved.py ${RUN_DIR}/checkpoints ==="
# EVOLUTION_EXIT_CODE is left UNSET until Python actually returns.  Seeding it
# to 0 here would make the trap mis-read an interruption (SIGTERM arriving
# mid-Python, before the `|| rc=$?` branch) as a clean completion.  Leaving it
# unset lets cleanup_server's ${EVOLUTION_EXIT_CODE:-1} default fall through to
# "cancelled" in the interrupted case.
_python_rc=0
uv run python run_improved.py "${RUN_DIR}/checkpoints" || _python_rc=$?
EVOLUTION_EXIT_CODE=$_python_rc
export EVOLUTION_EXIT_CODE

# If the evolution succeeded, the cleanup trap (EXIT) will set status=completed.
# If it failed or was interrupted, the trap sets status=cancelled.
# Either way, the trap handles teardown — there is no separate teardown block here.
echo "=== Job complete (exit_code=${EVOLUTION_EXIT_CODE}) ==="
date
