#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --mem=4G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --output=slurm-results/slurm-experiments-%j.out
#SBATCH --error=slurm-results/slurm-experiments-%j.err
# ==============================================================================
# run_experiments.sh — Master experiment runner for SmoothQuant comparison
#
# Submits jobs ONE PAIR AT A TIME (serial chain) to stay within QOS GPU quota.
# Each server+run pair must fully finish before the next pair is submitted.
#
# Usage:
#   bash run_experiments.sh              # run everything
#   bash run_experiments.sh baseline     # only baseline runs
#   bash run_experiments.sh smoothquant  # only SQ runs
#   bash run_experiments.sh batchsweep  # only batch sweep
#   bash run_experiments.sh quick       # one tiny baseline/SQ pair for quota-limited smoke testing
# ==============================================================================

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
N_BASELINE=${N_BASELINE:-5}
N_SQ=${N_SQ:-5}
if [[ -n "${BATCH_SIZES_OVERRIDE:-}" ]]; then
    read -r -a BATCH_SIZES <<< "${BATCH_SIZES_OVERRIDE}"
else
    BATCH_SIZES=(2 4 8 16)   # BS=1 already covered by baseline/SQ runs above
fi
SQ_ALPHA=0.85

EXPERIMENT_LOG="${REPO_ROOT}/experiment_log.txt"
SLURM_LOG_DIR="${REPO_ROOT}/slurm-results"
mkdir -p "${SLURM_LOG_DIR}"

MODE="${1:-all}"
EXTRA_RUN_EXPORTS="${EXTRA_RUN_EXPORTS:-}"
DEFAULT_RUN_EXPORTS=",LOCAL_SERVER_MAX_RETRIES=120,SERVER_READY_TIMEOUT=1800,SERVER_READY_INTERVAL=15"

if [[ "${MODE}" == "quick" ]]; then
    N_BASELINE=1
    N_SQ=1
    BATCH_SIZES=()
    EXTRA_RUN_EXPORTS="${EXTRA_RUN_EXPORTS},GE_NUM_GENERATIONS=1,GE_START_POPULATION_SIZE=4,GE_POPULATION_SIZE=4,GE_NUM_ELITES=2,GE_HOF_SIZE=2,LLM_GENERATION_MAX_RETRIES=1,LLM_JOB_COMPLETION_TIMEOUT=1200,LOCAL_SERVER_TIMEOUT=900,LOCAL_SERVER_MAX_NEW_TOKENS=1024"
fi
EXTRA_RUN_EXPORTS="${DEFAULT_RUN_EXPORTS}${EXTRA_RUN_EXPORTS}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${EXPERIMENT_LOG}"
}

# Submit one server + one evolution job pair, WAIT for both to finish,
# then return. This keeps only 1 server job alive at a time.
submit_and_wait() {
    local label="$1"
    local sq="$2"
    local bs="$3"
    local alpha="$4"

    RUN_ID=$(AUTOMATED_CALL=true bash scripts/create_run.sh "${label}")
    RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
    log "Created run: ${RUN_ID}  (SQ=${sq}, BS=${bs}, alpha=${alpha})"

    cat > "${RUN_DIR}/experiment_config.json" <<EOF
{
  "label": "${label}",
  "smoothquant_enabled": ${sq},
  "smoothquant_alpha": ${alpha},
  "batch_size": ${bs},
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "submitted_at": "$(date -Iseconds)"
}
EOF

    # Submit server job — retry up to 10 times (handles Kerberos expiry)
    local max_retries=10
    local retry_wait=300  # 5 minutes between retries
    SERVER_JOB_ID=""
    for attempt in $(seq 1 ${max_retries}); do
        SERVER_JOB_ID=$(sbatch \
            --job-name="server_${label}" \
            --export="ALL,SMOOTHQUANT=${sq},SMOOTHQUANT_ALPHA=${alpha},BATCH_SIZE=${bs},BATCH_WAIT_TIME=0,RUN_ID=${RUN_ID}${EXTRA_RUN_EXPORTS}" \
            --output="${SLURM_LOG_DIR}/slurm-server-%j.out" \
            --error="${SLURM_LOG_DIR}/slurm-server-%j.err" \
            server.sh 2>&1 | awk '{print $NF}')
        if [[ "${SERVER_JOB_ID}" =~ ^[0-9]+$ ]]; then
            break
        fi
        log "  sbatch failed (attempt ${attempt}/${max_retries}): ${SERVER_JOB_ID} — retrying in ${retry_wait}s"
        sleep ${retry_wait}
    done
    if [[ ! "${SERVER_JOB_ID}" =~ ^[0-9]+$ ]]; then
        log "  ERROR: could not submit server job after ${max_retries} attempts. Exiting."
        exit 1
    fi

    log "  Server job: ${SERVER_JOB_ID}"

    # Submit evolution job — retry same way
    MAIN_JOB_ID=""
    for attempt in $(seq 1 ${max_retries}); do
        MAIN_JOB_ID=$(sbatch \
            --job-name="run_${label}" \
            --dependency="after:${SERVER_JOB_ID}" \
            --export="ALL,SMOOTHQUANT=${sq},SMOOTHQUANT_ALPHA=${alpha},BATCH_SIZE=${bs},RUN_ID=${RUN_ID}${EXTRA_RUN_EXPORTS}" \
            --output="${SLURM_LOG_DIR}/slurm-main-%j.out" \
            --error="${SLURM_LOG_DIR}/slurm-main-%j.err" \
            run.sh 2>&1 | awk '{print $NF}')
        if [[ "${MAIN_JOB_ID}" =~ ^[0-9]+$ ]]; then
            break
        fi
        log "  sbatch failed (attempt ${attempt}/${max_retries}): ${MAIN_JOB_ID} — retrying in ${retry_wait}s"
        sleep ${retry_wait}
    done
    if [[ ! "${MAIN_JOB_ID}" =~ ^[0-9]+$ ]]; then
        log "  ERROR: could not submit evolution job after ${max_retries} attempts. Exiting."
        exit 1
    fi

    log "  Evolution job: ${MAIN_JOB_ID} (depends on ${SERVER_JOB_ID})"
    log "  Run dir: ${RUN_DIR}"

    # ── Wait for the evolution job to finish before submitting the next pair ──
    log "  Waiting for job ${MAIN_JOB_ID} to complete..."
    while true; do
        STATE=$(squeue -j "${MAIN_JOB_ID}" -h -o "%T" 2>/dev/null || echo "DONE")
        if [[ -z "${STATE}" || "${STATE}" == "DONE" ]]; then
            break
        fi
        log "    Job ${MAIN_JOB_ID} state: ${STATE} — sleeping 60s"
        sleep 60
    done

    log "  Run ${RUN_ID} finished."
    echo ""
}

# ── Banner ────────────────────────────────────────────────────────────────────
log "====== SmoothQuant Experiment Suite (serial mode) ======"
log "Mode: ${MODE} | Baseline: ${N_BASELINE} | SQ: ${N_SQ} | Batch sizes: ${BATCH_SIZES[*]}"
log "Model: $(grep '^MODEL_PATH=' .env | head -1 | cut -d= -f2)"
log "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
log "========================================================"
echo ""

# ── 1. Baseline runs ──────────────────────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "baseline" || "${MODE}" == "quick" ]]; then
    log "--- BASELINE runs (SQ=0, BS=1) ---"
    for i in $(seq 1 ${N_BASELINE}); do
        submit_and_wait "baseline_bs1_run${i}" 0 1 ${SQ_ALPHA}
    done
fi

# ── 2. SmoothQuant runs ───────────────────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "smoothquant" || "${MODE}" == "quick" ]]; then
    log "--- SMOOTHQUANT runs (SQ=1, BS=1, alpha=${SQ_ALPHA}) ---"
    for i in $(seq 1 ${N_SQ}); do
        submit_and_wait "smoothquant_bs1_run${i}" 1 1 ${SQ_ALPHA}
    done
fi

# ── 3. Batch sweep ────────────────────────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "batchsweep" ]]; then
    log "--- BATCH SWEEP Baseline (SQ=0) ---"
    for bs in "${BATCH_SIZES[@]}"; do
        submit_and_wait "baseline_bs${bs}" 0 ${bs} ${SQ_ALPHA}
    done

    log "--- BATCH SWEEP SmoothQuant (SQ=1) ---"
    for bs in "${BATCH_SIZES[@]}"; do
        submit_and_wait "smoothquant_bs${bs}" 1 ${bs} ${SQ_ALPHA}
    done
fi

log "====== All experiments complete ======"
log "Aggregate results:  python join_metrics.py"
log "Visualize:          bash run_dashboard.sh"
