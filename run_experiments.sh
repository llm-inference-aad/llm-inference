#!/bin/bash
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
# ==============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
N_BASELINE=5
N_SQ=5
BATCH_SIZES=(2 4 8 16)   # BS=1 already covered by baseline/SQ runs above
SQ_ALPHA=0.85

EXPERIMENT_LOG="${REPO_ROOT}/experiment_log.txt"
SLURM_LOG_DIR="${REPO_ROOT}/slurm-results"
mkdir -p "${SLURM_LOG_DIR}"

MODE="${1:-all}"

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

    # Submit server job
    SERVER_JOB_ID=$(sbatch \
        --job-name="server_${label}" \
        --export="ALL,SMOOTHQUANT=${sq},SMOOTHQUANT_ALPHA=${alpha},BATCH_SIZE=${bs},BATCH_WAIT_TIME=0,RUN_ID=${RUN_ID}" \
        --output="${SLURM_LOG_DIR}/slurm-server-%j.out" \
        --error="${SLURM_LOG_DIR}/slurm-server-%j.err" \
        server.sh \
        | awk '{print $NF}')

    log "  Server job: ${SERVER_JOB_ID}"

    # Submit evolution job — starts once server is running
    MAIN_JOB_ID=$(sbatch \
        --job-name="run_${label}" \
        --dependency="after:${SERVER_JOB_ID}" \
        --export="ALL,SMOOTHQUANT=${sq},SMOOTHQUANT_ALPHA=${alpha},BATCH_SIZE=${bs},RUN_ID=${RUN_ID}" \
        --output="${SLURM_LOG_DIR}/slurm-main-%j.out" \
        --error="${SLURM_LOG_DIR}/slurm-main-%j.err" \
        run.sh \
        | awk '{print $NF}')

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
if [[ "${MODE}" == "all" || "${MODE}" == "baseline" ]]; then
    log "--- BASELINE runs (SQ=0, BS=1) ---"
    for i in $(seq 1 ${N_BASELINE}); do
        submit_and_wait "baseline_bs1_run${i}" 0 1 ${SQ_ALPHA}
    done
fi

# ── 2. SmoothQuant runs ───────────────────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "smoothquant" ]]; then
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
