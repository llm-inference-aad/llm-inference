#!/bin/bash
# ==============================================================================
# run_experiments.sh — Master experiment runner for SmoothQuant comparison
#
# Submits all conditions automatically:
#   1. Baseline    (SMOOTHQUANT=0)  — N_BASELINE runs, batch_size=1
#   2. SmoothQuant (SMOOTHQUANT=1)  — N_SQ runs,       batch_size=1
#   3. Batch sweep (SMOOTHQUANT=0)  — batch sizes 1,2,4,8,16 (1 run each)
#   4. Batch sweep (SMOOTHQUANT=1)  — batch sizes 1,2,4,8,16 (1 run each)
#
# Usage:
#   bash run_experiments.sh              # run everything
#   bash run_experiments.sh baseline     # only baseline runs
#   bash run_experiments.sh smoothquant  # only SQ runs
#   bash run_experiments.sh batchsweep  # only batch sweep
#
# Each run:
#   - Starts an LLM server job (server.sh)
#   - Waits for the server to be up
#   - Submits the evolution job (run.sh) with a dependency on the server
#   - Logs the job IDs to experiment_log.txt
# ==============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ── Config ────────────────────────────────────────────────────────────────────
N_BASELINE=5          # number of baseline runs
N_SQ=5                # number of SmoothQuant runs
BATCH_SIZES=(1 2 4 8 16)
SQ_ALPHA=0.85         # migration strength for Llama-3.3

EXPERIMENT_LOG="${REPO_ROOT}/experiment_log.txt"
SLURM_LOG_DIR="${REPO_ROOT}/slurm-results"
mkdir -p "${SLURM_LOG_DIR}"

MODE="${1:-all}"  # all | baseline | smoothquant | batchsweep

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${EXPERIMENT_LOG}"
}

# Submit one server + one evolution job pair.
# Args: label  smoothquant(0|1)  batch_size  alpha
submit_pair() {
    local label="$1"
    local sq="$2"
    local bs="$3"
    local alpha="$4"

    # Create run directory
    RUN_ID=$(AUTOMATED_CALL=true bash scripts/create_run.sh "${label}")
    RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
    log "Created run: ${RUN_ID}  (SQ=${sq}, BS=${bs}, alpha=${alpha})"

    # Write per-run config override into run dir so it's self-documenting
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

    # Submit server job — export overrides on top of .env
    SERVER_JOB_ID=$(sbatch \
        --job-name="server_${label}" \
        --export="ALL,SMOOTHQUANT=${sq},SMOOTHQUANT_ALPHA=${alpha},BATCH_SIZE=${bs},BATCH_WAIT_TIME=0,RUN_ID=${RUN_ID}" \
        --output="${SLURM_LOG_DIR}/slurm-server-%j.out" \
        --error="${SLURM_LOG_DIR}/slurm-server-%j.err" \
        server.sh \
        | awk '{print $NF}')

    log "  Server job submitted: ${SERVER_JOB_ID}"

    # Submit evolution job — depends on server being running (not just submitted)
    # afterany lets it start once server is alive; run.sh will poll hostname.log
    MAIN_JOB_ID=$(sbatch \
        --job-name="run_${label}" \
        --dependency="after:${SERVER_JOB_ID}" \
        --export="ALL,SMOOTHQUANT=${sq},SMOOTHQUANT_ALPHA=${alpha},BATCH_SIZE=${bs},RUN_ID=${RUN_ID}" \
        --output="${SLURM_LOG_DIR}/slurm-main-%j.out" \
        --error="${SLURM_LOG_DIR}/slurm-main-%j.err" \
        run.sh \
        | awk '{print $NF}')

    log "  Evolution job submitted: ${MAIN_JOB_ID}  (depends on ${SERVER_JOB_ID})"
    log "  Run dir: ${RUN_DIR}"
    echo ""
}

# ── Banner ────────────────────────────────────────────────────────────────────
log "====== SmoothQuant Experiment Suite ======"
log "Mode: ${MODE}"
log "Baseline runs: ${N_BASELINE}"
log "SmoothQuant runs: ${N_SQ}"
log "Batch sizes (sweep): ${BATCH_SIZES[*]}"
log "Model: $(grep '^MODEL_PATH=' .env | head -1 | cut -d= -f2)"
log "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
log "=========================================="
echo ""

# ── 1. Baseline runs ──────────────────────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "baseline" ]]; then
    log "--- Submitting ${N_BASELINE} BASELINE runs (SQ=0, BS=1) ---"
    for i in $(seq 1 ${N_BASELINE}); do
        submit_pair "baseline_bs1_run${i}" 0 1 ${SQ_ALPHA}
        sleep 2  # small gap to avoid race on hostname.log
    done
fi

# ── 2. SmoothQuant runs ───────────────────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "smoothquant" ]]; then
    log "--- Submitting ${N_SQ} SMOOTHQUANT runs (SQ=1, BS=1, alpha=${SQ_ALPHA}) ---"
    for i in $(seq 1 ${N_SQ}); do
        submit_pair "smoothquant_bs1_run${i}" 1 1 ${SQ_ALPHA}
        sleep 2
    done
fi

# ── 3. Batch size sweep — Baseline ───────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "batchsweep" ]]; then
    log "--- Submitting BATCH SWEEP — Baseline (SQ=0) ---"
    for bs in "${BATCH_SIZES[@]}"; do
        # bs=1 already covered above, skip duplicate if running all
        if [[ "${MODE}" == "all" && "${bs}" == "1" ]]; then
            continue
        fi
        submit_pair "baseline_bs${bs}" 0 ${bs} ${SQ_ALPHA}
        sleep 2
    done

    log "--- Submitting BATCH SWEEP — SmoothQuant (SQ=1) ---"
    for bs in "${BATCH_SIZES[@]}"; do
        if [[ "${MODE}" == "all" && "${bs}" == "1" ]]; then
            continue
        fi
        submit_pair "smoothquant_bs${bs}" 1 ${bs} ${SQ_ALPHA}
        sleep 2
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
log "====== All jobs submitted ======"
log "Monitor with:  squeue -u ${USER}"
log "Watch queue:   watch -n 30 'squeue -u ${USER}'"
log "Full log:      cat ${EXPERIMENT_LOG}"
log ""
log "After runs complete:"
log "  python join_metrics.py          # aggregate results"
log "  bash run_dashboard.sh           # visualize in browser"
