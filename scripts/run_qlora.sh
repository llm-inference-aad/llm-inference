#!/bin/bash
# QLoRA fine-tuning SLURM job — supports consecutive jobs for multi-slot training.
#
# PACE has a 16-hour wall limit.  This script:
#   - runs for up to 15h59m
#   - saves a checkpoint ~30 min before the deadline (via TimeoutCallback)
#   - on the next submission with the same QLORA_RUN_ID, resumes automatically
#
# Usage:
#   # First job (fresh start):
#   QLORA_RUN_ID=run_v1 sbatch scripts/run_qlora.sh
#
#   # Subsequent jobs (same run-dir, auto-resume from checkpoint):
#   QLORA_RUN_ID=run_v1 sbatch scripts/run_qlora.sh
#
#   # Local test (no SLURM):
#   bash scripts/run_qlora.sh

#SBATCH --job-name=qlora_finetune
#SBATCH -t 15:59:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100"
#SBATCH --mem-per-gpu=40G
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --output=slurm-results/qlora-%j.out
#SBATCH --error=slurm-results/qlora-%j.err

set -Eeuo pipefail

# ── Repo root ─────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

# ── Run directory ─────────────────────────────────────────────────────────────
# Keep QLORA_RUN_ID constant across consecutive jobs to share the same run dir.
QLORA_RUN_ID="${QLORA_RUN_ID:-qlora_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${REPO_ROOT}/runs/${QLORA_RUN_ID}"
CKPT_DIR="${RUN_DIR}/qlora/checkpoints"

mkdir -p \
  "${CKPT_DIR}" \
  "${RUN_DIR}/qlora/logs" \
  "${RUN_DIR}/qlora/metrics" \
  "${RUN_DIR}/qlora/adapters" \
  "${REPO_ROOT}/slurm-results"

export RUN_ID="${QLORA_RUN_ID}"
export RUN_DIR

echo "=== QLoRA Fine-Tuning ==="
echo "RUN_ID:   ${RUN_ID}"
echo "RUN_DIR:  ${RUN_DIR}"
echo "Hostname: $(hostname)"
echo "Date:     $(date)"

# ── Modules ───────────────────────────────────────────────────────────────────
module load cuda
module load anaconda3 || true

# ── Environment ───────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found in $(pwd)"
  exit 1
fi
set -a; source .env; set +a

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install with: pipx install uv"
  exit 1
fi

# ── CUDA library path for uv venv ─────────────────────────────────────────────
UV_SITE_PKGS="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${UV_SITE_PKGS}/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "UV: $(uv --version)"
echo "Python: $(uv run python --version)"
nvidia-smi || true

# ── Checkpoint detection ──────────────────────────────────────────────────────
# Find the latest checkpoint so we can resume if this is a continuation job.
LATEST_CKPT=""
if [[ -d "${CKPT_DIR}" ]]; then
  LATEST_CKPT=$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null \
    | sort -t- -k2 -n | tail -1 || true)
fi

if [[ -n "${LATEST_CKPT}" ]]; then
  echo "=== Resuming from checkpoint: ${LATEST_CKPT} ==="
  RESUME_FLAG="--resume-from-checkpoint ${LATEST_CKPT}"
else
  echo "=== No checkpoint found — starting fresh ==="
  RESUME_FLAG=""
fi

# ── Train ─────────────────────────────────────────────────────────────────────
echo "=== Starting QLoRA training ==="
# shellcheck disable=SC2086
uv run python scripts/train_qlora.py \
  --config configs/qlora.yaml \
  ${RESUME_FLAG}

EXIT_CODE=$?
echo "=== Training exited with code ${EXIT_CODE} ==="

# ── Status check ──────────────────────────────────────────────────────────────
STATUS_FILE="${RUN_DIR}/qlora/status.json"
if [[ -f "${STATUS_FILE}" ]]; then
  STATUS=$(python3 -c "import json,sys; d=json.load(open('${STATUS_FILE}')); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")
  echo "Training status: ${STATUS}"
  if [[ "${STATUS}" == "timeout" ]]; then
    echo ""
    echo "*** Job hit the time budget. To continue training, re-submit with: ***"
    echo "    QLORA_RUN_ID=${QLORA_RUN_ID} sbatch scripts/run_qlora.sh"
    echo ""
  elif [[ "${STATUS}" == "completed" ]]; then
    echo "Training completed successfully."
    echo "Adapter saved to: ${RUN_DIR}/qlora/adapters/"
    echo ""
    echo "To serve the fine-tuned model, set in your .env:"
    echo "    ADAPTER_PATH=${RUN_DIR}/qlora/adapters"
    echo "and restart the server."
  fi
fi

# ── Move SLURM logs into run dir ──────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  sleep 3  # let buffers flush
  OUT="${REPO_ROOT}/slurm-results/qlora-${SLURM_JOB_ID}.out"
  ERR="${REPO_ROOT}/slurm-results/qlora-${SLURM_JOB_ID}.err"
  [[ -f "${OUT}" ]] && mv "${OUT}" "${RUN_DIR}/qlora/logs/" && echo "Moved .out log"
  [[ -f "${ERR}" ]] && mv "${ERR}" "${RUN_DIR}/qlora/logs/" && echo "Moved .err log"
fi

echo "=== Done: $(date) ==="
exit ${EXIT_CODE}
