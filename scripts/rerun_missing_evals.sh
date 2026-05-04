#!/bin/bash
# Re-submit eval jobs for genes that have model files but no results.
# Results are written to RUN_DIR/results/ and picked up by join_metrics.py.
#
# Usage: bash scripts/rerun_missing_evals.sh [RUN_ID]
#   RUN_ID defaults to my_run_20260428_011918

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-my_run_20260428_011918}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: Run directory not found: ${RUN_DIR}"
  exit 1
fi

# Load .env so VENV_PATH etc. are available for --export
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  echo "ERROR: .env not found in ${REPO_ROOT}"
  exit 1
fi
set -a; source "${REPO_ROOT}/.env"; set +a

MODELS_DIR="${REPO_ROOT}/sota/ExquisiteNetV2/models"
RESULTS_DIR="${RUN_DIR}/results"
mkdir -p "${RESULTS_DIR}"

GENES=(
  xXx0l4AwiVQhyGoytBdfRnRwJlC
  xXx60DrOmiQvRCnTqZ6J4OR1liP
  xXx7Ao1kEJGlOQny48xTg9fwSBJ
  xXxMAnWqnKd351IBOpWUvK17DRx
  xXxVFTFOvBK7etfmirqKm5ksftW
  xXxbJI6Z9pS9aXozpeAJmDGrA9m
  xXxqhKHU5XlOdgKBSuql1e7w5ip
)

for GENE in "${GENES[@]}"; do
  MODEL_FILE="${MODELS_DIR}/network_${GENE}.py"
  RESULT_FILE="${RESULTS_DIR}/${GENE}_results.txt"

  if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "SKIP ${GENE}: model file missing"
    continue
  fi

  if [[ -f "${RESULT_FILE}" ]]; then
    echo "SKIP ${GENE}: result already exists ($(cat ${RESULT_FILE}))"
    continue
  fi

  # Generate a per-gene eval script
  SCRIPT="/tmp/eval_${GENE}.sh"
  cat > "${SCRIPT}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=evalGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output=${REPO_ROOT}/slurm-results/eval-${GENE}-%j.out
#SBATCH --error=${REPO_ROOT}/slurm-results/eval-${GENE}-%j.err

echo "Evaluating gene: ${GENE}"
hostname
module load cuda

export LD_LIBRARY_PATH="${VENV_PATH}/lib/python3.13/site-packages/nvidia/nvjitlink/lib:\${LD_LIBRARY_PATH:-}"
source "${VENV_PATH}/bin/activate"

export RUN_DIR="${RUN_DIR}"
cd "${LLM_INFERENCE_ROOT_DIR:-${REPO_ROOT}}"

python ./sota/ExquisiteNetV2/train.py \
  -bs 216 \
  -network "models.network_${GENE}" \
  -data ./cifar10 \
  -end_lr 0.001 \
  -seed 21 \
  -val_r 0.2 \
  -amp
SBATCH

  JOB_ID=$(sbatch "${SCRIPT}" | awk '{print $4}')
  echo "SUBMITTED ${GENE} -> job ${JOB_ID}"
  rm "${SCRIPT}"
done

echo ""
echo "Done. Results will appear in: ${RESULTS_DIR}"
echo "Once complete, run:  python join_metrics.py"
