#!/bin/bash
set -Eeuo pipefail

# PageIndex local-LLM PoC runner for PACE-ICE
# - Submits server.sh as a Slurm job
# - Waits for hostname + health readiness
# - Runs PageIndex tree generation against that server
# - Cancels server job on exit (unless KEEP_PAGEINDEX_POC_SERVER=true)

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found at ${REPO_ROOT}/.env"
  exit 1
fi

set -a
source .env
set +a

HOSTNAME_FILE="${HOSTNAME_LOG_FILE:-${REPO_ROOT}/hostname.log}"
SERVER_JOB_FILE="${HOSTNAME_FILE%.log}_server_job.txt"
PORT="${SERVER_PORT:-8000}"
WAIT_SECONDS="${PAGEINDEX_POC_SERVER_WAIT_SEC:-900}"
KEEP_SERVER="${KEEP_PAGEINDEX_POC_SERVER:-false}"

MD_PATH_DEFAULT="${REPO_ROOT}/PageIndex/tutorials/doc-search/README.md"
PDF_PATH_DEFAULT=""

MD_PATH="${1:-$MD_PATH_DEFAULT}"
PDF_PATH="${2:-$PDF_PATH_DEFAULT}"

if [[ -n "${PDF_PATH}" && -n "${MD_PATH}" ]]; then
  echo "ERROR: Provide only one input (md OR pdf)."
  echo "Usage: bash scripts/pageindex_poc_cluster.sh [optional_md_path] [optional_pdf_path]"
  exit 1
fi

if [[ -z "${MD_PATH}" && -z "${PDF_PATH}" ]]; then
  echo "ERROR: No input path provided."
  exit 1
fi

if [[ -n "${MD_PATH}" && ! -f "${MD_PATH}" ]]; then
  echo "ERROR: Markdown file not found: ${MD_PATH}"
  exit 1
fi

if [[ -n "${PDF_PATH}" && ! -f "${PDF_PATH}" ]]; then
  echo "ERROR: PDF file not found: ${PDF_PATH}"
  exit 1
fi

cleanup() {
  if [[ "${KEEP_SERVER}" == "true" ]]; then
    echo "KEEP_PAGEINDEX_POC_SERVER=true; leaving server job running."
    return
  fi

  local jid=""
  if [[ -f "${SERVER_JOB_FILE}" ]]; then
    jid="$(cat "${SERVER_JOB_FILE}" 2>/dev/null || true)"
  fi

  if [[ -n "${jid}" ]]; then
    echo "Stopping server job ${jid} ..."
    scancel "${jid}" || true
  fi
}
trap cleanup EXIT

mkdir -p metrics/slurm-results
rm -f "${HOSTNAME_FILE}" "${SERVER_JOB_FILE}"

echo "Submitting server job via server.sh ..."
SUBMIT_OUT="$(SYSTEM_PROMPT='' SERVER_PORT="${PORT}" HOSTNAME_LOG_FILE="${HOSTNAME_FILE}" sbatch server.sh)"
echo "${SUBMIT_OUT}"
SERVER_JOB_ID="$(echo "${SUBMIT_OUT}" | awk '{print $NF}')"

if [[ -z "${SERVER_JOB_ID}" ]]; then
  echo "ERROR: Could not parse server job ID from sbatch output."
  exit 1
fi

echo "Server job ID: ${SERVER_JOB_ID}"
echo "Waiting for hostname file: ${HOSTNAME_FILE}"

start_ts="$(date +%s)"
while [[ ! -f "${HOSTNAME_FILE}" ]]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed > WAIT_SECONDS )); then
    echo "ERROR: Timed out waiting for hostname file (${WAIT_SECONDS}s)."
    exit 1
  fi
  sleep 2
done

SERVER_HOST="$(cat "${HOSTNAME_FILE}")"
if [[ -z "${SERVER_HOST}" ]]; then
  echo "ERROR: Hostname file is empty."
  exit 1
fi

BASE_URL="http://${SERVER_HOST}:${PORT}"
GEN_URL="${BASE_URL}/generate"

echo "Server host: ${SERVER_HOST}"
echo "Probing health endpoint: ${BASE_URL}/"

start_ts="$(date +%s)"
until curl -fsS "${BASE_URL}/" >/dev/null 2>&1; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed > WAIT_SECONDS )); then
    echo "ERROR: Timed out waiting for server health endpoint (${WAIT_SECONDS}s)."
    exit 1
  fi
  sleep 3
done

echo "Server is healthy. Using PageIndex local endpoint: ${GEN_URL}"

if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export PAGEINDEX_LOCAL_SERVER_URL="${GEN_URL}"
export PAGEINDEX_LOCAL_SERVER_TIMEOUT="${PAGEINDEX_LOCAL_SERVER_TIMEOUT:-300}"
export PAGEINDEX_LOCAL_MAX_NEW_TOKENS="${PAGEINDEX_LOCAL_MAX_NEW_TOKENS:-8192}"
export PAGEINDEX_LOCAL_TOP_P="${PAGEINDEX_LOCAL_TOP_P:-0.15}"
export PAGEINDEX_LOCAL_TEMPERATURE="${PAGEINDEX_LOCAL_TEMPERATURE:-0.1}"

cd "${REPO_ROOT}/PageIndex"

if [[ -n "${MD_PATH}" ]]; then
  echo "Running PageIndex PoC on markdown: ${MD_PATH}"
  python run_pageindex.py \
    --md_path "${MD_PATH}" \
    --model "${PAGEINDEX_MODEL:-local_server}" \
    --if-add-node-summary yes \
    --if-add-doc-description no \
    --if-add-node-text no

  DOC_NAME="$(basename "${MD_PATH}" | sed 's/\.[^.]*$//')"
  OUT_FILE="${REPO_ROOT}/PageIndex/results/${DOC_NAME}_structure.json"
else
  echo "Running PageIndex PoC on pdf: ${PDF_PATH}"
  python run_pageindex.py \
    --pdf_path "${PDF_PATH}" \
    --model "${PAGEINDEX_MODEL:-local_server}" \
    --if-add-node-summary yes \
    --if-add-doc-description no \
    --if-add-node-text no

  DOC_NAME="$(basename "${PDF_PATH}" | sed 's/\.[^.]*$//')"
  OUT_FILE="${REPO_ROOT}/PageIndex/results/${DOC_NAME}_structure.json"
fi

if [[ ! -f "${OUT_FILE}" ]]; then
  echo "ERROR: Expected output tree not found: ${OUT_FILE}"
  exit 1
fi

if [[ ! -s "${OUT_FILE}" ]]; then
  echo "ERROR: Output tree file is empty: ${OUT_FILE}"
  exit 1
fi

echo "✅ PoC complete. Tree generated at: ${OUT_FILE}"
