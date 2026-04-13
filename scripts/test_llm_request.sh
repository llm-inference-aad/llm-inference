#!/bin/bash

set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/test_llm_request.sh [options]

Options:
  --prompt TEXT                     Prompt to send (default: probe prompt)
  --max-new-tokens N                max_new_tokens (default: 128)
  --temperature F                   temperature (default: 0.0)
  --top-p F                         top_p (default: 1.0)
  --timeout-seconds N               HTTP timeout seconds (default: 120)
  --api-url URL                     Full /generate URL override
  --root-url URL                    Root URL override for health check
  --expected-model-id ID            Expected model id; fail if mismatch/unverifiable
  --expected-response-substring T   Generated text must contain T
  --expected-response-exact T       Generated text must equal T exactly
  --output FILE                     Write probe log JSON to FILE
  -h, --help                        Show help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PROMPT="Reply with exactly: LLM_PROBE_OK"
MAX_NEW_TOKENS=128
TEMPERATURE=0.0
TOP_P=1.0
TIMEOUT_SECONDS=120
API_URL="${API_URL:-}"
ROOT_URL="${ROOT_URL:-}"
EXPECTED_MODEL_ID="${EXPECTED_MODEL_ID:-}"
EXPECTED_RESPONSE_SUBSTRING="${EXPECTED_RESPONSE_SUBSTRING:-}"
EXPECTED_RESPONSE_EXACT="${EXPECTED_RESPONSE_EXACT:-}"
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --api-url)
      API_URL="$2"
      shift 2
      ;;
    --root-url)
      ROOT_URL="$2"
      shift 2
      ;;
    --expected-model-id)
      EXPECTED_MODEL_ID="$2"
      shift 2
      ;;
    --expected-response-substring)
      EXPECTED_RESPONSE_SUBSTRING="$2"
      shift 2
      ;;
    --expected-response-exact)
      EXPECTED_RESPONSE_EXACT="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

RUN_ID="${RUN_ID:-server-only}"
LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-${REPO_ROOT}}"
RUN_DIR="${RUN_DIR:-${LLM_INFERENCE_ROOT_DIR}/runs/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${RUN_DIR}/logs}"
RUN_METRICS_DIR="${RUN_METRICS_DIR:-${RUN_DIR}/metrics}"
HOSTNAME_LOG_FILE="${HOSTNAME_LOG_FILE:-${RUN_LOG_DIR}/hostname.log}"
LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${RUN_LOG_DIR}/loadbalancer.log}"
SERVER_PORT="${SERVER_PORT:-8000}"
LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-9000}"
USE_LOAD_BALANCER="${USE_LOAD_BALANCER:-false}"
MODEL_ID="${MODEL_ID:-}"
MODEL_PATH="${MODEL_PATH:-}"

mkdir -p "${RUN_LOG_DIR}" "${RUN_METRICS_DIR}" "${RUN_LOG_DIR}/llm_probe"

if [[ -z "${API_URL}" ]]; then
  if [[ "${USE_LOAD_BALANCER,,}" =~ ^(1|true|yes)$ ]]; then
    TARGET_MODE="load_balancer"
    HOST_FILE="${LOADBALANCER_LOG_FILE}"
    TARGET_PORT="${LOAD_BALANCER_PORT}"
  else
    TARGET_MODE="single_server"
    HOST_FILE="${HOSTNAME_LOG_FILE}"
    TARGET_PORT="${SERVER_PORT}"
  fi

  if [[ ! -f "${HOST_FILE}" ]]; then
    echo "Host file not found: ${HOST_FILE}" >&2
    echo "Start server first (server.sh or start_cluster.sh)." >&2
    exit 1
  fi

  TARGET_HOST="$(tr -d '[:space:]' < "${HOST_FILE}")"
  if [[ -z "${TARGET_HOST}" ]]; then
    echo "Host file is empty: ${HOST_FILE}" >&2
    exit 1
  fi

  API_URL="http://${TARGET_HOST}:${TARGET_PORT}/generate"
else
  TARGET_MODE="manual_url"
  TARGET_HOST="manual"
  TARGET_PORT="manual"
fi

if [[ -z "${ROOT_URL}" ]]; then
  ROOT_URL="${API_URL%/generate}/"
fi

JOB_ID="probe_${RUN_ID}"
GENE_ID="probe_$(date +%s)"
REQUEST_TS="$(date -Iseconds)"

REQUEST_PAYLOAD="$(python3 - <<'PY' "${PROMPT}" "${MAX_NEW_TOKENS}" "${TEMPERATURE}" "${TOP_P}" "${JOB_ID}" "${GENE_ID}"
import json
import sys

prompt, max_new_tokens, temperature, top_p, job_id, gene_id = sys.argv[1:]
obj = {
    "prompt": prompt,
    "max_new_tokens": int(max_new_tokens),
    "temperature": float(temperature),
    "top_p": float(top_p),
    "job_id": job_id,
    "gene_id": gene_id,
}
print(json.dumps(obj, ensure_ascii=False))
PY
)"

HEALTH_RAW="$(curl -sS --max-time "${TIMEOUT_SECONDS}" \
  -w $'\nHTTP_STATUS:%{http_code}\nTOTAL_TIME:%{time_total}\n' \
  "${ROOT_URL}" || true)"
HEALTH_STATUS="$(printf "%s" "${HEALTH_RAW}" | awk -F: '/^HTTP_STATUS:/ {print $2}' | tail -n1 | tr -d '[:space:]')"
HEALTH_TOTAL_TIME_SECONDS="$(printf "%s" "${HEALTH_RAW}" | awk -F: '/^TOTAL_TIME:/ {print $2}' | tail -n1 | tr -d '[:space:]')"
HEALTH_BODY="$(printf "%s" "${HEALTH_RAW}" | sed '/^HTTP_STATUS:/,$d')"

if [[ "${HEALTH_STATUS}" != "200" ]]; then
  echo "Health check failed at ${ROOT_URL} (status=${HEALTH_STATUS:-none})" >&2
  echo "Health body: ${HEALTH_BODY}" >&2
  exit 1
fi

RAW_RESPONSE="$(curl -sS --max-time "${TIMEOUT_SECONDS}" \
  -H "Content-Type: application/json" \
  -d "${REQUEST_PAYLOAD}" \
  -w $'\nHTTP_STATUS:%{http_code}\nTOTAL_TIME:%{time_total}\n' \
  "${API_URL}")"

HTTP_STATUS="$(printf "%s" "${RAW_RESPONSE}" | awk -F: '/^HTTP_STATUS:/ {print $2}' | tail -n1 | tr -d '[:space:]')"
TOTAL_TIME_SECONDS="$(printf "%s" "${RAW_RESPONSE}" | awk -F: '/^TOTAL_TIME:/ {print $2}' | tail -n1 | tr -d '[:space:]')"
RESPONSE_BODY="$(printf "%s" "${RAW_RESPONSE}" | sed '/^HTTP_STATUS:/,$d')"

if [[ "${HTTP_STATUS}" != "200" ]]; then
  echo "Request failed with status ${HTTP_STATUS}" >&2
  echo "Response body:" >&2
  echo "${RESPONSE_BODY}" >&2
  exit 1
fi

RESPONSE_PARSED="$(python3 - <<'PY' "${RESPONSE_BODY}"
import json
import sys

body = sys.argv[1].strip()
if not body:
    raise SystemExit("Empty response body")
obj = json.loads(body)
print(json.dumps({
    "generated_text": obj.get("generated_text", ""),
    "run_hash": obj.get("run_hash"),
    "evaluationScore": obj.get("evaluationScore"),
    "prompt_tokens": obj.get("prompt_tokens"),
    "completion_tokens": obj.get("completion_tokens"),
    "total_tokens": obj.get("total_tokens"),
    "_latency_sec": obj.get("_latency_sec"),
    "response_time_sec": obj.get("response_time_sec"),
}, ensure_ascii=False))
PY
)"

GENERATED_TEXT="$(python3 - <<'PY' "${RESPONSE_PARSED}"
import json
import sys
print(json.loads(sys.argv[1]).get("generated_text") or "")
PY
)"

RUN_HASH="$(python3 - <<'PY' "${RESPONSE_PARSED}"
import json
import sys
print(json.loads(sys.argv[1]).get("run_hash") or "")
PY
)"

if [[ -z "${GENERATED_TEXT}" ]]; then
  echo "Generated text is empty." >&2
  exit 1
fi

if [[ -n "${EXPECTED_RESPONSE_EXACT}" && "${GENERATED_TEXT}" != "${EXPECTED_RESPONSE_EXACT}" ]]; then
  echo "Generated text did not match --expected-response-exact." >&2
  exit 1
fi

if [[ -n "${EXPECTED_RESPONSE_SUBSTRING}" && "${GENERATED_TEXT}" != *"${EXPECTED_RESPONSE_SUBSTRING}"* ]]; then
  echo "Generated text did not contain --expected-response-substring." >&2
  exit 1
fi

METRICS_FILE=""
if [[ -n "${RUN_HASH}" && -f "${RUN_METRICS_DIR}/latency-${RUN_HASH}.json" ]]; then
  METRICS_FILE="${RUN_METRICS_DIR}/latency-${RUN_HASH}.json"
else
  METRICS_FILE="$(ls -1t "${RUN_METRICS_DIR}"/latency-*.json 2>/dev/null | head -n1 || true)"
fi

MODEL_META_JSON="{}"
if [[ -n "${METRICS_FILE}" && -f "${METRICS_FILE}" ]]; then
  MODEL_META_JSON="$(python3 - <<'PY' "${METRICS_FILE}"
import json
import os
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

model_path = data.get("model_path")
model_id = data.get("model_id")
if not model_id and model_path:
    model_id = os.path.basename(model_path.rstrip("/"))

print(json.dumps({
    "metrics_file": path,
    "engine": data.get("engine"),
    "model_path": model_path,
    "model_id": model_id,
    "model_revision": data.get("model_revision"),
    "model_version": data.get("model_version"),
    "model_quantization": data.get("model_quantization"),
    "tensor_parallel_size": data.get("tensor_parallel_size"),
}, ensure_ascii=False))
PY
)"
fi

ACTUAL_MODEL_ID="$(python3 - <<'PY' "${MODEL_META_JSON}"
import json
import sys
obj = json.loads(sys.argv[1] or "{}")
print(obj.get("model_id") or "")
PY
)"

if [[ -z "${EXPECTED_MODEL_ID}" ]]; then
  EXPECTED_MODEL_ID="${MODEL_ID}"
fi
if [[ -z "${EXPECTED_MODEL_ID}" && -n "${MODEL_PATH}" ]]; then
  EXPECTED_MODEL_ID="$(basename "${MODEL_PATH}")"
fi

MODEL_MATCH="unknown"
if [[ -n "${EXPECTED_MODEL_ID}" ]]; then
  if [[ -n "${ACTUAL_MODEL_ID}" && "${EXPECTED_MODEL_ID}" == "${ACTUAL_MODEL_ID}" ]]; then
    MODEL_MATCH="true"
  else
    MODEL_MATCH="false"
  fi
fi

if [[ -z "${OUTPUT_FILE}" ]]; then
  OUTPUT_FILE="${RUN_LOG_DIR}/llm_probe/llm_probe_$(date +%Y%m%d_%H%M%S).json"
fi

python3 - <<'PY' "${OUTPUT_FILE}" "${REQUEST_TS}" "${TARGET_MODE}" "${TARGET_HOST}" "${TARGET_PORT}" \
  "${API_URL}" "${ROOT_URL}" "${HEALTH_STATUS}" "${HEALTH_TOTAL_TIME_SECONDS}" "${HEALTH_BODY}" \
  "${HTTP_STATUS}" "${TOTAL_TIME_SECONDS}" "${EXPECTED_MODEL_ID}" "${ACTUAL_MODEL_ID}" "${MODEL_MATCH}" \
  "${REQUEST_PAYLOAD}" "${RESPONSE_BODY}" "${RESPONSE_PARSED}" "${MODEL_META_JSON}" "${METRICS_FILE}" \
  "${EXPECTED_RESPONSE_SUBSTRING}" "${EXPECTED_RESPONSE_EXACT}"
import json
import sys

(
    output_file,
    request_ts,
    target_mode,
    target_host,
    target_port,
    api_url,
    root_url,
    health_status,
    health_total_time_seconds,
    health_body,
    http_status,
    total_time_seconds,
    expected_model_id,
    actual_model_id,
    model_match,
    request_payload_json,
    response_body,
    response_parsed_json,
    model_meta_json,
    metrics_file,
    expected_response_substring,
    expected_response_exact,
) = sys.argv[1:]

record = {
    "timestamp": request_ts,
    "target": {
        "mode": target_mode,
        "host": target_host,
        "port": target_port,
        "root_url": root_url,
        "api_url": api_url,
    },
    "health_check": {
        "http_status": int(health_status),
        "http_total_time_sec": float(health_total_time_seconds),
        "raw_body": health_body,
    },
    "request": json.loads(request_payload_json),
    "response": {
        "http_status": int(http_status),
        "http_total_time_sec": float(total_time_seconds),
        "raw_body": response_body,
        "parsed": json.loads(response_parsed_json),
    },
    "verification": {
        "expected_model_id": expected_model_id or None,
        "actual_model_id": actual_model_id or None,
        "model_match": model_match,
        "expected_response_substring": expected_response_substring or None,
        "expected_response_exact": expected_response_exact or None,
    },
    "server_metadata": json.loads(model_meta_json or "{}"),
    "metrics_file": metrics_file or None,
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

echo "LLM probe succeeded."
echo "API URL:          ${API_URL}"
echo "Health URL:       ${ROOT_URL}"
echo "HTTP total time:  ${TOTAL_TIME_SECONDS}s"
echo "Expected model:   ${EXPECTED_MODEL_ID:-<unset>}"
echo "Actual model:     ${ACTUAL_MODEL_ID:-<unknown>}"
echo "Model match:      ${MODEL_MATCH}"
echo "Probe log:        ${OUTPUT_FILE}"

if [[ "${MODEL_MATCH}" == "false" ]]; then
  echo "Model mismatch detected (or model could not be verified)." >&2
  exit 1
fi
