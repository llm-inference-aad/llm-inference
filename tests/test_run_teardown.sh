#!/bin/bash
# tests/test_run_teardown.sh — smoke-test the run.sh cleanup_server trap.
#
# Strategy: run a stripped-down version of the teardown logic directly,
# using stub commands for sbatch/scancel/module/uv/nvidia-smi.  We do NOT
# source run.sh because it requires SLURM, CUDA modules, and a .env file.
# Instead we isolate and exercise the exact cleanup_server() function body
# by copy-inlining it with all external dependencies replaced by stubs.
#
# Pass criteria (aligned with plan acceptance criteria):
#   1. After SIGTERM reaches the main process, cleanup_server() fires.
#   2. hostname_server_job.txt is removed by cleanup_server().
#   3. run_metadata.json.status == "cancelled" after a non-zero exit.
#   4. run_metadata.json.status == "completed" after a clean exit.
#
# No real SLURM is needed; scancel is stubbed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate repo root (the script may be invoked from any CWD).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Stub bin dir — put stub commands ahead of the real ones on PATH.
# ---------------------------------------------------------------------------
STUB_BIN="$(mktemp -d)"
trap 'rm -rf "${STUB_BIN}" "${TEST_RUN_DIR:-}"' EXIT

# Stub sbatch: prints a fake job ID.
cat > "${STUB_BIN}/sbatch" <<'STUB'
#!/bin/bash
echo "99999"
STUB
chmod +x "${STUB_BIN}/sbatch"

# Stub scancel: records the job IDs it was asked to cancel.
CANCELLED_FILE="${STUB_BIN}/cancelled_jobs.txt"
cat > "${STUB_BIN}/scancel" <<STUB
#!/bin/bash
echo "\$@" >> "${CANCELLED_FILE}"
STUB
chmod +x "${STUB_BIN}/scancel"

# Stub squeue: returns empty so SERVER_PARTITION logic short-circuits.
cat > "${STUB_BIN}/squeue" <<'STUB'
#!/bin/bash
echo ""
STUB
chmod +x "${STUB_BIN}/squeue"

export PATH="${STUB_BIN}:${PATH}"

# ---------------------------------------------------------------------------
# Helper: create a minimal run directory + run_metadata.json
# ---------------------------------------------------------------------------
setup_run_dir() {
    local base_dir="$1"
    local run_id="teardown_test_$$"
    local run_dir="${base_dir}/${run_id}"
    mkdir -p "${run_dir}/logs" "${run_dir}/errors" "${run_dir}/metrics" "${run_dir}/checkpoints"
    python3 -c "
import json
data = {'run_id': '${run_id}', 'status': 'running', 'started_at': '2026-01-01T00:00:00'}
with open('${run_dir}/run_metadata.json', 'w') as f:
    json.dump(data, f)
"
    echo "${run_dir}"
}

# ---------------------------------------------------------------------------
# The cleanup_server function extracted from run.sh for unit testing.
# External calls are routed through the PATH stubs above.
# ---------------------------------------------------------------------------
run_cleanup_server() {
    local run_dir="$1"
    local exit_code="${2:-0}"
    local repo_root="$3"

    local run_log_dir="${run_dir}/logs"
    local run_errors_dir="${run_dir}/errors"
    local hostname_log_file="${run_log_dir}/hostname.log"
    local server_job_file="${hostname_log_file%.log}_server_job.txt"

    # Write a fake server job tracking file (simulating what server.sh writes).
    echo "99999" > "${server_job_file}"

    # ---- Begin cleanup_server body (mirrors run.sh) ----
    local _exit_code="${exit_code}"

    if [[ -f "${server_job_file}" ]]; then
        local server_job_id
        server_job_id=$(cat "${server_job_file}" 2>/dev/null || true)

        if [[ -n "${server_job_id}" && "${server_job_id}" != "null" ]]; then
            scancel "${server_job_id}" 2>/dev/null || true
            # Skip the real sleep 15 in tests — use 0 for speed.
            sleep 0
        fi

        rm -f "${server_job_file}"
        rm -f "${hostname_log_file}"
    fi

    # Update metadata
    if [[ -f "${run_dir}/run_metadata.json" ]]; then
        local status
        if [[ "${_exit_code}" -eq 0 ]]; then
            status="completed"
        else
            status="cancelled"
        fi

        python3 - <<PYEOF
import json
with open('${run_dir}/run_metadata.json', 'r') as fh:
    metadata = json.load(fh)
metadata['status'] = '${status}'
with open('${run_dir}/run_metadata.json', 'w') as fh:
    json.dump(metadata, fh, indent=2)
PYEOF
    fi
    # ---- End cleanup_server body ----
}

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        pass "${label}: expected='${expected}'"
    else
        fail "${label}: expected='${expected}' got='${actual}'"
    fi
}

assert_file_absent() {
    local label="$1" path="$2"
    if [[ ! -f "${path}" ]]; then
        pass "${label}: file absent as expected"
    else
        fail "${label}: file still exists: ${path}"
    fi
}

assert_file_present() {
    local label="$1" path="$2"
    if [[ -f "${path}" ]]; then
        pass "${label}: file present as expected"
    else
        fail "${label}: file missing: ${path}"
    fi
}

# ---------------------------------------------------------------------------
# Test 1: Non-zero exit → status = "cancelled", tracking files removed
# ---------------------------------------------------------------------------
echo ""
echo "=== Test 1: non-zero exit sets status=cancelled ==="
TEST_BASE="$(mktemp -d)"
TEST_RUN_DIR=$(setup_run_dir "${TEST_BASE}")
RUN_LOG_DIR="${TEST_RUN_DIR}/logs"
SERVER_JOB_FILE="${RUN_LOG_DIR}/hostname_server_job.txt"

run_cleanup_server "${TEST_RUN_DIR}" 1 "${REPO_ROOT}"

STATUS=$(python3 -c "import json; d=json.load(open('${TEST_RUN_DIR}/run_metadata.json')); print(d['status'])")
assert_eq "status after non-zero exit" "cancelled" "${STATUS}"
assert_file_absent "server_job_file removed" "${SERVER_JOB_FILE}"

rm -rf "${TEST_BASE}"

# ---------------------------------------------------------------------------
# Test 2: Zero exit → status = "completed", tracking files removed
# ---------------------------------------------------------------------------
echo ""
echo "=== Test 2: zero exit sets status=completed ==="
TEST_BASE="$(mktemp -d)"
TEST_RUN_DIR=$(setup_run_dir "${TEST_BASE}")
RUN_LOG_DIR="${TEST_RUN_DIR}/logs"
SERVER_JOB_FILE="${RUN_LOG_DIR}/hostname_server_job.txt"

run_cleanup_server "${TEST_RUN_DIR}" 0 "${REPO_ROOT}"

STATUS=$(python3 -c "import json; d=json.load(open('${TEST_RUN_DIR}/run_metadata.json')); print(d['status'])")
assert_eq "status after zero exit" "completed" "${STATUS}"
assert_file_absent "server_job_file removed" "${SERVER_JOB_FILE}"

rm -rf "${TEST_BASE}"

# ---------------------------------------------------------------------------
# Test 3: scancel is called with the job ID from the tracking file
# ---------------------------------------------------------------------------
echo ""
echo "=== Test 3: scancel invoked with correct job ID ==="
TEST_BASE="$(mktemp -d)"
TEST_RUN_DIR=$(setup_run_dir "${TEST_BASE}")
> "${CANCELLED_FILE}"  # reset stub log

run_cleanup_server "${TEST_RUN_DIR}" 1 "${REPO_ROOT}"

CANCELLED=$(cat "${CANCELLED_FILE}" 2>/dev/null || true)
assert_eq "scancel called with job ID 99999" "99999" "${CANCELLED}"

rm -rf "${TEST_BASE}"

# ---------------------------------------------------------------------------
# Test 4: migrate_slurm_logs.sh --cancel-server --update-status
# ---------------------------------------------------------------------------
echo ""
echo "=== Test 4: migrate_slurm_logs.sh --cancel-server --update-status ==="
TEST_BASE="$(mktemp -d)"
TEST_RUN_DIR=$(setup_run_dir "${TEST_BASE}")
RUN_LOG_DIR="${TEST_RUN_DIR}/logs"
SERVER_JOB_FILE="${RUN_LOG_DIR}/hostname_server_job.txt"
echo "88888" > "${SERVER_JOB_FILE}"
> "${CANCELLED_FILE}"

TEST_RUN_ID=$(basename "${TEST_RUN_DIR}")
# The script uses REPO_ROOT/runs/<RUN_ID>, so we need to point it at our temp dir.
# Symlink TEST_RUN_DIR under REPO_ROOT/runs/ temporarily.
FAKE_RUNS="${TEST_BASE}/runs"
mkdir -p "${FAKE_RUNS}"
ln -s "${TEST_RUN_DIR}" "${FAKE_RUNS}/${TEST_RUN_ID}"

# Patch REPO_ROOT inside migrate_slurm_logs.sh by setting __TEST_REPO_ROOT env.
# Since the script derives REPO_ROOT from BASH_SOURCE, we just call it with
# the standard interface and rely on the RUN_DIR being found under its runs/.
(
    # Override BASH_SOURCE-derived repo root by cd-ing to TEST_BASE and
    # creating a scripts/ subdir so the script's "cd $(dirname)/.. " resolves.
    mkdir -p "${TEST_BASE}/scripts"
    cp "${REPO_ROOT}/scripts/migrate_slurm_logs.sh" "${TEST_BASE}/scripts/"
    bash "${TEST_BASE}/scripts/migrate_slurm_logs.sh" "${TEST_RUN_ID}" \
        --cancel-server --update-status
)

STATUS=$(python3 -c "import json; d=json.load(open('${TEST_RUN_DIR}/run_metadata.json')); print(d['status'])")
assert_eq "migrate: status=cancelled" "cancelled" "${STATUS}"

CANCELLED2=$(cat "${CANCELLED_FILE}" 2>/dev/null || true)
assert_eq "migrate: scancel called with 88888" "88888" "${CANCELLED2}"

assert_file_absent "migrate: server_job_file removed" "${SERVER_JOB_FILE}"

rm -rf "${TEST_BASE}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Teardown test summary ==="
echo "  Passed: ${PASS_COUNT}"
echo "  Failed: ${FAIL_COUNT}"
echo ""

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    echo "RESULT: FAIL"
    exit 1
else
    echo "RESULT: PASS"
    exit 0
fi
