#!/bin/bash
set -euo pipefail

# Submits four independent sbatch jobs, one per RAG condition, each running 100 requests.
# Jobs wait for the server job to finish successfully.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_JOB_ID="${1:-5150006}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
RUN_BASE="${RUN_BASE:-runs/rag_100_jobs}"
MANIFEST_DIR="${ROOT_DIR}/${RUN_BASE}"
mkdir -p "${MANIFEST_DIR}"

SUBMIT_LOG="${MANIFEST_DIR}/submitted_jobs.json"
STATUS_LOG="${MANIFEST_DIR}/job_status.csv"

cat > "${SUBMIT_LOG}" <<'JSON'
{
  "server_job_id": "__SERVER_JOB_ID__",
  "num_requests_per_config": __NUM_REQUESTS__,
  "jobs": []
}
JSON

python - <<PY
from pathlib import Path
path = Path(${SUBMIT_LOG@Q})
text = path.read_text()
text = text.replace("__SERVER_JOB_ID__", ${SERVER_JOB_ID@Q})
text = text.replace("__NUM_REQUESTS__", str(${NUM_REQUESTS}))
path.write_text(text)
PY

configs=(
  "1_rag_only"
  "2_rag_plus_spec"
  "3_rag_plus_constrained"
  "4_rag_plus_both"
)

for cfg in "${configs[@]}"; do
  job_name="rag_${cfg}"
  out_dir="${MANIFEST_DIR}/${cfg}"
  mkdir -p "${out_dir}"
  job_id=$(sbatch --parsable \
    --dependency=afterok:${SERVER_JOB_ID} \
    --job-name="${job_name}" \
    --export=ALL,NUM_REQUESTS_PER_CONFIG=${NUM_REQUESTS},CONFIG_NAME=${cfg},OUTPUT_DIR=${out_dir} \
    --output="${out_dir}/slurm-%j.out" \
    --error="${out_dir}/slurm-%j.err" \
    <<'SBATCH'
#!/bin/bash
set -euo pipefail
cd /home/hice1/jgil37/scratch/llm-inference
python run_rag_4way_200.py --config-name "${CONFIG_NAME}" --num-requests-per-config "${NUM_REQUESTS_PER_CONFIG}" --output-dir "${OUTPUT_DIR}"
SBATCH
  )

  python - <<PY
import csv, json, os
from pathlib import Path
submit_log = Path(${SUBMIT_LOG@Q})
data = json.loads(submit_log.read_text())
data["jobs"].append({
    "job_id": str(${job_id@Q}),
    "config": ${cfg@Q},
    "output_dir": ${out_dir@Q},
})
submit_log.write_text(json.dumps(data, indent=2))

status_path = Path(${STATUS_LOG@Q})
exists = status_path.exists()
with status_path.open("a", newline="") as f:
    w = csv.writer(f)
    if not exists:
        w.writerow(["job_id", "config", "status"])
    w.writerow([${job_id@Q}, ${cfg@Q}, "SUBMITTED"])
PY

  echo "Submitted ${cfg} => job ${job_id}"
done

echo
echo "Manifest: ${SUBMIT_LOG}"
echo "Status log: ${STATUS_LOG}"
