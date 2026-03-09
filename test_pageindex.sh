#!/bin/bash
#SBATCH --job-name=pageindex_test
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --output=metrics/slurm-results/slurm-pageindex-test-%j.out
#SBATCH --error=metrics/slurm-results/slurm-pageindex-test-%j.err

set -Eeuo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p metrics/slurm-results

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH."
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
else
  echo "Warning: .env not found, using defaults"
fi

echo "=== PageIndex local-model integration test ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Time: $(date)"

DEFAULT_COMPLEX_MD="docs/rag_small_improvements.md"
EXTRA_ARGS=()
HAS_INPUT_PATH=false
for arg in "$@"; do
  case "$arg" in
    --md-path|--pdf-path|--md-path=*|--pdf-path=*)
      HAS_INPUT_PATH=true
      ;;
  esac
done

if [[ "$HAS_INPUT_PATH" == false ]]; then
  EXTRA_ARGS+=(--md-path "$DEFAULT_COMPLEX_MD")
fi

# Print all PageIndex local LLM prompts/responses unless explicitly disabled.
export PAGEINDEX_TRACE_LLM_CALLS="${PAGEINDEX_TRACE_LLM_CALLS:-1}"
echo "PAGEINDEX_TRACE_LLM_CALLS=$PAGEINDEX_TRACE_LLM_CALLS"
if [[ "$HAS_INPUT_PATH" == false ]]; then
  echo "Using default complex markdown: $DEFAULT_COMPLEX_MD"
fi

# Keep uv cache inside workspace in this environment.
UV_CACHE_DIR=.uv-cache uv run python scripts/test_pageindex_local.py "${EXTRA_ARGS[@]}" "$@"
