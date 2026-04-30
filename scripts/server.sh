#!/bin/bash
# Wrapper: submits server job (must be run from repo root)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
exec sbatch scripts/cluster/server.sh "$@"
