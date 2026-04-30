#!/bin/bash
# Wrapper to run finetune job - delegates to scripts/finetune/finetune_job.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec bash scripts/finetune/finetune_job.sh "$@"
