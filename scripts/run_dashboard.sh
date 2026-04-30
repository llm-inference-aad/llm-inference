#!/bin/bash
# Wrapper: delegates to scripts/cluster/run_dashboard.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec bash scripts/cluster/run_dashboard.sh "$@"
