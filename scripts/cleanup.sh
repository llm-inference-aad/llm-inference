#!/bin/bash
# Wrapper: delegates to scripts/cluster/cleanup.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec bash scripts/cluster/cleanup.sh "$@"
