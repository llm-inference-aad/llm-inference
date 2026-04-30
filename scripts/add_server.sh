#!/bin/bash
# Wrapper: delegates to scripts/cluster/add_server.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec bash scripts/cluster/add_server.sh "$@"
