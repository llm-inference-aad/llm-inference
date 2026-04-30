#!/bin/bash
# Wrapper: delegates to scripts/cluster/manage_servers.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec bash scripts/cluster/manage_servers.sh "$@"
