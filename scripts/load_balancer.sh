#!/bin/bash
# Wrapper: submits load balancer job
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec sbatch scripts/cluster/load_balancer.sh "$@"
