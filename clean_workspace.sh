#!/bin/bash
echo "Cleaning up tests/, 0/, and activate/ directories..."
rm -rf tests/ 0/ activate/

echo "Removing old and empty directories in runs/ and metrics/..."
find runs/ -type d -empty -delete
rm -rf metrics/oldData

echo "Moving scattered shell scripts to scripts/ directory..."
mkdir -p scripts
mv add_server.sh scripts/ 2>/dev/null
mv cleanup.sh scripts/ 2>/dev/null
mv compare_runs.sh scripts/ 2>/dev/null
mv load_balancer.sh scripts/ 2>/dev/null
mv manage_servers.sh scripts/ 2>/dev/null
mv monitor_cluster.sh scripts/ 2>/dev/null
mv run_dashboard.sh scripts/ 2>/dev/null
mv start_cluster.sh scripts/ 2>/dev/null

echo "Workspace cleanup complete."
