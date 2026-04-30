#!/bin/bash
# Helper script to run the Streamlit dashboard
# Activates the virtual environment and runs the dashboard

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate 2>/dev/null || source venv/bin/activate
streamlit run app/dashboard.py "$@"
