#!/bin/bash
# Helper script to run the Streamlit dashboard
# Activates the virtual environment and runs the dashboard

cd "$(dirname "$0")"
source venv/bin/activate
streamlit run dashboard.py "$@"

