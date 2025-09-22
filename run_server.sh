#!/bin/bash

# Simple script to run the server with virtual environment activated

echo "Loading environment variables from .env..."
export $(grep -v '^#' .env | grep -v '^$' | xargs)

echo "Activating virtual environment..."
source "${VENV_PATH}/bin/activate"

echo "Starting LLM server..."
echo "Server will be available at: http://localhost:${SERVER_PORT}"
echo "API docs will be available at: http://localhost:${SERVER_PORT}/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server using environment variables
python -m uvicorn server:app --host "${SERVER_HOST}" --port "${SERVER_PORT}" --workers "${SERVER_WORKERS}"
