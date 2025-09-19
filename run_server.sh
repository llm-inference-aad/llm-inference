#!/bin/bash

# Simple script to run the server with virtual environment activated

echo "Loading environment variables from .env..."
export $(grep -v '^#' .env | grep -v '^$' | xargs)

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting LLM server..."
echo "Server will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
