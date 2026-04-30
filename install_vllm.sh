#!/bin/bash
echo "Installing vLLM into .venv..."
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    # Use uv pip install or fallback to standard pip install
    if command -v uv >/dev/null 2>&1; then
        uv pip install -e .
    else
        pip install -e .
    fi
    echo "Installation complete."
else
    echo "Error: .venv not found. Please ensure your virtual environment is set up."
fi
