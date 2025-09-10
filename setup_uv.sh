#!/bin/bash
# Setup script for uv environment
# Run this script with: source setup_uv.sh

echo "Setting up uv environment..."

# Load anaconda module
module load anaconda3
echo "✅ Loaded anaconda3 module"

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"
echo "✅ Added uv to PATH"

# Verify uv is working
if command -v uv &> /dev/null; then
    echo "✅ uv is available (version: $(uv --version))"
else
    echo "❌ uv not found, installing..."
    pip install uv
    echo "✅ uv installed"
fi

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✅ Virtual environment exists"
else
    echo "Creating virtual environment..."
    uv sync
fi

echo ""
echo "🎉 Setup complete! You can now use:"
echo "  - uv run python script.py"
echo "  - uv add package-name"
echo "  - uv sync"
echo ""
