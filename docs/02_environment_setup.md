# Environment Variables Setup

This project uses environment variables to avoid hardcoding paths and sensitive information.

## Setup Instructions

### 1. Create your `.env` file

Copy the example file and customize it for your environment:

```bash
cp .env.example .env
```

### 2. Edit your `.env` file

Update the values in `.env` to match your system:

```bash
# Project root directory
LLM_INFERENCE_ROOT_DIR=/path/to/your/llm-inference/llm-inference

# Virtual environment path
VENV_PATH=/path/to/your/llm-inference/llm-inference/.venv

# Model paths
MODEL_PATH=/path/to/your/model
MODEL_PATH_SIMPLE=/path/to/your/model/snapshots/specific-version

# Server configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=1

# Log file paths
HOSTNAME_LOG_FILE=/path/to/your/hostname.log

# CUDA configuration
CUDA_VISIBLE_DEVICES=0
MKL_THREADING_LAYER=GNU

# Hugging Face token (uncomment and set your actual token)
HF_TOKEN=your_actual_token_here
HUGGING_FACE_HUB_TOKEN=your_actual_token_here
```

## Environment Variables Reference

### Required Variables

- `LLM_INFERENCE_ROOT_DIR`: Root directory of the project
- `VENV_PATH`: Path to the Python virtual environment
- `MODEL_PATH`: Path to the main model directory
- `MODEL_PATH_SIMPLE`: Path to specific model snapshot (for simple server)

### Optional Variables

- `SERVER_HOST`: Server host address (default: 0.0.0.0)
- `SERVER_PORT`: Server port (default: 8000)
- `SERVER_WORKERS`: Number of server workers (default: 1)
- `HOSTNAME_LOG_FILE`: Path to hostname log file
- `CUDA_VISIBLE_DEVICES`: CUDA device IDs (default: 0)
- `MKL_THREADING_LAYER`: MKL threading layer (default: GNU)

### Security Variables

- `HF_TOKEN`: Hugging Face API token
- `HUGGING_FACE_HUB_TOKEN`: Alternative Hugging Face token variable

## Usage

The `server.sh` script automatically loads environment variables from the `.env` file. If the `.env` file doesn't exist, it will use sensible defaults.

### Manual Loading

To manually load environment variables:

```bash
# Load all variables from .env
source .env

# Or load specific variables
export MODEL_PATH="/your/model/path"
export HF_TOKEN="your_token"
```

### Python Usage

In Python files, use `os.getenv()` to read environment variables:

```python
import os

model_path = os.getenv("MODEL_PATH", "default/path")
token = os.getenv("HF_TOKEN")
```

## Security Notes

- **Never commit `.env` files** - they're already in `.gitignore`
- **Use `.env.example`** for sharing configuration templates
- **Set proper file permissions** on `.env` files: `chmod 600 .env`
- **Rotate tokens regularly** and update environment variables

## Troubleshooting

### Environment variables not loading

1. Check that `.env` file exists: `ls -la .env`
2. Verify file format (no spaces around `=`): `KEY=value`
3. Check for comments (lines starting with `#` are ignored)
4. Ensure no trailing spaces in values

### Default values

If environment variables aren't set, the system will use these defaults:
- `LLM_INFERENCE_ROOT_DIR`: Current working directory
- `SERVER_HOST`: Server hostname
- `SERVER_PORT`: 8000
- `SERVER_WORKERS`: 1
- `CUDA_VISIBLE_DEVICES`: 0
