#!/bin/bash
# Quick script to switch between configurations
# Usage: ./switch_config.sh <config_name>
# Or: ./switch_config.sh help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define available configs
declare -A CONFIGS=(
    ["hf"]="HuggingFace Baseline (No vLLM)"
    ["vllm"]="vLLM Only (PagedAttention + Prefix Caching)"
    ["constrained"]="vLLM + Constrained Decoding"
    ["speculative"]="vLLM + Speculative Decoding"
    ["all"]="vLLM + Constrained + Speculative Decoding"
    ["draft_1b"]="vLLM + Speculative (Draft: 1B same-family)"
    ["draft_3b"]="vLLM + Speculative (Draft: 3B same-family)"
    ["draft_cross"]="vLLM + Speculative (Draft: cross-family)"
)

declare -A CONFIG_FILES=(
    ["hf"]=".env.hf_baseline"
    ["vllm"]=".env.vllm_only"
    ["constrained"]=".env.vllm_constrained"
    ["speculative"]=".env.vllm_speculative"
    ["all"]=".env.vllm_all"
    ["draft_1b"]=".env.vllm_draft_1b"
    ["draft_3b"]=".env.vllm_draft_3b"
    ["draft_cross"]=".env.vllm_draft_cross"
)

function show_help() {
    echo "LLM Inference Configuration Switcher"
    echo "===================================="
    echo ""
    echo "Available configurations:"
    echo ""
    for config in $(printf "%s\n" "${!CONFIGS[@]}" | sort); do
        printf "  %-15s • %s\n" "$config" "${CONFIGS[$config]}"
    done
    echo ""
    echo "Usage:"
    echo "  ./switch_config.sh hf              # Switch to HuggingFace baseline"
    echo "  ./switch_config.sh vllm            # Switch to vLLM only"
    echo "  ./switch_config.sh constrained     # Switch to vLLM + constraints"
    echo "  ./switch_config.sh speculative     # Switch to vLLM + speculation"
    echo "  ./switch_config.sh all             # Switch to vLLM + all features"
    echo "  ./switch_config.sh draft_1b        # Switch to vLLM + draft (1B)"
    echo "  ./switch_config.sh draft_3b        # Switch to vLLM + draft (3B)"
    echo "  ./switch_config.sh draft_cross     # Switch to vLLM + draft (cross-family)"
    echo "  ./switch_config.sh list            # List all configurations"
    echo "  ./switch_config.sh current         # Show current config"
    echo "  ./switch_config.sh help            # Show this help"
    echo ""
}

function switch_config() {
    local config=$1
    
    if [ -z "$config" ]; then
        show_help
        exit 1
    fi
    
    if [ "$config" = "help" ]; then
        show_help
        exit 0
    fi
    
    if [ "$config" = "list" ]; then
        echo "Available configurations:"
        for cfg in $(printf "%s\n" "${!CONFIGS[@]}" | sort); do
            echo "  • $cfg"
        done
        exit 0
    fi
    
    if [ "$config" = "current" ]; then
        if [ -f ".env" ]; then
            echo "Current configuration: .env"
            head -5 .env
        else
            echo "No .env file found"
        fi
        exit 0
    fi
    
    # Validate config name
    if [ -z "${CONFIG_FILES[$config]}" ]; then
        echo "❌ Unknown configuration: $config"
        echo ""
        show_help
        exit 1
    fi
    
    local config_file="${CONFIG_FILES[$config]}"
    local description="${CONFIGS[$config]}"
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        echo "❌ Configuration file not found: $config_file"
        exit 1
    fi
    
    # Backup current .env if it exists
    if [ -f ".env" ]; then
        timestamp=$(date +%s)
        backup_file=".env.backup.${timestamp}"
        cp .env "$backup_file"
        echo "📦 Backed up previous .env to: $backup_file"
    fi
    
    # Copy config to .env
    cp "$config_file" .env
    # Create result directories for this configuration and inject RUN_* env vars
    RESULT_ROOT="$SCRIPT_DIR/switch_config_results"
    CONFIG_RESULT_DIR="$RESULT_ROOT/$config"
    METRICS_DIR="$CONFIG_RESULT_DIR/metrics"
    LOG_DIR="$CONFIG_RESULT_DIR/logs"
    ERROR_DIR="$CONFIG_RESULT_DIR/errors"

    mkdir -p "$METRICS_DIR" "$LOG_DIR" "$ERROR_DIR"

    # Remove any existing RUN_* lines from .env to avoid duplicates
    sed -i '/^RUN_DIR=/d' .env || true
    sed -i '/^RUN_METRICS_DIR=/d' .env || true
    sed -i '/^RUN_LOG_DIR=/d' .env || true
    sed -i '/^RUN_ERRORS_DIR=/d' .env || true
    sed -i '/^RUN_ID=/d' .env || true

    # Append run-specific variables so the server writes metrics/logs into the config folder
    cat >> .env <<EOF
# Added by switch_config.sh
RUN_ID="$config"
RUN_DIR="$CONFIG_RESULT_DIR"
RUN_METRICS_DIR="$METRICS_DIR"
RUN_LOG_DIR="$LOG_DIR"
RUN_ERRORS_DIR="$ERROR_DIR"
EOF

    echo "✅ Switched to: $config"
    echo "📋 $description"
    echo ""
    echo "Configuration summary:"
    grep -E "^(VLLM_BACKEND|USE_VLLM|CONSTRAINT_TYPE|ENABLE_SPECULATIVE_DECODING|VLLM_SPECULATIVE_METHOD|RUN_METRICS_DIR|RUN_DIR)=" .env | head -20
    echo ""
    echo "Next steps:"
    echo "  1. Run locally:"
    echo "     python -m uvicorn server_vllm:app --host 0.0.0.0 --port 8001"
    echo ""
    echo "  2. Or submit to Slurm:"
    echo "     sbatch -p ice-gpu --gpus-per-node=1 server.sh"
    echo ""
    echo "  3. Test the server:"
    echo "     curl http://localhost:8001/"
    echo ""
}

# Call switch_config with arguments
switch_config "$@"
