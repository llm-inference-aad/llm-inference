#!/bin/bash
# Test 5 different configurations: baseline, constrained, speculative (suffix), speculative (draft), combined

set -e

MODEL_PATH="/home/hice1/jgil37/scratch/llm_models/meta-llama/Llama-3.2-1B-Instruct"
PORT=8003
PYTHONPATH=.

echo "========================================"
echo "5-Config Test Suite: 1B Model on CPU"
echo "========================================"
echo

# Function to run a single configuration test
run_config() {
    local config_name="$1"
    local enable_spec="$2"
    local spec_method="$3"
    local constraint_type="$4"
    
    echo "[TEST] $config_name"
    echo "  Speculative: $enable_spec | Method: $spec_method | Constraint: $constraint_type"
    
    # Send request
    if [ "$constraint_type" = "json" ]; then
        PAYLOAD=$(cat <<EOF
{
  "input": "Generate a JSON response with keys success and score",
  "constraint_type": "json",
  "json_schema": {
    "type": "object",
    "required": ["success", "score"],
    "properties": {
      "success": {"type": "boolean"},
      "score": {"type": "number"}
    }
  }
}
EOF
        )
    elif [ "$constraint_type" = "regex" ]; then
        PAYLOAD=$(cat <<EOF
{
  "input": "Generate a phone number",
  "constraint_type": "regex",
  "constraint": "^[0-9]{3}-[0-9]{3}-[0-9]{4}$"
}
EOF
        )
    else
        PAYLOAD=$(cat <<EOF
{
  "input": "Generate text response"
}
EOF
        )
    fi
    
    # Add speculative params if enabled
    if [ "$enable_spec" = "true" ]; then
        PAYLOAD=$(echo "$PAYLOAD" | python -c "
import sys, json
payload = json.load(sys.stdin)
payload['enable_speculative'] = True
payload['speculative_method'] = '$spec_method'
payload['num_speculative_tokens'] = 5
if '$spec_method' == 'draft_model':
    payload['draft_model'] = 'meta-llama/Llama-3.2-1B-Instruct'
print(json.dumps(payload))
")
    fi
    
    # Make request and save result
    curl -sS -X POST "http://127.0.0.1:$PORT/generate" \
        -H 'Content-Type: application/json' \
        -d "$PAYLOAD" \
        > "runs/server-only/metrics/smoke_tests/config_${config_name}.json" 2>&1 || echo "Request failed for $config_name"
    
    echo "  ✓ Result saved to config_${config_name}.json"
    echo
}

# Create results directory
mkdir -p runs/server-only/metrics/smoke_tests

# Run all 5 configurations
echo "Starting tests..."
echo

# Config 1: Baseline (no constraints, no speculative)
run_config "1_baseline" "false" "none" "none"

# Config 2: Constrained JSON only
run_config "2_constrained_json" "false" "none" "json"

# Config 3: Constrained Regex only
run_config "3_constrained_regex" "false" "none" "regex"

# Config 4: Speculative Decoding (suffix method) only
run_config "4_speculative_suffix" "true" "suffix" "none"

# Config 5: Combined (Constrained JSON + Speculative draft_model)
run_config "5_combined_json_spec" "true" "draft_model" "json"

echo "========================================"
echo "Results Summary"
echo "========================================"
for f in runs/server-only/metrics/smoke_tests/config_*.json; do
    name=$(basename "$f" .json)
    echo "✓ $name:"
    if [ -f "$f" ]; then
        python -m json.tool < "$f" 2>/dev/null | head -20 || cat "$f" | head -20
    fi
    echo
done

echo "All tests complete! Results saved to: runs/server-only/metrics/smoke_tests"
