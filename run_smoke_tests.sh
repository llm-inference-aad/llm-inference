#!/usr/bin/env bash
# Comprehensive smoke test for constrained and speculative decoding

SERVER_URL="http://127.0.0.1:8002"
RESULTS_DIR="runs/server-only/metrics/smoke_tests"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Constrained & Speculative Decoding Tests"
echo "========================================"
echo

# Test 1: Constrained JSON
echo "[1/4] Testing Constrained JSON Decoding..."
curl -sS -X POST "$SERVER_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Generate a JSON response",
    "constraint_type": "json",
    "json_schema": {
      "type": "object",
      "required": ["accuracy"],
      "properties": {
        "accuracy": {"type": "number"},
        "note": {"type": "string"}
      }
    }
  }' | tee "$RESULTS_DIR/test_1_constrained_json.json" | python -m json.tool
echo

# Test 2: Constrained Regex
echo "[2/4] Testing Constrained Regex Decoding..."
curl -sS -X POST "$SERVER_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Generate a response for regex",
    "constraint_type": "regex",
    "constraint": "^[0-9]{3}-[0-9]{2}-[0-9]{4}$"
  }' | tee "$RESULTS_DIR/test_2_constrained_regex.json" | python -m json.tool
echo

# Test 3: Speculative Decoding (Suffix method)
echo "[3/4] Testing Speculative Decoding (Suffix)..."
curl -sS -X POST "$SERVER_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Generate text with speculative decoding",
    "enable_speculative": true,
    "speculative_method": "suffix",
    "num_speculative_tokens": 5
  }' | tee "$RESULTS_DIR/test_3_speculative_suffix.json" | python -m json.tool
echo

# Test 4: Combined Constrained + Speculative
echo "[4/4] Testing Combined Constrained + Speculative Decoding..."
curl -sS -X POST "$SERVER_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Generate JSON with speculative decoding",
    "constraint_type": "json",
    "json_schema": {
      "type": "object",
      "properties": {
        "success": {"type": "boolean"},
        "score": {"type": "number"}
      }
    },
    "enable_speculative": true,
    "speculative_method": "draft_model",
    "draft_model": "meta-llama/Llama-3.2-1B-Instruct"
  }' | tee "$RESULTS_DIR/test_4_combined.json" | python -m json.tool
echo

# Parse and display results
echo "========================================"
echo "Test Results Summary"
echo "========================================"
echo

for test_file in "$RESULTS_DIR"/test_*.json; do
  test_name=$(basename "$test_file" .json)
  echo "✓ $test_name:"
  echo "  - constraint_type: $(grep -o '"constraint_type":"[^"]*"' "$test_file" | cut -d'"' -f4)"
  echo "  - speculative_enabled: $(grep -o '"speculative_decoding_enabled":[^,}]*' "$test_file" | cut -d':' -f2)"
  echo "  - latency: $(grep -o '"_latency_sec":[0-9.]*' "$test_file" | cut -d':' -f2)s"
  echo
done

echo "All test files saved to: $RESULTS_DIR"
echo "✓ Constrained and Speculative Decoding Smoke Tests Complete!"
