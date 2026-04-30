#!/bin/bash
# Quick test script for verifying a running server
# Usage: ./test_server.sh <server_url> [config_name]

SERVER_URL="${1:-http://localhost:8001}"
CONFIG_NAME="${2:-test}"

echo "Testing LLM Server: $SERVER_URL"
echo "=================================="
echo ""

# Test 1: Health check
echo "Test 1: Health Check (GET /)"
response=$(curl -s "$SERVER_URL/")
if echo "$response" | grep -q "running"; then
    echo "✅ Server is running"
    echo "$response" | jq . 2>/dev/null || echo "$response"
else
    echo "❌ Server health check failed"
    echo "$response"
    exit 1
fi
echo ""

# Test 2: Basic generation without constraints
echo "Test 2: Basic Generation (No Constraints)"
response=$(curl -s "$SERVER_URL/generate" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_new_tokens": 128,
    "temperature": 0.7
  }')

if echo "$response" | jq . 2>/dev/null | grep -q "generated_text"; then
    echo "✅ Basic generation works"
    latency=$(echo "$response" | jq '.e2e_latency_sec // ._latency_sec')
    echo "   Latency: ${latency}s"
    echo "   Output: $(echo "$response" | jq -r '.generated_text' | head -c 100)..."
else
    echo "❌ Generation failed"
    echo "$response"
    exit 1
fi
echo ""

# Test 3: JSON Constraint (if available)
echo "Test 3: JSON Constraint"
response=$(curl -s "$SERVER_URL/generate" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Extract name and age: John is 25 years old",
    "max_new_tokens": 256,
    "constraint_type": "json",
    "json_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      },
      "required": ["name"]
    }
  }')

if echo "$response" | jq . 2>/dev/null | grep -q "generated_text"; then
    constraint=$(echo "$response" | jq -r '.constraint_type // "none"')
    if [ "$constraint" != "none" ] && [ "$constraint" != "null" ]; then
        echo "✅ JSON constraint supported (constraint_type: $constraint)"
    else
        echo "⚠️  JSON constraint skipped (not enabled)"
    fi
    echo "   Output: $(echo "$response" | jq -r '.generated_text' | head -c 100)..."
else
    echo "⚠️  JSON constraint test failed (may not be enabled)"
    echo "$response" | jq . 2>/dev/null || echo "$response"
fi
echo ""

# Test 4: Multiple requests
echo "Test 4: Multiple Requests (Latency Distribution)"
declare -a latencies=()
for i in {1..3}; do
    response=$(curl -s "$SERVER_URL/generate" \
      -X POST \
      -H "Content-Type: application/json" \
      -d "{
        \"prompt\": \"Test request $i\",
        \"max_new_tokens\": 128
      }")
    
    latency=$(echo "$response" | jq '.e2e_latency_sec // ._latency_sec')
    latencies+=("$latency")
    echo "   Request $i: ${latency}s"
done

# Calculate average
avg_latency=$(printf '%s\n' "${latencies[@]}" | awk '{sum+=$1; count++} END {print sum/count}')
echo "✅ Average latency: ${avg_latency}s"
echo ""

# Final summary
echo "=================================="
echo "✅ Server tests completed!"
echo ""
echo "Configuration: $CONFIG_NAME"
echo "Server URL: $SERVER_URL"
echo ""
echo "Next steps:"
echo "  • Check metrics in: runs/*/metrics/latency-*.json"
echo "  • View logs: cat logs/server-runtime-*.out"
echo "  • Stop server: pkill -f 'uvicorn server'"
echo ""
