# Quick Start: Running Different Configurations

## TL;DR - Quick Commands

Switch to a configuration and run:

```bash
# 1. HuggingFace Baseline
./switch_config.sh hf

# 2. vLLM Only
./switch_config.sh vllm

# 3. vLLM + Constraints
./switch_config.sh constrained

# 4. vLLM + Speculation
./switch_config.sh speculative

# 5. vLLM + Constraints + Speculation
./switch_config.sh all

# Then run the server
sbatch -p ice-gpu --gpus-per-node=1 server.sh
```

---

## Configuration Manager

Use `switch_config.sh` to manage configurations:

```bash
# Show available configs
./switch_config.sh list

# Check what's currently active
./switch_config.sh current

# Switch to a config (creates backup)
./switch_config.sh hf
./switch_config.sh vllm
./switch_config.sh constrained
./switch_config.sh speculative
./switch_config.sh all
```

**What it does:**
- Copies the `.env.<config>` file to `.env`
- Backs up previous `.env` with timestamp
- Shows summary of key settings
- Displays next steps

---

## Running A Server

### Method 1: Local Interactive (for testing)

```bash
# Terminal 1: Start the server
python -m uvicorn server_vllm:app --host 0.0.0.0 --port 8001 --workers 1

# Terminal 2: Test it
./test_server.sh http://localhost:8001 hf

# Or make requests directly
curl http://localhost:8001/
```

### Method 2: Slurm Batch Job (for benchmarks)

```bash
# Submit your current .env as a job
sbatch -p ice-gpu --gpus-per-node=1 server.sh

# Check status
squeue -u $USER

# Follow logs
tail -f logs/server-runtime-*.out
```

### Method 3: Multiple Jobs in Parallel

Run all 5 configurations and compare:

```bash
# Run each configuration
for config in hf vllm constrained speculative all; do
    echo "Starting: $config"
    ./switch_config.sh $config
    export RUN_ID="benchmark_${config}"
    sbatch -p ice-gpu --gpus-per-node=1 server.sh
    sleep 2
done

# Monitor all jobs
watch -n 5 'squeue -u $USER'

# Collect metrics afterward
find runs -name "latency-*.json" | head -10
```

---

## Testing Your Server

After starting a server:

```bash
# Quick health check
curl http://localhost:8001/

# Run full test suite
./test_server.sh http://localhost:8001 my_config

# Send individual requests
curl http://localhost:8001/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 128}'
```

---

## Understanding Each Configuration

### 1️⃣ HuggingFace Baseline (`.env.hf_baseline`)

**When to use:** Baseline comparison, legacy workloads

```bash
./switch_config.sh hf
```

- **Backend:** HuggingFace Transformers pipeline
- **Batching:** Manual (4 workers)
- **KV Cache:** ~30% utilization
- **Features:** None
- **Expected latency:** 1.0x (baseline)

---

### 2️⃣ vLLM Only (`.env.vllm_only`)

**When to use:** Best latency without constraints

```bash
./switch_config.sh vllm
```

- **Backend:** vLLM engine
- **KV Cache:** ~95% utilization (PagedAttention)
- **Features:** Prefix caching
- **Batching:** Continuous
- **Expected latency:** 0.6x (-40% vs HF)

---

### 3️⃣ vLLM + Constraints (`.env.vllm_constrained`)

**When to use:** Guaranteed structured output (JSON, grammar, regex)

```bash
./switch_config.sh constrained
```

**Test with:**
```bash
curl http://localhost:8001/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Extract: Alice, age 30",
    "max_new_tokens": 256,
    "constraint_type": "json",
    "json_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      }
    }
  }'
```

- **Backend:** vLLM + constraints
- **Output:** 100% valid JSON
- **Performance impact:** -10% latency
- **Expected latency:** 0.65x (-35% vs HF)

---

### 4️⃣ vLLM + Speculation (`.env.vllm_speculative`)

**When to use:** Maximum speed with unconstrained output

```bash
./switch_config.sh speculative
```

- **Backend:** vLLM + suffix speculation
- **Speculation:** Adaptive tree depth
- **Output:** Unconstrained (any text valid)
- **Performance gain:** ~2.0-2.5x speedup
- **Expected latency:** 0.3x (-70% vs HF)

---

### 5️⃣ All Features (`.env.vllm_all`)

**When to use:** Both 100% valid output AND fast generation

```bash
./switch_config.sh all
```

- **Backend:** vLLM with constraints + speculation
- **Temperature:** Auto-capped to 0.25 (improves acceptance)
- **Output:** 100% valid + fast
- **Performance:** Speculation speedup + constraint validation
- **Expected latency:** 0.35x (-65% vs HF)

---

## Comparing Results

After running multiple configurations:

### 1. Find all metrics
```bash
find runs -name "latency-*.json" | sort
```

### 2. Compare latencies
```bash
for f in runs/*/metrics/latency-*.json; do
    config=$(basename $(dirname $(dirname "$f")))
    avg=$(jq '[.requests[].generation_time_sec] | add/length' "$f")
    echo "$config: ${avg}s"
done
```

### 3. Extract specific fields
```bash
# Show all field names
jq '.requests[0] | keys' runs/benchmark_*/metrics/latency-*.json

# Extract constraint_type usage
jq -r '.requests[] | select(.constraint_type != null) | .constraint_type' runs/benchmark_constrained/metrics/*.json | sort | uniq -c

# Check speculative acceptance
jq -r '.requests[] | select(.speculative_accepted != null) | .speculative_accepted' runs/benchmark_*/metrics/*.json
```

### 4. Generate comparison report
```bash
cat > compare_configs.py << 'EOF'
import json
import glob

configs = {}
for metrics_file in glob.glob("runs/*/metrics/latency-*.json"):
    with open(metrics_file) as f:
        data = json.load(f)
        config_name = metrics_file.split('/')[1]
        latencies = [r['generation_time_sec'] for r in data['requests']]
        configs[config_name] = {
            'avg_latency': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'requests': len(latencies)
        }

print("Configuration Comparison")
print("=" * 60)
for config in sorted(configs.keys()):
    stats = configs[config]
    print(f"{config:30} Avg: {stats['avg_latency']:.3f}s "
          f"(min: {stats['min_latency']:.3f}, max: {stats['max_latency']:.3f})")
EOF

python compare_configs.py
```

---

## Troubleshooting

### Server won't start

```bash
# Check if port is in use
lsof -i :8001

# Check logs
tail -f logs/server-runtime-*.out
tail -f runs/*/logs/server-runtime-*.err

# Try different port
export SERVER_PORT=8002
sbatch server.sh
```

### Constraints not working

```bash
# Verify config is correct
grep CONSTRAINT .env

# Enable logging
export CONSTRAINT_LOGGING_ENABLED=true

# Check that vLLM is active
grep VLLM_BACKEND .env  # Should be true
```

### Speculation taking too long

```bash
# Try fewer speculative tokens
export VLLM_NUM_SPECULATIVE_TOKENS=3

# Or lower the temperature cap
export SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP=0.15
```

---

## Environment Variables Cheat Sheet

| Variable | Values | Default |
|----------|--------|---------|
| `VLLM_BACKEND` | `true` / `false` | `true` |
| `CONSTRAINT_TYPE` | `json` / `grammar` / `regex` / `` | `` |
| `ENABLE_SPECULATIVE_DECODING` | `true` / `false` | `false` |
| `VLLM_SPECULATIVE_METHOD` | `suffix` / `ngram` / `draft_model` | `suffix` |
| `VLLM_NUM_SPECULATIVE_TOKENS` | int | `5` |
| `CONSTRAINT_LOGGING_ENABLED` | `true` / `false` | `false` |

---

## Example: Full Benchmark Run

```bash
#!/bin/bash
# Full benchmark: all 5 configs

configs=("hf" "vllm" "constrained" "speculative" "all")

for config in "${configs[@]}"; do
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    
    # Switch config
    ./switch_config.sh "$config"
    
    # Set run ID for tracking
    export RUN_ID="benchmark_${config}"
    
    # Submit to Slurm
    JOB_ID=$(sbatch -p ice-gpu --gpus-per-node=1 server.sh | awk '{print $4}')
    echo "Submitted job: $JOB_ID"
    
    # Wait for completion
    while squeue -j $JOB_ID &>/dev/null; do
        sleep 10
    done
    
    echo "Completed: $config"
done

echo "All benchmarks completed!"
echo "Results in: runs/benchmark_*"
```

Save as `benchmark_all.sh`, then:
```bash
chmod +x benchmark_all.sh
./benchmark_all.sh
```

---

## Next Steps

1. **Choose a config:** `./switch_config.sh <name>`
2. **Start server:** `sbatch server.sh`
3. **Test it:** `./test_server.sh http://localhost:8001 <name>`
4. **View results:** Check `runs/benchmark_*/metrics/`
5. **Compare:** Use the comparison script above
