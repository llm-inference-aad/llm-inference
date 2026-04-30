# 1B Model CPU Testing Guide

## Quick Start

Run the guided test workflow:

```bash
bash run_1b_tests.sh
```

This will prompt you to choose a testing method. Choose based on your needs:

---

## Testing Methods

### Method A: Mock Server (Instant, No Model Required)
**Best for**: Quick validation before downloading model

```bash
bash run_1b_tests.sh
# Select: A
```

**What happens**:
- Starts lightweight mock FastAPI server (port 8002)
- Runs 5 configuration test cases immediately
- Results saved (no actual model inference)

**Time**: ~30 seconds  
**Output**: `runs/server-only/metrics/smoke_tests/config_*.json`

---

### Method B: Local Real Model on CPU  
**Best for**: Single-machine testing, immediate results

```bash
bash run_1b_tests.sh
# Select: B
```

**What happens**:
1. Checks if 1B model weights are downloaded
2. If missing, downloads model (⏱️ 10-15 min, first time only)
3. Starts `server.py` with 4-bit quantization on CPU
4. Runs 5 configuration tests
5. Cleans up server

**Requirements**:
- ~8GB free RAM (with 4-bit quantization)
- ~5GB disk space (model cache)
- 15-30 minutes (first run includes download)

**Time**: ~20 min (subsequent runs)  
**Output**: `runs/server-only/metrics/smoke_tests/config_*.json`

---

### Method C: Cluster Testing via sbatch (Recommended)
**Best for**: Handling large models, parallel experiments

```bash
bash run_1b_tests.sh
# Select: C
```

Or submit directly:

```bash
sbatch submit_1b_test.sh
```

**What happens**:
1. Submits job to cluster compute node
2. Resumes model download if interrupted
3. Starts server with 4-bit quantization
4. Runs 5 configuration tests
5. Saves results and logs

**Advantages**:
- Isolated compute environment
- 16GB memory allocation
- Comprehensive logging
- Job QoS tracking

**Monitor progress**:
```bash
# Get job ID
JOB_ID=12345  # from sbatch output

# Check status
squeue -j $JOB_ID

# Live logs
tail -f slurm_logs/1b_test_${JOB_ID}.log

# Check results
ls -lh runs/server-only/metrics/smoke_tests/
```

**Time**: ~30 min (depends on cluster queue)

---

## Testing Configurations

The test suite runs 5 configurations to evaluate different decoding strategies:

| Config | Constraints | Speculative | Purpose |
|--------|------------|------------|---------|
| 1. Baseline | ❌ None | ❌ None | Performance baseline |
| 2. JSON | ✅ JSON schema | ❌ None | Constraint overhead |
| 3. Regex | ✅ Regex pattern | ❌ None | Constraint overhead |
| 4. Speculative | ❌ None | ✅ Suffix method | Speedup without constraints |
| 5. Combined | ✅ JSON schema | ✅ Draft model | Full feature test |

---

## Test Results

Results are saved as JSON files in `runs/server-only/metrics/smoke_tests/`:

```bash
# View results
ls -1 runs/server-only/metrics/smoke_tests/config_*.json

# Example result
cat runs/server-only/metrics/smoke_tests/config_1_baseline.json
```

**Expected fields**:
```json
{
  "prompt": "...",
  "generated_text": "...",
  "tokens_generated": 50,
  "generation_time_ms": 1234,
  "constraint_validated": true,
  "speculative_enabled": false,
  "tokens_total": 100
}
```

---

## Manual Steps (If Not Using Guided Script)

### Step 1: Download Model

```bash
# Verify/download model weights (5-15 min)
python download_and_verify_model.py
```

### Step 2: Start Server on CPU

```bash
export MODEL_PATH="/home/hice1/jgil37/scratch/llm_models/meta-llama/Llama-3.2-1B-Instruct"

# Start with 4-bit quantization (fits in ~8GB RAM)
python server.py \
    --model_path "$MODEL_PATH" \
    --device_map auto \
    --enable_quantization \
    --quantization_bits 4 \
    --batch_size 1 \
    --port 8003
```

### Step 3: Run Tests (in another terminal)

```bash
bash test_five_configs.sh
```

---

## Troubleshooting

### Model download fails
```bash
# Check disk space
df -h /home/hice1/jgil37/scratch

# Check cache
du -sh /home/hice1/jgil37/scratch/llm_models

# Verify HuggingFace token (if gated model)
cat ~/.huggingface/token
```

### Server won't start
```bash
# Check if port is in use
lsof -i :8003

# Check server logs
tail -50 /tmp/server.log

# Try different port
python server.py --port 8004
```

### Out of memory
```bash
# Reduce batch size
export BATCH_SIZE=1

# Use more aggressive quantization
export QUANTIZATION_BITS=4  # 8-bit = 16GB, 4-bit = 8GB
```

### Tests timeout
```bash
# Increase curl timeout
# Edit test_five_configs.sh and add --max-time 300

# Or run tests individually with longer timeouts
curl --max-time 300 -X POST http://127.0.0.1:8003/generate \
    -H 'Content-Type: application/json' \
    -d '{...}'
```

---

## Performance Comparison

After tests complete, compare configurations:

```python
import json
import os
from pathlib import Path

results_dir = Path("runs/server-only/metrics/smoke_tests")

for config_file in sorted(results_dir.glob("config_*.json")):
    with open(config_file) as f:
        data = json.load(f)
    
    config_name = config_file.stem
    print(f"\n{config_name}:")
    print(f"  Time: {data.get('generation_time_ms', 'N/A')} ms")
    print(f"  Tokens: {data.get('tokens_generated', 'N/A')}")
    print(f"  Constraint Valid: {data.get('constraint_validated', 'N/A')}")
    if 'speculative_accepted' in data:
        print(f"  Speculative Acceptance: {data['speculative_accepted']}")
```

---

## Next Steps

1. **Choose testing method** → Run `bash run_1b_tests.sh`
2. **Review results** → Check `runs/server-only/metrics/smoke_tests/`
3. **Analyze performance** → Compare timing and token generation across configs
4. **Iterate** → Adjust batch sizes, quantization, or parameters as needed

---

## Files Overview

| File | Purpose |
|------|---------|
| `run_1b_tests.sh` | 🎯 Guided workflow (START HERE) |
| `download_and_verify_model.py` | Download & test 1B model |
| `test_five_configs.sh` | Run 5 test configurations |
| `submit_1b_test.sh` | sbatch job submission script |
| `server.py` | HF Transformers-based server (CPU-optimized) |
| `server_mock.py` | Mock server for quick testing |

---

## Questions?

Check logs:
```bash
# Local testing
tail -50 /tmp/server.log

# Cluster testing
tail -50 slurm_logs/1b_test_${JOB_ID}.log

# Download issues
cat slurm_logs/download_log.txt
```
