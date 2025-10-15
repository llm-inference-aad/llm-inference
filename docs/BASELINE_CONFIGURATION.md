# LLMGE Baseline Configuration Guide

## Project Goal: LLM Inference Optimization

**Objective:** Establish a baseline with existing LLMGE setup, then compare metrics after applying optimization techniques (RAG, speculative decoding, etc.).

---

## Understanding the Architecture

### Current Setup (2-Tier Execution)

```
┌─────────────────────────────────────────────────────────────┐
│ SLURM Job (sbatch run.sh)                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Main Evolution Loop (run_improved.py)                    │ │
│ │ - Population management                                  │ │
│ │ - Genetic operators                                      │ │
│ │ - Fitness evaluation orchestration                       │ │
│ │                                                           │ │
│ │   For each gene:                                         │ │
│ │   ┌───────────────────────────────────────────────────┐ │ │
│ │   │ if LOCAL=True:                                     │ │ │
│ │   │   bash train.sh  (runs on same SLURM node)        │ │ │
│ │   │ else:                                              │ │ │
│ │   │   sbatch train.sh  (submits new SLURM job)        │ │ │
│ │   └───────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key Point:** `LOCAL=True/False` controls **gene evaluation distribution**, NOT the main job!

---

## Baseline Configuration Parameters

### 1. Evolution Parameters (`src/cfg/constants.py`)

```python
# Population & Generations
POPULATION_SIZE = 8      # Number of genes per generation
N_GENERATIONS = 10       # Number of evolution iterations

# Genetic Algorithm
MUTPB = 0.5             # Mutation probability
CXPB = 0.3              # Crossover probability
TOURNSIZE = 3           # Tournament selection size
```

**Baseline Recommendation:**
- Start with `POPULATION_SIZE=8`, `N_GENERATIONS=5-10` for initial baseline
- This gives you ~40-80 LLM inference calls to establish latency baseline
- Increase for more robust statistics after proving the pipeline works

### 2. Execution Mode

```python
# Gene Evaluation Distribution
LOCAL = True            # Run evals on same node (faster startup)
RUN_COMMAND = 'bash'    # Controlled by LOCAL flag

# Alternative for distributed execution:
LOCAL = False           # Submit each eval as separate SLURM job
RUN_COMMAND = 'sbatch'  # More parallel, higher overhead
```

**Baseline Recommendation:**
- **Use `LOCAL=True`** for baseline
- Pros: Simpler, faster startup, easier debugging, fewer SLURM jobs
- Cons: Sequential evaluation (but genes are independent)
- Switch to `LOCAL=False` only if you need massive parallelization

### 3. LLM Configuration

```python
# LLM Model Selection
LLM_MODEL = 'local_server'  # Use your DeepSeek server

# Server Settings
LLM_API_BASE = 'http://localhost:8000'
INFERENCE_SUBMISSION = False  # Direct HTTP calls, not job submission

# Token Limits (already optimized for DeepSeek)
MAX_NEW_TOKENS_FN = 32000
MAX_NEW_TOKENS_TRAIN = 32000
MAX_NEW_TOKENS_ARCH = 32000
```

**Baseline Recommendation:**
- ✅ Already optimized for DeepSeek-R1-Distill-Qwen-32B
- Token limits set to 32k (well below 130k window)
- Keep `INFERENCE_SUBMISSION=False` for direct API calls

### 4. Batching Configuration (`server.py`)

```python
# Batch Processing
BATCH_SIZE = 8          # Max requests per batch
BATCH_WAIT_TIME = 2     # Max wait time (seconds) to fill batch

# Model Settings
MODEL_PATH = "/storage/ice-shared/vip-vvk/llm_storage/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

**Baseline Recommendation:**
- Match `BATCH_SIZE` to `POPULATION_SIZE` (currently both=8)
- This batches all gene requests together for efficient inference
- `BATCH_WAIT_TIME=2` prevents indefinite waiting

### 5. SLURM Resource Allocation (`run.sh`)

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:H100:1
```

**Baseline Recommendation:**
- 1 node, 32GB RAM, 1 H100 GPU (sufficient for sequential evals)
- Increase `--nodes` or use `LOCAL=False` for parallelization later
- Time depends on dataset size and population

---

## Metrics to Collect for Baseline

### 1. LLM Inference Metrics (Automatic)

Located in: `runs/{run_id}/metrics/latency-{hash}.json`

Key metrics:
- `_latency_sec`: End-to-end latency per request
- `batch_processing_time_sec`: Pure inference time
- `queue_wait_time_sec`: Time waiting for batch to fill
- `prompt_length`: Input tokens
- `max_new_tokens`: Output limit

### 2. Model Performance Metrics (Automatic)

Located in: `runs/{run_id}/results/{gene_id}_results.txt`

Format: `test_acc,params,val_acc,train_time`

### 3. Goodput Metrics (Derived)

- Percentage of genes successfully evaluated per generation
- Calculate with: `python scripts/plot_latency_vs_goodput.py`

---

## Running the Baseline

### Step 1: Start LLM Server

```bash
# In one terminal
sbatch server.sh

# Monitor server
tail -f runs/auto_*/logs/slurm-server-*.out

# Verify it's running
curl http://localhost:8000/health
```

### Step 2: Run Evolution

```bash
# Submit main job
sbatch run.sh

# Monitor progress
tail -f runs/auto_*/logs/slurm-main-*.out
```

### Step 3: Collect Baseline Metrics

```bash
# After run completes, analyze
python scripts/plot_latency_vs_accuracy.py
python scripts/plot_latency_vs_goodput.py

# Metrics are in:
ls runs/auto_*/metrics/
cat runs/auto_*/metrics/latency-*.json
```

---

## What to Measure (Pre-Optimization)

### Latency Metrics
1. **Mean LLM Latency**: Average `_latency_sec` across all requests
2. **P50/P95/P99**: Latency percentiles
3. **Batch Efficiency**: `batch_processing_time / _latency_sec` ratio
4. **Queue Wait Time**: How long requests wait for batching

### Throughput Metrics
1. **Requests/Second**: Total requests / total time
2. **Genes/Hour**: Successful evaluations per hour
3. **Generation Time**: Time to complete one full generation

### Quality Metrics
1. **Goodput**: Percentage of successful evaluations
2. **Accuracy Distribution**: Test accuracy across population
3. **Convergence Speed**: Best fitness per generation

---

## Optimization Techniques (Post-Baseline)

### 1. RAG (Retrieval-Augmented Generation)
**Goal:** Reduce prompt size, improve quality

Implementation ideas:
- Build index of successful architectures
- Retrieve similar genes before generating new ones
- Reduce redundant context in prompts

**Expected Impact:**
- ⬇️ Prompt length (latency reduction)
- ⬆️ Quality of generated architectures
- ➡️ Batch processing time (same, but fewer retries)

### 2. Speculative Decoding
**Goal:** Faster token generation

Implementation ideas:
- Use smaller draft model to predict tokens
- Verify with DeepSeek model in parallel
- Accept correct predictions, reject wrong ones

**Expected Impact:**
- ⬇️ Batch processing time (30-40% faster)
- ⬇️ Mean latency
- ➡️ Quality (unchanged)

### 3. Prompt Optimization
**Goal:** Reduce token count without quality loss

Implementation ideas:
- Compress examples/instructions
- Remove redundant explanations
- Use token-efficient formatting

**Expected Impact:**
- ⬇️ Prompt length
- ⬇️ Latency (proportional to length reduction)
- ⚠️ Quality (must validate)

### 4. Batch Size Optimization
**Goal:** Balance latency vs throughput

Implementation ideas:
- Experiment with different `BATCH_SIZE`
- Measure latency vs batch_size tradeoff
- Find optimal wait time

**Expected Impact:**
- ⬆️ Throughput (with larger batches)
- ⬆️ Per-request latency (if too large)
- Need to find sweet spot

### 5. Parallel Evaluation
**Goal:** Higher throughput

Implementation:
- Set `LOCAL=False` in constants.py
- Each gene submits separate SLURM job
- Evaluate population in parallel

**Expected Impact:**
- ⬆️ Genes/hour (massive parallelization)
- ⬆️ SLURM job overhead
- ⬇️ Time per generation

---

## Experimental Workflow

### Phase 1: Baseline (Current Task)
1. Run with current config (`LOCAL=True`, `BATCH_SIZE=8`, etc.)
2. Collect metrics for 1-2 full runs
3. Calculate mean/median/p95 for all metrics
4. **Document these numbers as your baseline**

### Phase 2: Optimization Experiments
For each technique:
1. Implement changes (RAG, speculative decoding, etc.)
2. Run with same configuration as baseline
3. Collect metrics in new run directory
4. Compare with baseline using plotting scripts

### Phase 3: Analysis
```bash
# Compare runs
python scripts/plot_latency_vs_accuracy.py --run-id baseline_run
python scripts/plot_latency_vs_accuracy.py --run-id rag_optimized_run

# Statistical comparison
python scripts/compare_runs.py baseline_run rag_optimized_run
```

---

## Recommended Baseline Config (Summary)

```python
# src/cfg/constants.py
LOCAL = True                    # Sequential eval, simpler
POPULATION_SIZE = 8             # 8 genes per generation
N_GENERATIONS = 10              # 10 generations
BATCH_SIZE = 8                  # Match population size
MAX_NEW_TOKENS_* = 32000       # Already optimized
LLM_MODEL = 'local_server'     # DeepSeek on pace-ice
```

```bash
# run.sh
#SBATCH --nodes=1              # Single node
#SBATCH --gres=gpu:H100:1      # One GPU
#SBATCH --mem-per-cpu=32G      # 32GB RAM
#SBATCH --time=48:00:00        # Generous time limit
```

---

## Key Files to Monitor

### During Run
- `runs/{run_id}/logs/slurm-main-*.out` - Evolution progress
- `runs/{run_id}/logs/slurm-server-*.out` - LLM server logs

### After Run
- `runs/{run_id}/metrics/latency-*.json` - LLM inference metrics
- `runs/{run_id}/results/*_results.txt` - Model accuracy/params
- `runs/{run_id}/checkpoints/*.pkl` - Population state

### Analysis
- `scripts/plots/latency_vs_accuracy_*.png` - Correlation analysis
- `scripts/plots/latency_vs_goodput_*.png` - Goodput trends

---

## Troubleshooting

### "No metrics collected"
- Ensure server is running: `curl http://localhost:8000/health`
- Check `RUN_ID` env var is exported in `run.sh`
- Verify `server.py` detects `RUN_ID` (not "server-only")

### "Low goodput (many failed evals)"
- Check for errors in `slurm-main-*.out`
- Increase token limits if outputs truncated
- Verify LLM server has sufficient resources

### "Very slow evolution"
- Consider `LOCAL=False` for parallel evals
- Check if H100 is actually being used
- Monitor batch efficiency (should be >90%)

---

## Next Steps After Baseline

1. **Document baseline numbers** (create `docs/BASELINE_METRICS.md`)
2. **Choose first optimization** (RAG recommended - high impact, moderate complexity)
3. **Implement in isolated branch** (e.g., `feat/rag-optimization`)
4. **Run with same config** (only change the optimization technique)
5. **Compare metrics** (use plotting scripts)
6. **Iterate** (refine optimization based on results)

Good luck with your inference optimization research! 🚀
