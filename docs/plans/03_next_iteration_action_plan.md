# Next Iteration Action Plan

**Date:** December 2, 2025
**Sprint Goal:** Validate fitness inheritance fix, implement server optimizations (Batching + Direct HTTP), and launch baseline experiments.

---

## Phase 1: Critical Fixes & Infrastructure (IN PROGRESS)

### 1.1 Server Shutdown Automation ⚡ HIGH PRIORITY
**Goal:** Prevent GPU waste and organize logs properly

**Tasks:**
- [ ] **Add server job tracking to `server.sh`**
  - Write `SLURM_JOB_ID` to `hostname_server_job.txt`
  - Location: After hostname write in `server.sh`
- [ ] **Implement automatic server shutdown in `run.sh`**
  - Add `scancel` logic after main job completes
  - Move server logs to `runs/{RUN_ID}/logs/`
  - Clean up tracking files
- [ ] **Increase server time limit safety net**
  - Change from 16 hours to 72 hours in `server.sh`

### 1.2 Enable Remote Fallback (CODE COMPLETE)
**Goal:** Prevent evolution stall when primary server goes down

**Tasks:**
- [x] **Implement fallback logic in `llm_utils.py`** (Done)
- [ ] **Update `.env` configuration**
  - Set `ENABLE_LLM_REMOTE_FALLBACK=true`
  - Verify `LLM_REMOTE_FALLBACK_TARGET=mixtral_hf`

---

## Phase 2: Optimization & Performance (COMPLETED)

### 2.1 Server Batching ✅ DONE
**Goal:** Improve GPU utilization and throughput.
**Implementation:**
- Modified `server.py` to support batching.
- Configured `BATCH_SIZE=4` (Safe for A100-40GB).
- Added `BATCH_WAIT_TIME=0.5s`.

### 2.2 Direct HTTP Execution ✅ DONE
**Goal:** Eliminate SLURM queue contention for LLM operations.
**Implementation:**
- Modified `run_improved.py` to use `ThreadPoolExecutor` for `llm_mutation` and `llm_crossover` tasks when `LOCAL=False`.
- Bypasses `sbatch` for simple HTTP requests, executing them directly on the manager node.
- Controlled via `LLM_DIRECT_HTTP` constant.

---

## Phase 3: Experiment Configuration & Execution

### 3.1 Baseline Configuration (No RAG)
* **Goal**: Establish baseline latency and accuracy.
* **Config**:
  * `src/cfg/constants.py`:
    * `LOCAL = False` (Enables parallel execution)
    * `RAG_ENABLED = False`
    * `population_size = 16`
  * `server.py`:
    * `BATCH_SIZE = 4`

### 3.2 RAG Configuration
* **Goal**: Measure impact of RAG on latency and code quality.
* **Config**:
  * Same as above, but set `RAG_ENABLED = True` in `src/cfg/constants.py`.

### 3.3 Execution Commands
```bash
# 1. Start Server
export BATCH_SIZE=4
sbatch server.sh

# 2. Start Evolution (Distributed + Direct HTTP)
# Wait for server to start (check hostname.log)
export USE_LOAD_BALANCER=false
sbatch run.sh
```

---

## Phase 4: Monitoring & Analysis

### 4.1 Real-time Fitness Inheritance Monitoring
**Goal:** Confirm the bug fix is working during the run

**Tasks:**
- [ ] **Create monitoring script `scripts/monitor_inheritance.sh`**
  ```bash
  #!/bin/bash
  echo "Monitoring fitness inheritance events..."
  tail -f runs/latest/logs/slurm-main-*.out | grep --line-buffered "Inheriting fitness"
  ```

### 4.2 Post-Run Analysis
**Goal:** Quantify the impact of fitness inheritance and batching.

**Tasks:**
- [ ] **Analyze Goodput**: Use `scripts/plot_latency_vs_goodput.py`.
- [ ] **Analyze Latency**: Use `scripts/analyze_e2e_latency.py`.

---

## Technical FAQ

### VRAM & Batch Size
*   **Model**: DeepSeek-R1-Distill-Qwen-32B
*   **Batch Size**: 4 is recommended for A100-40GB to avoid OOM with large contexts.
*   **Wait Time**: 0.5s is negligible compared to generation time.

### GPU Resources
*   **Evaluation Jobs**: 1 GPU, 16GB RAM (Optimal for ExquisiteNetV2).
*   **Server Job**: 1 GPU, 160GB RAM (Full node/slice).
