# PACE-ICE Workflow Guide: Commands & Concepts

**Author:** Surya Atmuri  
**Date:** October 9, 2025  
**Purpose:** Clarify common commands and concepts for the LLM-GE workflow on PACE-ICE

---

## Command Deep Dives

### 1. `tail -f slurm-results/slurm-server-*.out`

**What it does:**
- **`tail`** - Display the last 10 lines of a file (by default)
- **`-f`** - "Follow" mode: continuously monitors the file for new content
- **`slurm-results/slurm-server-*.out`** - Shell glob pattern that matches files like `slurm-server-3275695.out`

**Combined effect:**
Displays the last 10 lines of your server log file and keeps updating in real-time as new lines are written. Think of it like "streaming" the log output to your terminal.

**Example output you'll see:**
```
[2025-10-09 14:32:15] Model weights loaded in 45.23 seconds
[2025-10-09 14:32:18] Tokenizer loaded in 2.15 seconds
[2025-10-09 14:32:20] ===== MODEL LOADING COMPLETED =====
[2025-10-09 14:32:20] Server is ready to serve requests!
Request received for gene xXx01 at 14:35:10
Processing batch of 3 requests
Request completed in 5.23s (E2E), 3.45s (batch processing)
```

**How to exit:**
- Press `Ctrl+C` to stop following and return to your prompt

**Common variations:**
```bash
# Show last 50 lines instead of 10, then follow
tail -n 50 -f slurm-results/slurm-server-*.out

# Follow multiple log files simultaneously
tail -f slurm-results/slurm-*.out

# Follow with timestamps for when you're viewing each line
tail -f slurm-results/slurm-server-*.out | while read line; do echo "$(date '+%H:%M:%S') $line"; done
```

---

### 2. `grep` - Global Regular Expression Print

**What it is:**
A powerful text search utility that scans files for lines matching a pattern.

**Basic syntax:**
```bash
grep [OPTIONS] PATTERN [FILE...]
```

**Common use cases in our workflow:**

#### a) Search for specific text in logs
```bash
# Find all instances of "Fallback to parent"
grep "Fallback to parent" slurm-results/slurm-main-*.out

# Output:
# slurm-results/slurm-main-3275696.out:Fallback to parent code triggered
# slurm-results/slurm-main-3275696.out:Fallback to parent code triggered
```

#### b) Count occurrences with `-c`
```bash
# Count how many times LLM fell back to parent
grep -c "Fallback to parent" slurm-results/slurm-main-*.out

# Output:
# slurm-results/slurm-main-3275696.out:12
```

#### c) Show line numbers with `-n`
```bash
# Find errors with line numbers
grep -n "SyntaxError" slurm-results/slurm-main-*.out

# Output:
# slurm-results/slurm-main-3275696.out:1466:    SyntaxError: invalid syntax
```

#### d) Case-insensitive search with `-i`
```bash
# Find "error" regardless of capitalization
grep -i "error" slurm-results/slurm-main-*.out
```

#### e) Invert match with `-v` (show lines that DON'T match)
```bash
# Show all non-error lines
grep -v "ERROR" slurm-results/slurm-server-*.out
```

#### f) Search recursively with `-r`
```bash
# Search all files in a directory tree
grep -r "def augment_network" src/
```

#### g) Combine with pipes for powerful filtering
```bash
# Find validation errors and count by type
grep "validation error:" slurm-results/slurm-main-*.out | cut -d: -f3 | sort | uniq -c

# Output:
#   5 SyntaxError
#   3 NameError
#   2 IndentationError
```

**Pro tips:**
```bash
# Use quotes if pattern has spaces
grep "Model is ready" slurm-results/slurm-server-*.out

# Use regex patterns
grep "gene_[0-9]\{3\}" slurm-results/slurm-main-*.out  # Matches gene_001, gene_123, etc.

# Color highlight matches for easier reading
grep --color=always "ERROR" slurm-results/*.out

# Show 3 lines of context before and after match
grep -C 3 "failed validation" slurm-results/slurm-main-*.out
```

---

### 3. SSH to Server Node for GPU Monitoring

**Background:**
When you submit a Slurm job with `sbatch server.sh`, the scheduler assigns your job to a **compute node** (e.g., `sched-ice-5-1.pace.gatech.edu`). To check GPU utilization on that node, you need to SSH into it.

**Step-by-step process:**

#### Step 1: Find which node your job is running on
```bash
squeue -u $USER

# Output:
#  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
# 327569    ice-qo LLMGE01_  satmuri R    1:23:45      1 sched-ice-5-1
```
Look at the `NODELIST` column → `sched-ice-5-1`

#### Step 2: SSH to that node
```bash
ssh sched-ice-5-1

# Or with full hostname:
ssh sched-ice-5-1.pace.gatech.edu
```

**Note:** You can SSH to compute nodes only while your job is running on them.

#### Step 3: Monitor GPU utilization
```bash
# One-time snapshot
nvidia-smi

# Output shows:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-80GB    On   | 00000000:07:00.0 Off |                    0 |
# | N/A   42C    P0    65W / 300W |  45123MiB / 81920MiB |     87%      Default |
# +-------------------------------+----------------------+----------------------+

# Continuous monitoring (refreshes every 2 seconds)
watch -n 2 nvidia-smi

# More compact view with just GPU utilization
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Monitor specific metrics over time
nvidia-smi dmon -s puct -c 100  # 100 samples of power, utilization, clock, temperature
```

**Key metrics to watch:**

| Metric | What it Means | Good Value |
|--------|---------------|------------|
| **GPU-Util** | % of GPU cores active | >80% during inference |
| **Memory-Usage** | VRAM used/total | Near total = fully loaded |
| **Temp** | GPU temperature (°C) | 40-80°C normal |
| **Pwr:Usage** | Current power draw | High during compute |

**Exit SSH:**
```bash
exit  # or press Ctrl+D
```

**Shortcut command (one-liner):**
```bash
# Find node and SSH in one command
ssh $(squeue -u $USER -o "%N" | tail -1)
```

**Alternative: Monitor from login node**
```bash
# Some clusters allow monitoring without SSH
scontrol show job 327569 | grep "NodeList"
```

---

### 4. Smoke Test - Definition & Purpose

**Definition:**
A **smoke test** is a quick, preliminary test to verify that the most critical functions of a system are working correctly before running comprehensive tests. The name comes from electronics testing: "If you turn it on and smoke comes out, it's broken."

**Characteristics:**
- **Fast:** Should complete in seconds/minutes, not hours
- **Shallow:** Tests basic functionality, not edge cases
- **Binary:** Pass/fail result - system works or doesn't
- **Automated:** Should be scriptable and repeatable

**For our LLM-GE workflow, a smoke test checks:**

1. ✅ **Environment loads correctly**
   ```bash
   module load python/3.12.5 && echo "✅ Python loaded"
   ```

2. ✅ **Dependencies are installed**
   ```bash
   uv run python -c "import torch; import transformers; print('✅ Core deps OK')"
   ```

3. ✅ **Configuration files exist**
   ```bash
   test -f .env && echo "✅ .env exists"
   test -f hostname.log && echo "✅ hostname.log exists" || echo "⚠️  hostname.log missing"
   ```

4. ✅ **Server is reachable**
   ```bash
   curl -s http://$(cat hostname.log):8000/ | grep "LLM API is running"
   ```

5. ✅ **LLM can generate code**
   ```bash
   uv run python -c "from src.llm_utils import submit_local_server; submit_local_server('print(1+1)', max_new_tokens=10)"
   ```

**Example smoke test script:**
```bash
#!/bin/bash
# smoke_test.sh - Quick sanity check for LLM-GE

set -e  # Exit on any error

echo "🔍 Running smoke test..."

# Test 1: Environment
echo "1/5 Testing environment..."
module load python/3.12.5
export PATH="$HOME/.local/bin:$PATH"

# Test 2: Dependencies
echo "2/5 Testing dependencies..."
uv run python -c "import torch, transformers; print('  ✅ Core imports OK')"

# Test 3: Configuration
echo "3/5 Testing configuration..."
test -f .env || { echo "  ❌ .env missing"; exit 1; }
test -f hostname.log || { echo "  ❌ Server not started"; exit 1; }
echo "  ✅ Config files present"

# Test 4: Server health
echo "4/5 Testing server..."
SERVER_URL="http://$(cat hostname.log):8000/"
curl -sf "$SERVER_URL" > /dev/null || { echo "  ❌ Server not responding"; exit 1; }
echo "  ✅ Server is alive"

# Test 5: Code compilation
echo "5/5 Testing code compilation..."
python -m compileall -q src || { echo "  ❌ Code has syntax errors"; exit 1; }
echo "  ✅ Code compiles"

echo ""
echo "✅ All smoke tests passed! System is operational."
```

**Usage:**
```bash
# Run before submitting production jobs
bash smoke_test.sh

# Output:
# 🔍 Running smoke test...
# 1/5 Testing environment...
# 2/5 Testing dependencies...
#   ✅ Core imports OK
# 3/5 Testing configuration...
#   ✅ Config files present
# 4/5 Testing server...
#   ✅ Server is alive
# 5/5 Testing code compilation...
#   ✅ Code compiles
# 
# ✅ All smoke tests passed! System is operational.
```

**When to run smoke tests:**
- ✅ After pulling new code changes
- ✅ Before submitting long-running jobs
- ✅ After modifying environment setup
- ✅ When debugging "it was working yesterday" issues
- ✅ As part of CI/CD pipeline (if applicable)

---

## Quick Reference Card

```bash
# Monitor server startup in real-time
tail -f slurm-results/slurm-server-*.out

# Count fallbacks to parent code
grep -c "Fallback to parent" slurm-results/slurm-main-*.out

# Find which node is running your job
squeue -u $USER -o "%N" | tail -1

# SSH to compute node and monitor GPU
ssh $(squeue -u $USER -o "%N" | tail -1)
nvidia-smi

# Quick smoke test
module load python/3.12.5 && \
export PATH="$HOME/.local/bin:$PATH" && \
uv run python -c "import torch; print('✅ Ready')"
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Slurm** | Workload manager for HPC clusters (schedules jobs on compute nodes) |
| **Compute Node** | Server in the cluster that runs your job (has GPUs) |
| **Login Node** | Server where you SSH first (NO GPUs, don't run jobs here) |
| **Job ID** | Unique number assigned to your submitted job (e.g., 3275695) |
| **Batch Job** | Non-interactive job submitted via `sbatch` |
| **Glob Pattern** | Wildcard pattern (e.g., `*.out` matches all .out files) |
| **Pipe (`|`)** | Sends output of one command as input to another |
| **Grep** | Search text tool ("Global Regular Expression Print") |
| **SSH** | Secure Shell - remote login protocol |
| **nvidia-smi** | NVIDIA System Management Interface (GPU monitoring tool) |

---

## Pro Tips

1. **Combine `tail` with `grep` for filtered live logs:**
   ```bash
   tail -f slurm-main-*.out | grep --line-buffered "ERROR\|Fallback"
   ```

2. **Use `less +F` as alternative to `tail -f`:**
   ```bash
   less +F slurm-server-*.out  # Press Ctrl+C to stop following, then 'q' to quit
   ```

3. **Set up aliases in `~/.bashrc` for common tasks:**
   ```bash
   alias llmlog='tail -f slurm-results/slurm-main-*.out'
   alias srvlog='tail -f slurm-results/slurm-server-*.out'
   alias mygpu='ssh $(squeue -u $USER -o "%N" | tail -1) nvidia-smi'
   ```

4. **Use `screen` or `tmux` to keep monitoring sessions alive:**
   ```bash
   screen -S monitoring
   tail -f slurm-results/*.out
   # Press Ctrl+A, then D to detach
   # Later: screen -r monitoring to reattach
   ```

---

**Questions or need clarification? Add notes below:**

---
