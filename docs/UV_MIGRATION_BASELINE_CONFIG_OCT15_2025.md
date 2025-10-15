# UV Migration & Baseline Configuration Summary - October 15, 2025

## Changes Completed

### 1. ✅ UV Migration

**Installed uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Installed to ~/.local/bin/uv
```

**Migrated environment:**
- Backed up old venv: `.venv` → `.venv-backup-20251015`
- Created new venv with uv: `uv venv .venv --python 3.12`
- Installed all dependencies: `uv sync` (completed in ~2.5 minutes!)
- Generated lock file: `uv.lock` (for reproducibility)

**Performance:**
- ⚡ 10-100x faster than pip
- 🔒 Deterministic resolution with uv.lock
- ✅ All 107 packages installed successfully

**Usage:**
```bash
# Activate as usual
source .venv/bin/activate

# Or run without activation
uv run python script.py
```

### 2. ✅ Updated .gitignore

**Added exclusions:**
```
runs/                 # Team-member-specific results/metrics/logs
.venv-backup*/        # Old venv backups
```

**Rationale:**
- Each team member will have their own runs with different results
- Metrics and logs are local to each developer's experiments
- Only code and documentation should be tracked

### 3. ✅ Moved E2E Latency Analysis Script

**Changes:**
- `metrics/e2e-latency.py` → `scripts/analyze_e2e_latency.py`
- Updated to support new metrics structure: `runs/{run_id}/metrics/`
- Added `--run-id` parameter for run-specific analysis
- Support both naming conventions:
  - New: `latency-{hash}.json`
  - Legacy: `-latency-{hash}.json`

**New Usage:**
```bash
# List available runs
python scripts/analyze_e2e_latency.py --list

# Analyze by run_id (new)
python scripts/analyze_e2e_latency.py --run-id auto_20251014_191652

# Analyze by hash (legacy)
python scripts/analyze_e2e_latency.py abc123def456

# Compare runs
python scripts/analyze_e2e_latency.py --compare hash1 hash2
```

### 4. ✅ Added Scripts Documentation

**Created:**
- `scripts/README.md` - Complete usage guide for all analysis tools
- `scripts/plots/README.md` - Description of generated plot files

**Documented:**
- `plot_latency_vs_accuracy.py` - Correlation analysis
- `plot_latency_vs_goodput.py` - Goodput trends
- `analyze_e2e_latency.py` - Comprehensive latency stats
- Data source hierarchy (new vs legacy structure)
- uv run examples

### 5. ✅ Configured Baseline Constants

**Key Changes in `src/cfg/constants.py`:**

```python
# BASELINE CONFIGURATION
LOCAL = False                    # Parallel evaluation (was True)
num_generations = 10             # 10 generations for baseline
start_population_size = 8        # Matches BATCH_SIZE
population_size = 8              # Consistent sizing
```

**Added documentation block:**
- Explains why these settings are recommended
- Notes relationship with BATCH_SIZE in server.py
- Guides for adjusting for optimization experiments

**Rationale:**
- `LOCAL=False`: Parallel gene evaluation for better throughput measurement
- `population_size=8`: Matches `BATCH_SIZE=8` in server.py for efficient batching
- `num_generations=10`: Sufficient for statistical significance

---

## Git Commits Made

### Commit 1: Update .gitignore
- Exclude `runs/` directory (team-specific data)
- Exclude `.venv-backup*` directories
- Maintains uv.lock exclusion

### Commit 2: Move E2E latency script
- Relocated to scripts/ for better organization
- Updated for new metrics structure
- Maintains backwards compatibility

### Commit 3: Add scripts documentation
- Complete usage guide for all tools
- Examples with uv run
- Data source documentation

### Commit 4: Configure baseline constants
- Set LOCAL=False for distributed evaluation
- Document recommended baseline settings
- Explain optimization workflow

---

## Testing Scripts

All scripts now work with uv:

```bash
# Test plotting scripts
uv run python scripts/plot_latency_vs_accuracy.py --help
uv run python scripts/plot_latency_vs_goodput.py --help
uv run python scripts/analyze_e2e_latency.py --help

# Or activate venv first
source .venv/bin/activate
python scripts/plot_latency_vs_accuracy.py
```

---

## Next Steps for Baseline Run

### 1. Start LLM Server
```bash
sbatch server.sh
# Monitor: tail -f runs/auto_*/logs/slurm-server-*.out
```

### 2. Run Evolution
```bash
sbatch run.sh
# Monitor: tail -f runs/auto_*/logs/slurm-main-*.out
```

### 3. Expected Behavior with LOCAL=False

**During Run:**
- Main job: `slurm-main-{job_id}.out` (evolution loop)
- Gene evaluations: `eval-{job_id}.out` (one per gene, submitted as SLURM jobs)
- All jobs run in parallel (8 genes = 8 concurrent SLURM jobs)

**After Run:**
- All logs migrated to `runs/{run_id}/logs/`
- Metrics in `runs/{run_id}/metrics/latency-{hash}.json`
- Results in `runs/{run_id}/results/{gene_id}_results.txt`

### 4. Analyze Results
```bash
# Latency vs accuracy
uv run python scripts/plot_latency_vs_accuracy.py

# Goodput analysis
uv run python scripts/plot_latency_vs_goodput.py

# Detailed latency analysis
uv run python scripts/analyze_e2e_latency.py --run-id latest

# Plots saved to scripts/plots/
```

### 5. Document Baseline Metrics

Create `docs/BASELINE_METRICS.md` with:
- Mean/median/P95 latency
- Throughput (genes/hour, requests/sec)
- Goodput percentage
- Batch efficiency
- Best/worst gene latencies

This becomes your reference point for comparing optimization techniques!

---

## Configuration Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `LOCAL` | `False` | Parallel evaluation (distributed SLURM) |
| `num_generations` | `10` | Baseline statistical significance |
| `population_size` | `8` | Matches server BATCH_SIZE |
| `start_population_size` | `8` | Consistent with population |
| `BATCH_SIZE` (server.py) | `8` | Efficient batching |
| `MAX_NEW_TOKENS` | `32000` | Optimized for DeepSeek 130k window |
| `LLM_MODEL` | `'local_server'` | DeepSeek-R1-Distill-Qwen-32B |

---

## Benefits of This Configuration

1. **Parallel Execution**: All 8 genes evaluated simultaneously
2. **Efficient Batching**: Population size matches batch size
3. **Better Metrics**: Parallel execution reveals true throughput
4. **Realistic Baseline**: Represents production-like parallel workload
5. **Easy Comparison**: Consistent setup for optimization experiments

---

## Optimization Experiments Workflow

After baseline is established:

1. **Branch for experiment:**
   ```bash
   git checkout -b feat/rag-optimization
   ```

2. **Implement technique** (RAG, speculative decoding, etc.)

3. **Run with same config** (same population_size, generations)

4. **Compare metrics:**
   ```bash
   python scripts/analyze_e2e_latency.py --compare baseline_hash optimized_hash
   ```

5. **Document improvements** in run-specific README

6. **Iterate and refine**

---

## Files Modified

- ✅ `.gitignore` - Exclude runs/ and backups
- ✅ `metrics/e2e-latency.py` → `scripts/analyze_e2e_latency.py`
- ✅ `scripts/README.md` - New comprehensive guide
- ✅ `scripts/plots/README.md` - Updated descriptions
- ✅ `src/cfg/constants.py` - Baseline configuration

## Files Created

- ✅ `uv.lock` - Dependency lock file (not tracked)
- ✅ `.venv/` - New uv-managed virtual environment (not tracked)

---

## Team Notes

- ⚠️ **Do not commit runs/ directory** - Each member has their own experiments
- ✅ **uv is installed locally** - Each member should install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- ✅ **uv.lock is excluded** - Regenerated from pyproject.toml with `uv sync`
- 📊 **Share plots/docs only** - Results in papers/reports, not git

---

Ready to establish your baseline! 🚀
