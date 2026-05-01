# Analysis Scripts

This directory contains CRUD and data viz scripts for LLMGE runs.

## Utility Scripts

### `migrate_slurm_logs.sh` 🛠️

Manually migrate SLURM logs to a specific run directory when automatic migration fails (e.g., due to job timeout or cancellation).

**Usage:**

```bash
# With job ID
./scripts/migrate_slurm_logs.sh auto_20251015_180811 3403227

# Auto-detect job ID from metadata
./scripts/migrate_slurm_logs.sh auto_20251015_180811
```

**When to use:**
- Job was cancelled due to time limit
- Job failed before log migration step
- Manual cleanup after debugging

### `test_vllm_sbatch_smoke.py` ✅

End-to-end smoke test for vLLM through Slurm.

It submits `server.sh` via `sbatch`, polls readiness, sends a `/generate` request, verifies run-scoped metrics, and confirms vLLM startup markers in logs.

**Usage:**

```bash
python scripts/test_vllm_sbatch_smoke.py --repo-root .
```

Optional:

```bash
python scripts/test_vllm_sbatch_smoke.py --keep-server
```

## Available Scripts

### 1. `retries_vs_generation.py` 📊

Analyzes and plots retry attempts and failures per generation from SLURM logs.

**Features:**
- Counts LLM retry attempts per generation
- Optionally tracks final failures after all retries exhausted
- Supports "latest" run or specific run_id
- Exports plots to file or displays interactively

**Usage:**

```bash
# Analyze latest run
python scripts/retries_vs_generation.py latest

# Analyze specific run
python scripts/retries_vs_generation.py auto_20251103_210131

# Include final failures in the plot
python scripts/retries_vs_generation.py latest --include-failures

# Save plot to file
python scripts/retries_vs_generation.py latest --output retries.png

# Full example
python scripts/retries_vs_generation.py latest --include-failures --output plots/retries.png
```

**Output:**
- Console: Statistics table showing retries (and failures) per generation
- Plot: Bar chart of retry counts vs generation number

### 2. `accuracy_vs_generation.py`

Plots average accuracy vs generation number for evolution runs.

**Usage:**

```bash
# Analyze latest run
python scripts/accuracy_vs_generation.py latest

# Analyze specific run
python scripts/accuracy_vs_generation.py auto_20251103_210131
```

### 3. `plot_run_summary.py` ⭐ **NEW**

Generates comprehensive summary visualizations from a run's results and LLM metrics.

**Outputs:**
- `run_summary.png` - 4-panel figure with accuracy distributions, training time analysis, and statistics
- `llm_metrics.png` - 4-panel figure with LLM latency, prompt length analysis, and batching stats

**Usage:**

```bash
# Auto-detect latest run
uv run python scripts/plot_run_summary.py

# Specific run
uv run python scripts/plot_run_summary.py --run-dir runs/auto_20251014_191652
```

### 4. `plot_latency_vs_accuracy.py`

Correlates LLM inference latency with model test accuracy for gene-level analysis.



**Usage:**

```bash
# Auto-detect latest run
uv run python scripts/plot_latency_vs_accuracy.py

# Specific run
uv run python scripts/plot_latency_vs_accuracy.py --run-id auto_20251014_191652

# Custom output
uv run python scripts/plot_latency_vs_accuracy.py --output my_plot.png
```

### 5. `plot_latency_vs_goodput.py`

Analyzes goodput (successful evaluations) across generations with latency overlay.

**Usage:**

```bash
# Auto-detect latest run
uv run python scripts/plot_latency_vs_goodput.py

# Specific run
uv run python scripts/plot_latency_vs_goodput.py --run-id auto_20251014_191652
```

### 6. `plot_pareto_enhanced.py`

Plots Pareto fronts for multi-objective optimization (accuracy vs. model size).

**Usage:**

```bash
# Plot latest run
uv run python scripts/plot_pareto_enhanced.py

# Plot specific run
uv run python scripts/plot_pareto_enhanced.py --run-id auto_20251014_191652

# Compare multiple runs
uv run python scripts/plot_pareto_enhanced.py --compare run1 run2 run3

# Custom output
uv run python scripts/plot_pareto_enhanced.py --output my_pareto.png
```

### 7. `analyze_e2e_latency.py`


### 5. `analyze_e2e_latency.py`

Comprehensive end-to-end latency analysis with statistics and visualizations.

**Usage:**

```bash
# List available runs
uv run python scripts/analyze_e2e_latency.py --list

# Analyze specific run by run_id
uv run python scripts/analyze_e2e_latency.py --run-id auto_20251014_191652

# Analyze by hash (legacy)
uv run python scripts/analyze_e2e_latency.py abc123def456

# Compare multiple runs
uv run python scripts/analyze_e2e_latency.py --compare hash1 hash2 hash3
```

## Quick Start

For a quick overview of your latest run, use:

```bash
uv run python scripts/plot_run_summary.py
```

This will generate comprehensive visualizations without needing to specify any arguments.

## Output Location

All plots are saved to `scripts/plots/` by default.

## Data Sources

Scripts automatically search for data in the following order:

1. **New structure:** `runs/{run_id}/metrics/latency-{hash}.json`
2. **Legacy structure:** `metrics/data/-latency-{hash}.json`

Results files: `runs/{run_id}/results/{gene_id}_results.txt`
