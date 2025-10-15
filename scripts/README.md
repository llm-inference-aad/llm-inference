# Analysis Scripts# Plots Directory



This directory contains analysis and visualization scripts for LLMGE runs.This directory contains generated visualization outputs from analysis scripts.



## Available Scripts## Generated Files



### 1. `plot_latency_vs_accuracy.py`- `latency_vs_accuracy_{run_id}.png` - Latency vs accuracy scatter plots

Correlates LLM inference latency with model test accuracy.- `latency_vs_goodput_{run_id}.png` - Goodput over generations plots



**Usage:**## Usage

```bash

# Auto-detect latest runPlots are automatically saved here when running the analysis scripts:

python scripts/plot_latency_vs_accuracy.py

```bash

# Specific run# Plot latency vs accuracy (defaults to latest run)

python scripts/plot_latency_vs_accuracy.py --run-id auto_20251014_191652python scripts/plot_latency_vs_accuracy.py



# Custom output# Plot goodput over generations (defaults to latest run)

python scripts/plot_latency_vs_accuracy.py --output my_plot.pngpython scripts/plot_latency_vs_goodput.py

```

# Specify a particular run

### 2. `plot_latency_vs_goodput.py`python scripts/plot_latency_vs_accuracy.py --run-id auto_20251014_191652

Analyzes goodput (successful evaluations) across generations with latency overlay.

# Custom output location

**Usage:**python scripts/plot_latency_vs_accuracy.py --output my_custom_plot.png

```bash```

# Auto-detect latest run

python scripts/plot_latency_vs_goodput.py## Organization



# Specific run- Plots are organized by run_id in their filenames

python scripts/plot_latency_vs_goodput.py --run-id auto_20251014_191652- Old plots are not automatically deleted (manual cleanup if needed)

```- All plots use high DPI (300) for publication quality


### 3. `analyze_e2e_latency.py`
Comprehensive end-to-end latency analysis with statistics and visualizations.

**Usage:**
```bash
# List available runs
python scripts/analyze_e2e_latency.py --list

# Analyze specific run by run_id
python scripts/analyze_e2e_latency.py --run-id auto_20251014_191652

# Analyze by hash (legacy)
python scripts/analyze_e2e_latency.py abc123def456

# Compare multiple runs
python scripts/analyze_e2e_latency.py --compare hash1 hash2 hash3
```

## Output Location

All plots are saved to `scripts/plots/` by default.

## Requirements

All scripts require the virtual environment to be activated:

```bash
# Traditional venv
source .venv/bin/activate

# Or use uv directly
uv run python scripts/plot_latency_vs_accuracy.py
```

## Data Sources

Scripts automatically search for data in the following order:

1. **New structure:** `runs/{run_id}/metrics/latency-{hash}.json`
2. **Legacy structure:** `metrics/data/-latency-{hash}.json`

Results files: `runs/{run_id}/results/{gene_id}_results.txt`
