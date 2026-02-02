# 03 - Running Experiments & Analysis

This guide provides instructions for launching experiments and analyzing the results.

## 1. Recommended Baseline Configuration

To establish a strong performance baseline for LLMGE, the following parameters are recommended in `src/cfg/constants.py`.

```python
# src/cfg/constants.py

# --- Recommended Baseline Configuration ---
# Set LOCAL to False to enable distributed evaluation, where each gene is
# evaluated in a separate SLURM job. This provides true parallelism.
LOCAL = False

# A generation count of 10 with a population of 8 provides a solid
# sample of ~80 evaluations for robust statistical analysis.
num_generations = 10
population_size = 8

# The tournament size for parent selection.
TOURNSIZE = 3

# Crossover and mutation probabilities.
CXPB = 0.3
MUTPB = 0.5
# --- End Recommended Baseline Configuration ---
```

## 2. Launching a Run

The entire workflow is launched with two commands.

1.  **Start the LLM Inference Server**:
    This command submits a job that loads the model and runs the FastAPI server. Monitor its progress to ensure the model loads successfully before proceeding.
    ```bash
    sbatch server.sh
    ```
    *Watch the log:* `tail -f slurm-results/slurm-server-*.out`

2.  **Start the Evolution Run**:
    Once the server is ready, launch the main evolution job. This will automatically create a new run directory and begin the process.
    ```bash
    sbatch run.sh
    ```
    *Watch the log:* `tail -f slurm-results/slurm-main-*.out` (initially, then it moves to `runs/{run_id}/logs/`)

### Continuing a Previous Run

To resume an experiment from its last checkpoint, set the `RUN_ID` environment variable before submitting.

```bash
export RUN_ID=auto_20251015_120000  # Use the ID of the run you want to continue
sbatch run.sh
```

## 3. Analyzing Results

The `scripts/` directory contains powerful, automated analysis tools.

**Note**: Ensure your virtual environment is active (`source .venv/bin/activate`) or use `uv run`.

### `analyze_e2e_latency.py`
Provides comprehensive end-to-end latency statistics.

```bash
# List all available runs
uv run python scripts/analyze_e2e_latency.py --list

# Analyze the latest run automatically
uv run python scripts/analyze_e2e_latency.py --run-id latest

# Analyze a specific run by its ID
uv run python scripts/analyze_e2e_latency.py --run-id auto_20251015_120000
```

### `plot_latency_vs_accuracy.py`
Correlates LLM inference latency with final model test accuracy.

```bash
# Plot the latest run automatically
uv run python scripts/plot_latency_vs_accuracy.py

# Plot a specific run
uv run python scripts/plot_latency_vs_accuracy.py --run-id auto_20251015_120000
```

### `plot_latency_vs_goodput.py`
Analyzes the percentage of successful evaluations per generation.

```bash
# Plot the latest run automatically
uv run python scripts/plot_latency_vs_goodput.py
```

All generated plots are saved to the `scripts/plots/` directory, named with the corresponding `run_id`.
