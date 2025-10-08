# Pareto Front Visualization and Log Organization

This note captures the additions made while wiring up the Pareto front workflow and cleaning up Slurm outputs.

## Overview
- Added `scripts/plot_pareto.py` to aggregate `{gene}_results.txt` files written by `sota/ExquisiteNetV2/train.py` and render a Pareto front (maximize test accuracy, minimize parameter count).
- Introduced a shared `results/slurm/` directory (`src/cfg/constants.py`) and routed every Slurm batch script through it so logs are no longer scattered across the repo.
- Expanded project dependencies with `matplotlib>=3.9.0` (see `pyproject.toml`) for plotting support.

## Producing Results
1. Ensure the virtual environment is active and the project is installed (e.g., `source $VENV_PATH/bin/activate` followed by `pip install -e .`).
2. Launch the evolutionary loop locally with `python run_improved.py first_test`, or submit the cluster wrapper `sbatch server.sh` and `sbatch run.sh`. Either path trains candidate networks via `sota/ExquisiteNetV2/train.py`, which drops comma-separated metrics into `sota/ExquisiteNetV2/results/`.
3. Once at least one `{gene}_results.txt` exists, continue to visualization.

## Plotting the Pareto Front
```
# Basic usage
python scripts/plot_pareto.py --results-dir sota/ExquisiteNetV2/results --output pareto_front.png

# Preview interactively (requires display forwarding or local run)
python scripts/plot_pareto.py --results-dir sota/ExquisiteNetV2/results --output pareto_front.png --show
```

The script will:
- Load each file, interpret the four metrics (test accuracy, total parameters, validation accuracy, training time).
- Mark all models as light-blue points (parameters are shown on a log10 scale).
- Identify and highlight the non-dominated set in orange, draw the connecting curve, and annotate the gene suffix for quick reference.
- Save the resulting figure to the path provided via `--output`.

If the results directory is missing or empty, the tool exits gracefully after printing a message.

## Centralized Slurm Logs
- `src/cfg/constants.py` now defines `SLURM_LOG_DIR = results/slurm/` and guarantees the directory exists at import time.
- Both Slurm templates (`PYTHON_BASH_SCRIPT_TEMPLATE`, `LLM_BASH_SCRIPT_TEMPLATE`) now emit `--output` and `--error` directives pointing to `results/slurm/`.
- `run.sh` and `server.sh` mirror the behavior by placing their job logs under the same folder and creating it as needed.

This structure keeps the repository root clean and makes it easier to archive or purge Slurm logs without touching model artifacts.

## Troubleshooting
- **No plot generated:** verify that evaluation jobs completed successfully and produced files inside `sota/ExquisiteNetV2/results/`.
- **Matplotlib import error:** rerun `pip install -e .` (or `pip install matplotlib`) inside the virtual environment to pick up the new dependency.
- **Log location:** inspect `results/slurm/` for both the evolution loop logs (`slurm-main-*.out/err`) and the underlying LLM/evaluation jobs (`llm-*.out`, `eval-*.out`).
