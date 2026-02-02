# 02 - System Architecture & Workflow

This document details the core architecture of the LLMGE system, including the execution model, run management, and metrics collection.

## 1. Two-Tier Execution Model

The system uses a two-tier architecture for executing the evolution.

```
┌─────────────────────────────────────────────────────────────┐
│ SLURM Job (sbatch run.sh)                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Main Evolution Loop (run_improved.py)                   │ │
│ │ - Population management & genetic operators             │ │
│ │ - Fitness evaluation orchestration                      │ │
│ │                                                         │ │
│ │   For each gene, the system decides how to evaluate:    │ │
│ │   ┌───────────────────────────────────────────────────┐ │ │
│ │   │ if LOCAL=True:                                    │ │ │
│ │   │   `bash train.sh` (Runs on the same SLURM node)   │ │ │
│ │   │ else:                                             │ │ │
│ │   │   `sbatch train.sh` (Submits a new SLURM job)     │ │ │
│ │   └───────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key Point**: The `LOCAL` flag in `src/cfg/constants.py` controls **how individual gene evaluations are distributed**, not whether the main evolution job runs on SLURM.

- **`LOCAL = True` (Default for baselining)**: Simpler, faster startup, and easier debugging. All evaluation output appears in the main `slurm-main-*.out` log.
- **`LOCAL = False` (For massive parallelization)**: Submits each evaluation as a separate job. Generates many `eval-*.out` logs.

## 2. Automated Run Management

To ensure reproducibility and organization, every experiment is managed within a self-contained, timestamped directory.

### Directory Structure

All experimental artifacts are stored under the `runs/` directory.

```
runs/
├── latest -> auto_20251015_120000  (A symlink to the most recent run)
│
└── auto_20251015_120000/
    ├── checkpoints/               # Evolution checkpoints (*.pkl)
    ├── results/                   # Final model results (*_results.txt)
    ├── metrics/                   # Latency and performance data (*.json)
    ├── logs/                      # All SLURM logs for the run (*.out, *.err)
    └── run_metadata.json          # Git commit, timestamp, etc.
```

### Automated Workflow

1.  **Execution**: `sbatch run.sh` initiates a new run.
2.  **Directory Creation**: The script automatically creates a new `runs/auto_YYYYMMDD_HHMMSS/` directory.
3.  **Environment Variable**: It exports the `RUN_DIR` environment variable, making the path available to all subprocesses.
4.  **Artifact Storage**:
    - `run_improved.py` saves checkpoints to `checkpoints/`.
    - `train.py` saves model results to `results/`.
    - `server.py` saves latency metrics to `metrics/`.
5.  **Log Migration**: At the end of the run, `run.sh` moves all associated SLURM logs from the global `slurm-results/` directory into the run-specific `logs/` folder.

## 3. Metrics Collection

The system is designed for comprehensive end-to-end latency tracking.

- **`server.py`**: The FastAPI server detects the `RUN_ID` environment variable and writes detailed latency metrics for each inference request to `runs/{RUN_ID}/metrics/latency-{hash}.json`.
- **Data Format**: Each JSON file includes the `run_id`, server session hash, and a list of requests, each tagged with its corresponding `gene_id`.
- **Backward Compatibility**: If `RUN_ID` is not set (e.g., when running the server standalone), metrics are written to the legacy `metrics/data/` directory. Analysis scripts are designed to check both locations.
