# Pull Request Summary: `feat/surya`

This pull request introduces a massive overhaul of the LLMGE project, focusing on stability, reproducibility, developer experience, and preparing the ground for systematic inference optimization experiments. It transforms the repository from a collection of experimental scripts into a robust, well-documented framework.

---

## ✨ Major Features & Enhancements

### 1. Comprehensive Documentation Overhaul
- **Consolidated 15+ temporary documents** into a coherent, five-part guide for new team members, covering everything from introduction and setup to architecture and development best practices.
- **New Documentation Suite**:
  - `docs/00_introduction.md`
  - `docs/01_setup.md`
  - `docs/02_architecture.md`
  - `docs/03_running_experiments.md`
  - `docs/04_development_guide.md`
- **Added `scripts/README.md`** to document the usage of all analysis scripts.

### 2. Python Environment Migration to `uv`
- **Migrated the entire Python environment** from `venv` + `pip` to the high-performance `uv` package manager.
- **Benefits**:
  - **~10-100x faster** dependency installation.
  - **Deterministic builds** via `uv.lock` (though not tracked in git).
  - **Simplified workflow** with `uv sync` and `uv run`.
- The `.gitignore` file was updated to exclude `uv.lock` and `.venv-backup*/` directories.

### 3. Automated Run Management System
- **Introduced a self-contained run structure**. Every experiment is now stored in a unique, timestamped directory: `runs/auto_YYYYMMDD_HHMMSS/`.
- **Each run directory encapsulates**:
  - `checkpoints/`: Evolution checkpoints.
  - `results/`: Final model results.
  - `metrics/`: End-to-end latency and performance data.
  - `logs/`: All SLURM logs (`.out` and `.err`) associated with the run.
- **Fully Automated**: The `run.sh` script handles directory creation, environment variable propagation (`RUN_DIR`), and log migration, ensuring zero manual effort is needed to keep experiments organized.

### 4. End-to-End Metrics & Analysis Scripts
- **Run-Specific Metrics**: The `server.py` now detects the `RUN_ID` and saves detailed latency metrics to the corresponding `runs/{run_id}/metrics/` directory.
- **Enhanced Analysis Scripts**:
  - All plotting scripts (`plot_latency_vs_accuracy.py`, `plot_latency_vs_goodput.py`) were updated to automatically find data in the new run structure.
  - **"Latest" Run Default**: Scripts now default to analyzing the **latest run**, making them much more convenient.
  - **Auto-detect Metrics Hash**: The scripts automatically find the most recent metrics file, removing a tedious manual step.
  - **Standardized Outputs**: Plots are now saved to `scripts/plots/` with run-specific filenames (e.g., `latency_vs_accuracy_{run_id}.png`).
  - A legacy analysis script was moved to `scripts/analyze_e2e_latency.py` and significantly improved to support the new run-based analysis.

### 5. Baseline Configuration for LLMGE Optimization
- **Configured `src/cfg/constants.py`** for a "strong LLMGE baseline" run.
- **Key Settings**:
  - `LOCAL = False`: Enables true parallelism by evaluating each gene in a separate SLURM job.
  - `num_generations = 10` and `population_size = 8`: Provides a solid sample of ~80 evaluations for robust statistical analysis.
- **Increased Generation Count**: The default `num_generations` was increased from 1 to 10, enabling more meaningful experiments out-of-the-box.
- This provides a standardized starting point for measuring the impact of future optimizations like RAG or speculative decoding.

---

## 🐛 Bug Fixes & Stability Improvements

### 1. Critical Path & SLURM Log Fixes
- **Absolute Paths**: Fixed a critical bug where `RUN_DIR` was a relative path, causing results to be written to incorrect, nested locations. `RUN_DIR` is now an absolute path, ensuring stability.
- **SLURM Log Migration**: The `run.sh` script now automatically moves all SLURM logs into the run-specific `logs/` directory upon completion, making each run fully self-contained.
- **Working Directory Consistency**: Ensured that all training scripts execute from the repository root, preventing path-related errors.
- **Robust `check4results()`**: The fitness checking function was updated to search multiple possible locations for results files, making it resilient to path issues and backward compatible.

### 2. LLM & Evolution Robustness
- **Removed Hard Token Cap**: Fixed a critical stability issue by removing the hard-coded `2048` token limit on LLM responses. This prevents the model from returning truncated, invalid code and was a major source of `SyntaxError`.
- **Restored Mutant Prompt Templates**: Corrected the prompt templates to provide the LLM with a clearer, code-centric context, improving the quality of generated code.

---

## 📝 Minor Changes & Housekeeping

- Updated `.gitignore` to exclude the `runs/` directory, preventing team-specific experimental data from being committed.
- Centralized all SLURM job logging into a temporary `slurm-results/` directory before being migrated to a run-specific `logs/` folder.
- Updated the model path to use the `DeepSeek-R1-Distill-Qwen-32B` model.
