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

## 🐛 Critical Bug Fixes & Stability Improvements

This release includes several critical, blocking bug fixes that significantly improve the stability and correctness of the framework, especially for distributed runs.

### 1. ✅ SLURM Log Polling & Pathing (Blocking)
- **Problem**: The system was unable to track the status of distributed jobs (`LOCAL = False`) because it was looking for SLURM logs in the wrong directory.
- **Fix**:
    - **Correct Log Polling**: Modified `run_improved.py` to correctly poll for logs in the `slurm-results/` directory, unblocking all distributed runs.
    - **Absolute `RUN_DIR` Path**: Ensured `RUN_DIR` is an absolute path to prevent results from being written to incorrect, nested locations.
    - **Automated Log Migration**: The `run.sh` script now automatically moves all SLURM logs from `slurm-results/` into the run-specific `runs/{run_id}/logs/` directory upon completion.

### 2. ✅ LLM Prompt & Token Limit Fixes (Blocking)
- **Problem**: Malformed prompts and excessively high token limits were causing API errors and producing poor-quality, truncated code.
- **Fix**:
    - **Restored Roleplay Prompts**: Restored the 5 `mutant*.txt` templates to their correct persona-based format, improving mutation quality.
    - **Corrected Token Limits**: Reduced the default `max_new_tokens` from 32,000 to `8192` for the local DeepSeek server and `4096` for Hugging Face endpoints, preventing API errors and improving performance.

### 3. ✅ Probabilistic Quality Control (Major)
- **Problem**: The `PROB_QC` constant was non-functional, as the decision to run a quality control check was not being sampled for each job.
- **Fix**: Re-implemented probabilistic sampling in `run_improved.py`. The system now correctly uses the `PROB_QC` value to determine whether to run a QC check for each mutation and crossover operation.

### 5. Server-Side Request Batching
- **Problem**: The inference server processed requests one by one, leading to inefficient GPU utilization and lower throughput.
- **Fix**: Implemented an `asyncio` request queue in `server.py`. The server now collects incoming requests and processes them in batches, significantly improving inference throughput and overall system performance.

### 4. Robust Fallback & Retry Logic
- **Problem**: A single invalid code generation from the LLM (e.g., a `SyntaxError`) could crash an entire evolution run.
- **Fix**: Implemented a multi-layered validation and retry system in `src/llm_utils.py`.
    - **3-Layer Validation**: All generated code is validated for block extraction, syntax correctness (`compile()`), and runtime safety (`exec()`).
    - **Retry Loop**: If validation fails, the system re-prompts the LLM up to `MAX_RETRIES` times.
    - **Fallback to Original**: If all retries fail, the system discards the invalid code and returns the original, un-mutated individual, ensuring the evolution process always continues.

---

## 📝 Minor Changes & Housekeeping

- Updated `.gitignore` to exclude the `runs/` directory, preventing team-specific experimental data from being committed.
- Centralized all SLURM job logging into a temporary `slurm-results/` directory before being migrated to a run-specific `logs/` folder.
- Updated the model path to use the `DeepSeek-R1-Distill-Qwen-32B` model.
