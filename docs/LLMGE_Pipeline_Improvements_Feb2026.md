# LLMGE Pipeline Improvements & Fixes - Feb 2026

This document details the critical fixes and optimizations implemented in the `feature/rag-pipeline-surya` branch to stabilize the LLM Guided Evolution pipeline.

## 1. Critical Fix: Direct Job Completion Wait

**Problem**: The pipeline was failing to generate any valid genes because `direct_` LLM jobs (submitted to a local thread pool) were being checked in a non-blocking manner.
- `check4job_completion` returned `None` when a job was still running.
- The calling functions (`delayed_creation_check`, `delayed_mate_check`) interpreted `None` as a failure (Falsy value).
- **Result**: All 16 initial genes were marked as "Error Loading Model Files" (Zombies) before the LLM even finished processing, leading to empty generations.

**Fix**:
- Modified `run_improved.py`: `check4job_completion` now explicitly **block-waits** for `direct_` jobs using `future.result(timeout=...)`.
- Increased default timeout from 120s to **14,400s (4 hours)** to match `LOCAL_SERVER_TIMEOUT`.
- **References**:
    - `run_improved.py` (Line ~447): Added `future.result(timeout=timeout)` waiting logic.
    - `run_improved.py` (Line ~421): Updated default `timeout` parameter.

## 2. Self-Correcting Code Generation Loop

**Problem**: The LLM often generates code with minor syntax errors or import issues. Simply retrying with the same prompt is inefficient.

**Fix**: Implemented a **Context-Aware Self-Correction Loop**.
- When validation fails, the **error message** (`validation_error`) and the **previously generated faulty code** (`candidate_code`) are fed back into the retry prompt.
- **Mechanism**:
    - `llm_utils.py`: `_format_retry_prompt` now constructs a "Fix this error" prompt.
    - `generate_augmented_code` loops up to `LLM_GENERATION_MAX_RETRIES`.
    - **Prompt Format**:
      ```text
      PREVIOUS ATTEMPT:
      {faulty_code}

      ERROR:
      {error_message}

      INSTRUCTION:
      Fix the code above based on the error message. Output only the corrected Python code.
      ```
- **References**:
    - `src/llm_utils.py`: `generate_augmented_code` and `_format_retry_prompt`.

## 3. Zombie Gene Protection

**Problem**: Genes that failed generation (missing files) were being passed to evaluation/mating, causing downstream crashes.

**Fix**:
- Added `verify_population_integrity` in `run_improved.py`.
- Proactively scans the population for "Zombie Genes" (valid status but missing `.py` file).
- Marks them as `DEAD` or re-triggers generation before the EA loop proceeds.
- **References**:
    - `run_improved.py`: `verify_population_integrity` function.

## 4. Hardware & Model Configuration Adjustments

**Problem**: Llama-3.3-70B (4-bit) was slow on a single GPU. There was concern about CPU offloading and batching overhead masking performance issues.

**Optimization**:
- **Scaled to 2x GPUs**:
    - Updated `server.sh` to request `#SBATCH --gres=gpu:2` and `#SBATCH -C "H100|A100-80GB"`.
    - Llama 70B automatically shards across 2 GPUs (via `device_map="auto"`), preventing any CPU offloading and increasing inference speed.
- **Simplified Batching**:
    - Set `SERVER_BATCH_SIZE=1` in `.env`.
    - **Rationale**: Removes server-side batching/queuing as a variable to isolate inference latency.
- **References**:
    - `.env`: `SERVER_BATCH_SIZE=1`, `MODEL_PATH` set to Llama 70B.
    - `server.sh`: Updated GPU request.

## 5. Other Improvements

- **Tokenizer Fix**: Added `padding_side='left'` in `server.py` to fix "right-padding" warnings and improve batching correctness.
- **Logging**: Added VRAM usage logging in `server.py`.
- **Validation**: Enhanced `validate_module_source` in `llm_utils.py` to instantiate classes, catching `__init__` errors early.
