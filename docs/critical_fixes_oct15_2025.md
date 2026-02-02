# Critical Bug Fixes - October 15, 2025

This document summarizes critical blocking issues identified in code review and their resolutions.

## Issues Fixed

### 1. ✅ SLURM Log Polling (Blocking)

**Problem**: With `LOCAL = False`, jobs are submitted through SLURM, but `check4job_completion()` and `check4results()` were still looking for logs in the current working directory (`slurm-{job_id}.out`) instead of the configured `SLURM_LOG_DIR` (`slurm-results/`).

**Impact**: All distributed runs would stall until timeout, as the system could never find the completion logs.

**Fix**:
- Updated `check4job_completion()` in `run_improved.py` to use:
  - `slurm-results/llm-{job_id}.out` when `LOCAL = False`
  - `slurm-{job_id}.out` when `LOCAL = True` (backward compatibility)
- Updated `check4results()` in `run_improved.py` to use:
  - `slurm-results/eval-{job_id}.out` for evaluation job logs

**Files Modified**: `run_improved.py`

---

### 2. ✅ Roleplay Prompt Templates (Blocking)

**Problem**: The roleplay mutation templates (`mutant0.txt` through `mutant4.txt`) contained complete Python code examples instead of persona-based instructions. This caused the LLM to receive the example code as the prompt, degrading mutation quality immediately.

**Impact**: The LLM would receive malformed prompts, leading to poor or invalid code generation.

**Fix**: Restored all 5 templates to their original persona-based instruction format:
- `mutant0.txt`: Machine learning practitioner (performance focus)
- `mutant1.txt`: Software engineer (clean code focus)
- `mutant2.txt`: Performance optimization specialist
- `mutant3.txt`: Deep learning researcher (training dynamics)
- `mutant4.txt`: Code reviewer (readability focus)

**Files Modified**:
- `templates/FixedPrompts/roleplay/mutant0.txt`
- `templates/FixedPrompts/roleplay/mutant1.txt`
- `templates/FixedPrompts/roleplay/mutant2.txt`
- `templates/FixedPrompts/roleplay/mutant3.txt`
- `templates/FixedPrompts/roleplay/mutant4.txt`

---

### 3. ✅ Token Limits for DeepSeek (Blocking)

**Problem**: All `max_new_tokens` defaults were set to 32,000, which exceeds:
- Hugging Face hosted endpoint limits (would result in 422/400 errors)
- Practical limits for our local DeepSeek server

**Impact**: Requests to HF would fail immediately, and local server requests would be unnecessarily slow.

**Fix**: Adjusted token limits to practical values:
- **Hugging Face endpoints** (Mixtral, Llama3): `4096` tokens (matches their limits)
- **Local DeepSeek server**: `8192` tokens (reasonable for code generation)
- **QC checks**: `4096` tokens (sufficient for validation)
- Random token sampling in `submit_mixtral()`: `2048-4096` (down from `20000-32000`)

**Files Modified**:
- `src/llm_utils.py`:
  - `llm_code_qc_hf()`: 32000 → 4096
  - `submit_mixtral_hf()`: 32000 → 4096
  - `submit_llama3_hf()`: 32000 → 4096
  - `submit_mixtral()`: 32000 → 4096, random range 20000-32000 → 2048-4096
  - `submit_local_server()`: 32000 → 8192
- `server.py`:
  - `LLMRequest.max_new_tokens`: 32000 → 8192
  - Pipeline default: 32000 → 8192

---

### 4. ✅ QC Probabilistic Sampling (Major)

**Problem**: The `QC_CHECK_BOOL` constant was being passed directly to mutation/crossover scripts without resampling. This meant `PROB_QC` in `constants.py` had no effect - QC was either always on or always off.

**Impact**: The `PROB_QC` tuning knob was non-functional, preventing experimentation with quality control rates.

**Fix**: Replaced the constant `QC_CHECK_BOOL` with per-job probabilistic sampling based on `PROB_QC`:
```python
# Sample QC check based on PROB_QC probability
qc_check = random.random() < PROB_QC
```

This is now done separately for each mutation and crossover operation, allowing `PROB_QC` to control the frequency of quality control checks as intended.

**Files Modified**: `run_improved.py` (both mutation and crossover code paths)

---

## Testing Recommendations

1. **SLURM Log Polling**: Run a test with `LOCAL = False` and verify that the system correctly finds logs in `slurm-results/llm-*.out` and `slurm-results/eval-*.out`.

2. **Prompt Templates**: Review the first few generated prompts to ensure they follow the persona-based format and include the code block placeholder `{}`.

3. **Token Limits**: Monitor the first few LLM requests to ensure they complete without 422/400 errors and generate complete code (not truncated).

4. **QC Sampling**: Set `PROB_QC = 0.5` and verify that approximately 50% of jobs use quality control (check the command-line arguments in the SLURM logs).

---

## Notes

- All fixes prioritize compatibility with the **PACE-ICE hosted DeepSeek model**, as that is the primary inference target.
- The `MIXTRAL_MAX_NEW_TOKENS` environment variable can still be used to override defaults if needed for experimentation.
- The original `QC_CHECK_BOOL` constant is no longer used and can be considered deprecated in favor of the probabilistic `PROB_QC` approach.
