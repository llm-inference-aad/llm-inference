# File Path Fixes - October 14, 2025

## Summary
Fixed critical path resolution issues that were causing results files to be written to incorrect locations and causing FileNotFoundError during fitness evaluation.

## Problems Identified

### Problem 1: Results File Path Mismatch
**Symptom:** FileNotFoundError when checking fitness results after training completed
```
FileNotFoundError: [Errno 2] No such file or directory: 
'/home/.../sota/ExquisiteNetV2/results/xXx7drDYdpvt42t4w4Q4z5zj5iK_results.txt'
```

**Root Cause:** 
- `train.py` was writing results to: `sota/ExquisiteNetV2/runs/auto_20251014_183755/results/`
- `check4results()` was looking in: `sota/ExquisiteNetV2/results/`
- Mismatch occurred because `train.py` interpreted `RUN_DIR` as a relative path from its working directory

**Impact:** Evolution run crashed after evaluating only 1 out of 8 genes, preventing fitness comparison

### Problem 2: Relative RUN_DIR Path
**Symptom:** Nested `runs/` directories created inside `sota/ExquisiteNetV2/`

**Root Cause:**
- `run.sh` set `RUN_DIR="runs/${RUN_ID}"` as a relative path
- When training bash scripts executed from `sota/ExquisiteNetV2/`, they created `runs/` relative to that directory
- This caused: `sota/ExquisiteNetV2/runs/auto_20251014_183755/` instead of top-level `runs/auto_20251014_183755/`

**Impact:** Results files written to wrong location, breaking the run management system

## Solutions Implemented

### Fix 1: Make RUN_DIR an Absolute Path
**File:** `run.sh`
```bash
# Before:
RUN_DIR="runs/${RUN_ID}"

# After:
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}"
```

**Effect:** Now `RUN_DIR` is always an absolute path like `/home/.../llm-inference/runs/auto_20251014_183755`, regardless of subprocess working directory

### Fix 2: Ensure Training Scripts Run from Repository Root
**File:** `src/cfg/constants.py` - `PYTHON_BASH_SCRIPT_TEMPLATE`
```bash
# Added before running Python script:
cd "${LLM_INFERENCE_ROOT_DIR:-{root_dir}}"
```

**File:** `run_improved.py` - `write_bash_script_py()`
```python
# Updated template formatting to include root_dir:
bash_script_content = PYTHON_BASH_SCRIPT_TEMPLATE.format(
    python_runline=python_runline,
    slurm_log_dir=SLURM_LOG_DIR,
    root_dir=ROOT_DIR,  # Added this parameter
)
```

**Effect:** Training bash scripts now cd to repository root before executing, ensuring consistent relative paths

### Fix 3: Update check4results() to Search Multiple Locations
**File:** `run_improved.py` - `check4results()`

Added fallback search logic:
1. **Primary:** Absolute run directory path: `ROOT_DIR/runs/{run_id}/results/{gene_id}_results.txt`
2. **Secondary:** SOTA-relative path (legacy from bug): `SOTA_ROOT/runs/{run_id}/results/{gene_id}_results.txt`
3. **Tertiary:** Legacy SOTA location: `SOTA_ROOT/results/{gene_id}_results.txt`

**Effect:** Handles both old results (from before the fix) and new results (after the fix), with helpful error messages showing all searched paths

## Verification

### Path Resolution Test
```
✓ train.py writes to: /home/.../llm-inference/runs/auto_20251014_183755/results/
✓ check4results looks in: /home/.../llm-inference/runs/auto_20251014_183755/results/ (Priority 1)
✓ Paths match: TRUE
```

### File Structure (Correct)
```
llm-inference/
├── runs/
│   └── auto_20251014_183755/
│       ├── checkpoints/
│       ├── logs/
│       └── results/           # ← Results files go here (CORRECT)
└── sota/
    └── ExquisiteNetV2/
        ├── models/
        │   └── network_*.py
        └── results/           # ← Legacy location (still supported)
```

### File Structure (Old Bug - No Longer Happens)
```
sota/
└── ExquisiteNetV2/
    └── runs/                  # ← Nested runs (BUG - prevented by absolute path)
        └── auto_20251014_183755/
            └── results/
```

## Testing Recommendations

### Before Starting New Run
1. Verify no nested `runs/` in SOTA:
   ```bash
   ls -la sota/ExquisiteNetV2/runs/  # Should NOT exist
   ```

2. Clean up any test results:
   ```bash
   rm -rf runs/auto_*/results/*
   ```

### After Run Completes
1. Check results are in correct location:
   ```bash
   ls -la runs/auto_*/results/
   # Should see: *_results.txt files for all evaluated genes
   ```

2. Verify no files in wrong locations:
   ```bash
   ls -la sota/ExquisiteNetV2/results/        # Should be empty or legacy only
   ls -la sota/ExquisiteNetV2/runs/           # Should NOT exist
   ```

## Commits
- `d5bfa17` - Fix results file path resolution in check4results()
- `f567d04` - Fix RUN_DIR path to be absolute and ensure correct working directory

## Related Issues
- Token limit fixes (separate commit) - Addressed truncated LLM outputs
- Multiple gene_id entries in latency metrics - Expected behavior (retries/QC)
