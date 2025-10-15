# Cleanup Status - October 15, 2025

## Summary

Completed comprehensive refactoring to organize the project by run_id and clarify execution modes.

---

## ✅ Completed

### 1. Metrics Organization
- **Before:** Flat structure in `metrics/data/` with server-hash naming
- **After:** Organized by run_id in `runs/{run_id}/metrics/`
- **Files Changed:**
  - `server.py`: Auto-detect RUN_ID, write to run-specific metrics dir
  - `scripts/plot_latency_vs_accuracy.py`: Search new structure first, auto-detect hash
  - `scripts/plot_latency_vs_goodput.py`: Same updates
- **Documentation:** `docs/METRICS_REFACTOR_OCT15_2025.md`

### 2. LOCAL Mode Clarification
- **Discovery:** `LOCAL = True` in `src/cfg/constants.py`
- **Impact:** Gene evaluations run locally with bash (no separate SLURM jobs)
- **Result:** No `eval-*.out/err` or `llm-*.out/err` logs (expected behavior!)
- **Documentation:** Updated `docs/RUN_ORGANIZATION_OCT15_2025.md` with explanation

### 3. Python Environment Documentation
- **Created:** `docs/VENV_VS_UV_MIGRATION.md`
- **Content:**
  - Current setup: `.venv` with traditional pip
  - Why migrate to uv: 10-100x faster, better resolution, infrastructure team recommendation
  - Migration path: Gradual (recommended) or direct
  - Updated workflows and SLURM integration
  - Troubleshooting guide

---

## 📁 Current Directory Structure

```
llm-inference/
├── runs/
│   ├── auto_20251014_191652/
│   │   ├── checkpoints/
│   │   ├── results/
│   │   ├── logs/                    # SLURM logs (migrated at job completion)
│   │   └── metrics/                 # NEW! Organized by run
│   │       └── latency-{hash}.json
│   └── latest -> auto_20251014_191652
│
├── metrics/
│   └── data/
│       └── -latency-{hash}.json     # Legacy (backwards compatibility)
│
├── slurm-results/
│   ├── slurm-main-3388893.out       # Old runs (not yet migrated)
│   ├── slurm-main-3388973.out
│   └── slurm-server-*.out           # Server logs (temporary staging)
│
├── scripts/
│   ├── plot_latency_vs_accuracy.py  # Updated: auto-detect metrics
│   ├── plot_latency_vs_goodput.py   # Updated: auto-detect metrics
│   └── plots/                       # Generated visualizations
│
├── docs/
│   ├── FIXES_OCT14_2025.md          # Path resolution fixes
│   ├── RUN_ORGANIZATION_OCT15_2025.md  # Run organization + LOCAL mode
│   ├── METRICS_REFACTOR_OCT15_2025.md  # Metrics refactoring
│   └── VENV_VS_UV_MIGRATION.md      # NEW! Python environment guide
│
└── .venv/                           # Current Python environment
```

---

## 🧹 Cleanup Options

### slurm-results/ Directory

**Current State:**
- Contains old logs from jobs: 3388893, 3388973, 3388977
- Server logs: `slurm-server-*.out/err`
- New runs automatically migrate logs to `runs/{run_id}/logs/`

**Options:**

1. **Migrate Old Logs** (Recommended)
   ```bash
   # For each old run, move logs to appropriate run directory
   # Example for job 3388893 (run auto_20251014_191652):
   mv slurm-results/slurm-main-3388893.* runs/auto_20251014_191652/logs/
   
   # After migrating all, keep slurm-results/ for temporary staging
   ```

2. **Archive and Remove**
   ```bash
   # Archive old logs
   tar czf slurm-results-archive-$(date +%Y%m%d).tar.gz slurm-results/
   
   # Keep slurm-results/ directory (needed for SBATCH output directives)
   rm slurm-results/*.out slurm-results/*.err
   ```

3. **Keep for Reference**
   - Leave unmigrated logs in place for debugging old runs
   - They don't interfere with new runs
   - Future runs will self-organize

**Recommendation:** Option 1 (migrate), then keep `slurm-results/` empty as temporary staging area.

### metrics/data/ (Legacy Metrics)

**Current State:**
- One file: `-latency-9a695d7d19fc494a.json` (from job 3388893)
- Used by old runs before metrics refactoring

**Options:**

1. **Move to Corresponding Run** (Best)
   ```bash
   # Identify which run this metrics file belongs to
   # (check timestamps or gene_ids in the file)
   
   mv metrics/data/-latency-9a695d7d19fc494a.json \\
      runs/auto_20251014_191652/metrics/latency-9a695d7d19fc494a.json
   ```

2. **Keep for Backwards Compatibility**
   - Plotting scripts still search legacy location
   - Useful for analyzing old runs
   - Doesn't interfere with new runs

**Recommendation:** Move to corresponding run directory, then keep `metrics/data/` for any future standalone server sessions.

---

## 🎯 Next Steps (Optional)

### 1. Clean Up Old Logs
```bash
# Identify which logs belong to which runs
grep -l "RUN_ID" slurm-results/*.out

# Migrate to appropriate run directories
# See "Cleanup Options" above
```

### 2. Test uv Migration
```bash
# Install uv (if available)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create test environment
uv venv .venv-uv
uv sync

# Test
source .venv-uv/bin/activate
python run_improved.py --help
```

### 3. Start New Evolution Run
```bash
# New runs will automatically use the improved structure
sbatch run.sh

# Check that metrics appear in runs/{run_id}/metrics/
```

---

## 📊 What's Working Now

✅ **Path Resolution:** Absolute paths, multi-path fallback  
✅ **Run Organization:** Self-contained runs with all data in one place  
✅ **Log Management:** Automatic migration to run-specific directories  
✅ **Metrics Organization:** Per-run metrics with auto-detection  
✅ **Plotting Scripts:** Smart defaults (latest run, auto-detect hash)  
✅ **Documentation:** Comprehensive guides for all changes  
✅ **Backwards Compatibility:** Legacy structures still work  

---

## 🔍 Key Files Modified (Oct 14-15, 2025)

| File | Changes | Purpose |
|------|---------|---------|
| `run_improved.py` | Multi-path results search | Fix FileNotFoundError |
| `run.sh` | Absolute RUN_DIR, log migration | Fix SLURM path resolution |
| `src/cfg/constants.py` | Token limits (32k), LOCAL=True | Support DeepSeek model |
| `server.py` | Run-specific metrics dir | Organize by run_id |
| `scripts/plot_*.py` | Auto-detect metrics | Smart defaults |
| `docs/*.md` | Comprehensive docs | Context and guides |

---

## 📝 Summary

The project is now well-organized with:
- Run-specific directories for all data
- Automatic log and metrics management
- Smart defaults for analysis scripts
- Comprehensive documentation
- Backwards compatibility

The only remaining question is whether to migrate old logs/metrics to their respective run directories or keep them as-is for reference.
