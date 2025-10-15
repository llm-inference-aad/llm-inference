# Run Organization Improvements - October 15, 2025

## Summary of Changes

All improvements have been implemented to make runs self-contained and analysis scripts more convenient.

## 1. Run-ID Increments Automatically ✓

**Q: Does run-id increase for each successive run?**  
**A: Yes!** Run IDs use timestamp format: `auto_YYYYMMDD_HHMMSS`

Examples in chronological order:
- `auto_20251014_183755` (Oct 14, 6:37:55 PM)
- `auto_20251014_191652` (Oct 14, 7:16:52 PM)
- `auto_20251015_103045` (Oct 15, 10:30:45 AM)

The timestamp naturally increases with each run, making them sortable and easily identifiable.

## 2. SLURM Logs Now Organized by Run-ID ✓

### What Changed

**Before:**
```
slurm-results/
├── slurm-main-3388893.out
├── slurm-main-3388893.err
├── eval-1234567.out
├── eval-1234567.err
└── ... (all mixed together)
```

**After:**
```
runs/
├── auto_20251014_191652/
│   ├── checkpoints/
│   ├── results/
│   └── logs/                           # ← All SLURM logs here!
│       ├── slurm-main-3388973.out
│       ├── slurm-main-3388973.err
│       ├── eval-1234567.out
│       └── eval-1234568.out
└── auto_20251015_103045/
    └── logs/                           # ← Next run's logs
        └── ...
```

### How It Works

1. **During run:** SLURM logs initially go to `slurm-results/` (required by SLURM's static directives)
2. **At end of run:** `run.sh` automatically moves ALL logs to `runs/{run_id}/logs/`:
   - Main job logs: `slurm-main-{job_id}.out/err`
   - Evaluation logs: `eval-*.out/err`
   - LLM job logs: `llm-*.out/err`
3. **Cleanup:** Old `slurm-results/` can be safely deleted once runs complete

### Benefits

- ✅ Self-contained runs - everything in one place
- ✅ Easy to archive/share entire run
- ✅ No more cluttered `slurm-results/` directory
- ✅ Logs tagged with run_id in metadata

## 3. Plotting Scripts Default to Latest Run ✓

### What Changed

**Before:**
```bash
$ python scripts/plot_latency_vs_accuracy.py
ERROR: the following arguments are required: --run-id, --run-hash
```

**After:**
```bash
$ python scripts/plot_latency_vs_accuracy.py
ℹ️  Using latest run: auto_20251015_103045
ℹ️  Auto-detecting metrics hash...
ℹ️  Detected metrics hash: 9a695d7d19fc494a
📊 Analyzing run: auto_20251015_103045
🔍 Metrics hash: 9a695d7d19fc494a
💾 Output: scripts/plots/latency_vs_accuracy_auto_20251015_103045.png
✅ Loaded 18 metric requests
✅ Loaded 8 accuracy results
...
```

### Updated Scripts

Both plotting scripts now have smart defaults:

#### `plot_latency_vs_accuracy.py`
- `--run-id`: Optional, defaults to `latest` symlink
- `--run-hash`: Optional, auto-detects from most recent metrics file
- `--output`: Optional, defaults to `scripts/plots/latency_vs_accuracy_{run_id}.png`

#### `plot_latency_vs_goodput.py`
- `--run-id`: Optional, defaults to `latest` symlink
- `--run-hash`: Optional, auto-detects from most recent metrics file
- `--output`: Optional, defaults to `scripts/plots/latency_vs_goodput_{run_id}.png`

### Usage Examples

```bash
# Simplest - just run it!
python scripts/plot_latency_vs_accuracy.py
python scripts/plot_latency_vs_goodput.py

# Specific run
python scripts/plot_latency_vs_accuracy.py --run-id auto_20251014_191652

# Specific run with custom metrics
python scripts/plot_latency_vs_accuracy.py \\
    --run-id auto_20251014_191652 \\
    --run-hash abc123def456

# Custom output location
python scripts/plot_latency_vs_accuracy.py --output my_plot.png
```

### Auto-Detection Logic

1. **Run-ID:**
   - If not provided → Uses `runs/latest` symlink
   - Resolves symlink to actual run directory
   - Shows helpful list of available runs if `latest` doesn't exist

2. **Metrics Hash:**
   - Searches `metrics/data/` for `*-latency-*.json` files
   - Picks the most recently modified file
   - Extracts hash from filename pattern: `-latency-{hash}.json`
   - Falls back gracefully if no metrics found (goodput-only plot)

## 4. New Plots Directory Structure ✓

### Directory Created

```
scripts/
├── plot_latency_vs_accuracy.py
├── plot_latency_vs_goodput.py
└── plots/                              # ← New!
    ├── README.md
    ├── latency_vs_accuracy_auto_20251014_191652.png
    ├── latency_vs_accuracy_auto_20251015_103045.png
    ├── latency_vs_goodput_auto_20251014_191652.png
    └── latency_vs_goodput_auto_20251015_103045.png
```

### Benefits

- ✅ All plots in one organized location
- ✅ Filenames include run_id for easy identification
- ✅ High DPI (300) publication-quality images
- ✅ No clutter in repository root
- ✅ Easy to `.gitignore` plots/ if desired

## Complete File Structure After Improvements

```
llm-inference/
├── runs/                               # All runs organized here
│   ├── latest -> auto_20251015_103045 # Symlink to latest
│   ├── auto_20251014_191652/
│   │   ├── checkpoints/
│   │   │   └── checkpoint_gen_*.pkl
│   │   ├── results/
│   │   │   └── *_results.txt
│   │   ├── logs/                       # ← SLURM logs here!
│   │   │   ├── slurm-main-3388973.out
│   │   │   ├── slurm-main-3388973.err
│   │   │   └── eval-*.out/err
│   │   └── run_metadata.json
│   └── auto_20251015_103045/
│       └── ... (same structure)
├── scripts/
│   ├── plot_latency_vs_accuracy.py     # Smart defaults
│   ├── plot_latency_vs_goodput.py      # Smart defaults
│   └── plots/                          # Generated visualizations
│       ├── README.md
│       └── *.png
├── metrics/
│   └── data/
│       └── -latency-*.json             # Auto-detected
├── slurm-results/                      # Temporary (can delete when empty)
└── sota/ExquisiteNetV2/
    ├── models/
    │   └── network_*.py
    └── results/                        # Legacy (still supported)
```

## Cleanup Instructions

After your next successful run completes, you can safely:

```bash
# 1. Verify logs were moved
ls -la runs/auto_*/logs/
# Should see slurm logs there

# 2. Remove old slurm-results directory
rm -rf slurm-results/

# 3. Optional: Add plots to .gitignore if you don't want to commit them
echo "scripts/plots/*.png" >> .gitignore
```

## Testing the Changes

### Test SLURM Log Organization
```bash
# Start a run
sbatch run.sh

# After it completes, check logs are in run directory
ls -la runs/auto_*/logs/

# Should see:
# - slurm-main-{job_id}.out
# - slurm-main-{job_id}.err  
# - eval-*.out (for each gene evaluation)
# - eval-*.err
```

### Test Plotting Defaults
```bash
# Just run without arguments
python scripts/plot_latency_vs_accuracy.py

# Should:
# ✅ Auto-detect latest run
# ✅ Auto-detect metrics hash
# ✅ Save to scripts/plots/latency_vs_accuracy_{run_id}.png
# ✅ Show helpful progress messages
```

## Commits

- `356acc2` - Improve run organization and plotting defaults
- `10c76c4` - Fix SLURM path resolution using SLURM_SUBMIT_DIR
- `f567d04` - Fix RUN_DIR path to be absolute and ensure correct working directory
- `d5bfa17` - Fix results file path resolution in check4results()

## Summary

✅ **Run-IDs increment automatically** (timestamp-based)  
✅ **SLURM logs organized by run** (moved to `runs/{run_id}/logs/`)  
✅ **Plotting scripts have smart defaults** (no arguments needed)  
✅ **Plots directory created** (`scripts/plots/`)  
✅ **Auto-detection of metrics hash**  
✅ **Self-contained, portable run directories**  

Everything is now organized, automated, and convenient! 🎉
