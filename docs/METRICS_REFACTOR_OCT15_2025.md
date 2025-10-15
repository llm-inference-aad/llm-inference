# Metrics Organization Refactoring (Oct 15, 2025)

## Overview
Reorganized metrics collection to be **organized by run_id** instead of server session hash. This makes it easier to correlate metrics with specific evolution runs and keeps all run-related data together.

## Changes

### 1. New Metrics Directory Structure

**Before:**
```
metrics/
  data/
    -latency-9a695d7d19fc494a.json  # Flat structure, named by server hash
    -latency-abc123def456.json
```

**After:**
```
runs/
  auto_20251014_191652/
    metrics/
      latency-9a695d7d19fc494a.json  # Organized by run_id
      latency-abc123def456.json      # Multiple server sessions per run
  auto_20251015_120000/
    metrics/
      latency-def456abc123.json

metrics/
  data/
    -latency-*.json  # Legacy files (for backwards compatibility)
```

### 2. Updated Metrics JSON Format

Added `run_id` field to metrics files:

```json
{
  "run_id": "auto_20251014_191652",
  "run_hash": "9a695d7d19fc494a",
  "session_start": "2025-10-14T18:28:56.565101",
  "model_path": "/storage/...",
  "batch_size": 8,
  "batch_wait_time": 2,
  "requests": [...]
}
```

### 3. Server Mode Detection

`server.py` now detects execution mode:

- **Evolution Run Mode** (`RUN_ID` env var set by `run.sh`):
  - Writes to: `runs/{RUN_ID}/metrics/latency-{hash}.json`
  - Includes `run_id` field in JSON

- **Standalone Server Mode** (`RUN_ID` not set):
  - Falls back to: `metrics/data/-latency-{hash}.json` (legacy structure)
  - Sets `run_id` to `"server-only"`

### 4. Updated Plotting Scripts

Both `plot_latency_vs_accuracy.py` and `plot_latency_vs_goodput.py` now:

1. **Search Order** for metrics:
   - Try `runs/{run_id}/metrics/latency-{hash}.json` first (new structure)
   - Fall back to `metrics/data/-latency-{hash}.json` (legacy structure)
   - Auto-detect most recent metrics file if hash not provided

2. **Simplified Usage**:
   ```bash
   # Auto-detect everything (uses latest run and most recent metrics)
   python scripts/plot_latency_vs_accuracy.py
   
   # Specify run only (auto-detect metrics hash)
   python scripts/plot_latency_vs_accuracy.py --run-id auto_20251014_191652
   
   # Specify both
   python scripts/plot_latency_vs_accuracy.py --run-id latest --run-hash abc123
   ```

## Migration Path

### For Existing Metrics

Legacy metrics files in `metrics/data/` will continue to work:
- Plotting scripts check legacy location as fallback
- Old runs can still be analyzed with `--run-hash` parameter

### For Future Runs

All new runs automatically use the new structure:
1. `run.sh` exports `RUN_ID` environment variable
2. `server.py` detects `RUN_ID` and organizes metrics accordingly
3. Plotting scripts automatically find metrics in run directory

## Benefits

1. **Better Organization**: All run-related data in one place
   - `runs/{run_id}/checkpoints/`
   - `runs/{run_id}/results/`
   - `runs/{run_id}/metrics/`
   - `runs/{run_id}/logs/`

2. **Simplified Workflow**: No need to manually track server hashes
   - Auto-detection finds the right metrics file
   - `run_id` field in JSON for double-checking

3. **Multiple Server Sessions**: Can restart server during a run
   - Each session gets unique hash
   - All sessions grouped under same run_id

4. **Backwards Compatible**: Legacy metrics still accessible

## Technical Details

### Environment Variables

- `RUN_ID`: Set by `run.sh` (format: `auto_YYYYMMDD_HHMMSS`)
- `METRICS_PATH`: Optional override for metrics base path (default: `./metrics`)

### File Naming

- **New structure**: `latency-{16-char-hash}.json` (no leading dash)
- **Legacy structure**: `-latency-{16-char-hash}.json` (with leading dash)

This naming difference helps scripts distinguish between new and legacy formats.

## Related Documentation

- [Run Organization (Oct 15, 2025)](./RUN_ORGANIZATION_OCT15_2025.md)
- [Path Fixes (Oct 14, 2025)](./FIXES_OCT14_2025.md)
