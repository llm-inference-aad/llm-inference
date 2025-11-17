# Retry Tracking System

## Overview

The retry tracking system monitors how many attempts it takes to create valid individuals during the evolutionary process. This helps answer questions like "On average, how many retries does it take to make a valid individual?"

## Implementation Details

### 1. Retry Counter in GLOBAL_DATA

Each gene/individual now has a `retry_count` field in `GLOBAL_DATA`:

```python
GLOBAL_DATA[gene_id] = {
    'sub_flag': successful_sub_flag,
    'job_id': job_id,
    'status': 'subbed file',
    'fitness': None,
    'start_time': time.time(),
    'retry_count': 0  # NEW: Tracks cumulative retries
}
```

### 2. How Retry Counts Work

- **Initial individuals**: Start with `retry_count = 0`
- **Mutation**: Child inherits parent's retry count + 1
- **Crossover**: Child inherits max(parent1_retry_count, parent2_retry_count) + 1
- **Failed attempts**: When a mutation/crossover fails, the failed gene is deleted and the individual keeps its old gene_id (no retry count increase for the individual itself, but the attempt is tracked separately)

### 3. RETRY_STATS Dictionary

A new global dictionary tracks statistics per generation:

```python
RETRY_STATS = {
    0: {
        'total_retries': 150,           # Sum of all retry_counts for successful individuals
        'successful_individuals': 50,   # Number of individuals that succeeded
        'failed_attempts': 10,          # Number of mutations/crossovers that failed this generation
    },
    1: { ... },
    ...
}
```

### 4. Checkpoint Integration

Retry statistics are automatically:
- **Calculated** when saving each checkpoint
- **Displayed** in the console during checkpoint save
- **Saved** in the checkpoint pickle file
- **Loaded** when resuming from a checkpoint (backwards compatible)

### 5. Console Output

When saving a checkpoint, you'll see:

```
📊 Generation 5 Retry Statistics:
   Successful individuals: 48
   Total retries (cumulative): 156
   Failed attempts (this gen): 12
   Average retries per individual: 3.25

Checkpoint saved as runs/my-run/checkpoints/checkpoint_gen_5.pkl
```

## Usage

### During a Run

Retry statistics are automatically calculated and displayed when each generation completes.

### Post-Run Analysis

Use the provided analysis script:

```bash
# Analyze a specific run
python scripts/analyze_retry_stats.py runs/my-run/checkpoints

# Or use a run ID
python scripts/analyze_retry_stats.py runs/baseline-metrics-2025-11-17/checkpoints
```

The script will:
1. Load all checkpoint files
2. Display per-generation statistics
3. Calculate overall summary statistics
4. Save a JSON summary to `retry_stats_summary.json`

### Example Output

```
================================================================================
RETRY STATISTICS ANALYSIS
================================================================================

Generation 0:
  ✓ Successful individuals: 50
  ↻ Total retries (cumulative): 0
  ✗ Failed attempts (this gen): 0
  📊 Average retries per individual: 0.00

Generation 1:
  ✓ Successful individuals: 48
  ↻ Total retries (cumulative): 156
  ✗ Failed attempts (this gen): 12
  📊 Average retries per individual: 3.25

Generation 2:
  ✓ Successful individuals: 47
  ↻ Total retries (cumulative): 189
  ✗ Failed attempts (this gen): 15
  📊 Average retries per individual: 4.02

================================================================================
OVERALL SUMMARY
================================================================================

Across all 3 generations:
  ✓ Total successful individuals: 145
  ↻ Total retries (cumulative): 345
  ✗ Total failed attempts: 27
  📊 Overall average retries per individual: 2.38

  📈 Min avg retries per generation: 0.00
  📈 Max avg retries per generation: 4.02
  📈 Median avg retries per generation: 3.25

📁 Summary saved to: runs/my-run/checkpoints/retry_stats_summary.json
```

## Key Metrics Explained

### Total Retries (Cumulative)
Sum of all `retry_count` values for successful individuals. Includes retries from parent lineage.

**Example**: If an individual has `retry_count=5`, it means 5 LLM generation attempts occurred in its lineage before it was successfully created.

### Failed Attempts (This Gen)
Number of mutation/crossover operations that failed during this specific generation. These are immediate failures tracked when `update_individual()` is called with `process_success=False`.

### Average Retries Per Individual
`total_retries / successful_individuals` - answers "how many retries on average does it take to create a valid individual?"

## Interpreting Results

- **Low average retries (< 2)**: LLM is generating valid code most of the time
- **Medium average retries (2-5)**: Some retry overhead but acceptable
- **High average retries (> 5)**: May indicate:
  - Prompt quality issues
  - Overly complex mutation/crossover operations
  - Evaluation criteria too strict

## Data Files

### Checkpoint Files
- Location: `runs/<run_id>/checkpoints/checkpoint_gen_<N>.pkl`
- Contains: Full `RETRY_STATS` dictionary with all generations up to N

### Summary JSON
- Location: `runs/<run_id>/checkpoints/retry_stats_summary.json`
- Contains: Aggregated statistics in JSON format
- Useful for: Plotting, analysis scripts, comparison across runs

## Backwards Compatibility

Old checkpoints without `RETRY_STATS` are supported:
- The analysis script will attempt to reconstruct statistics from `GLOBAL_DATA`
- If `retry_count` is missing from gene data, it defaults to 0
- Failed attempts cannot be reconstructed for old checkpoints (shown as 0)

## Future Enhancements

Potential additions:
1. Track retry reasons (syntax error, import error, evaluation failure)
2. Per-operation retry rates (mutation vs crossover)
3. Retry rate correlation with fitness scores
4. Real-time retry dashboard during evolution
