# Evaluation Workflow: Comparing vLLM vs HuggingFace

## Quick Start

### 1. Run Experiment with vLLM
```bash
# Edit .env: set USE_VLLM=true
sbatch server.sh
sbatch run.sh
```

### 2. Run Experiment without vLLM
```bash
# Edit .env: set USE_VLLM=false
sbatch server.sh
sbatch run.sh
```

### 3. Compare Results
```bash
bash compare_runs.sh runs/<vllm_run_id> runs/<no_vllm_run_id>
```

---

## What Happens Automatically

When you run `sbatch run.sh`, the following happens automatically at the end:

1. **Server shutdown** - LLM server is stopped gracefully
2. **Log collection** - All SLURM logs moved to `runs/<run_id>/logs/`
3. **Evaluation report generation** - Comprehensive metrics report created

### Generated Files

After each run completes, you'll find these files in `runs/<run_id>/`:

```
runs/2024-01-15_experiment1/
├── evaluation_report.txt      ← Human-readable metrics summary
├── evaluation_report.json     ← Machine-readable metrics (for scripts)
├── logs/
│   ├── slurm-server-*.out    ← Server output (with detailed metrics)
│   ├── slurm-server-*.err    ← Server errors
│   ├── slurm-main-*.out      ← Main job output
│   └── slurm-main-*.err      ← Main job errors
├── metrics/
│   └── latency-*.json        ← Detailed per-request metrics
├── checkpoints/              ← Evolution checkpoints
└── run_metadata.json         ← Run configuration
```

---

## Evaluation Report Contents

The `evaluation_report.txt` file contains:

### Configuration
- vLLM Status (ENABLED/DISABLED)
- Backend type (vLLM vs HuggingFace)

### Performance Metrics
- **Total Requests**: Number of inference requests processed
- **Latency** (seconds):
  - Average
  - Min
  - Max
  - Median (if available)
  - P95, P99 (if available)
- **Throughput**:
  - Requests per second
  - Requests per minute

### Resource Usage
- Process Memory (MB)
- GPU Memory Used/Total (MB)
- GPU Memory Utilization (%)

### Model Performance
- Average Evaluation Score (model quality metric)

---

## Comparison Workflow

### Manual Comparison

1. **View individual reports:**
   ```bash
   cat runs/<run_id>/evaluation_report.txt
   ```

2. **Compare two runs side-by-side:**
   ```bash
   bash compare_runs.sh runs/vllm_run runs/no_vllm_run
   ```

This will show:
- Side-by-side metric comparison
- Percentage differences
- Which run performed better (✅/❌ indicators)

### Example Output

```
================================================================================
                            SIDE-BY-SIDE COMPARISON
================================================================================
Run 1: 2024-01-15_vllm_test
Run 2: 2024-01-15_no_vllm

Metric                         Run 1            Run 2          Change
--------------------------------------------------------------------------------
Backend:         vLLM (continuous batching)  HuggingFace (standard)

Total Requests                   150              150          same

Avg Latency                   2.4567 sec       4.1234 sec    -40.4% ✅
Min Latency                   1.2345 sec       2.3456 sec    -47.4% ✅
Max Latency                   5.6789 sec       8.9012 sec    -36.2% ✅

Throughput (req/sec)          0.0417 req       0.0242 req    +72.3% ✅
Throughput (req/min)           2.50 req         1.45 req     +72.4% ✅

GPU Memory Used              35421.00 MB      28934.00 MB    +22.4% ❌
GPU Memory Util                 86.50%           70.60%      +22.5% ❌

Evaluation Score               0.8234           0.8241        -0.1%
================================================================================
```

---

## Metrics Explanation

### Latency (Lower is Better)
- **Average**: Mean time per request
- **Min/Max**: Range of response times
- **P95/P99**: 95th/99th percentile (tail latency)

vLLM typically reduces latency by 30-60% due to continuous batching.

### Throughput (Higher is Better)
- **Requests/sec**: Processing rate
- **Requests/min**: Same metric, different scale

vLLM typically increases throughput by 2-5x.

### Memory Usage
- **GPU Memory**: vLLM uses more GPU memory for KV cache optimization
- **Process Memory**: Server process RAM usage

vLLM may use 10-30% more GPU memory but provides better performance.

### Evaluation Score
- Measures output quality (should be similar between backends)
- If scores differ significantly, check for configuration issues

---

## Advanced Usage

### Generate Report for Existing Run
```bash
uv run python generate_evaluation_report.py runs/<run_id>
```

### View Detailed Per-Request Metrics
```bash
cat runs/<run_id>/metrics/latency-*.json
```

### Extract Specific Metrics with jq
```bash
# Get all latencies
jq '.requests[].\_latency_sec' runs/<run_id>/metrics/latency-*.json

# Get average evaluation score
jq '.requests | map(.evaluation_score) | add / length' runs/<run_id>/metrics/latency-*.json

# Get throughput over time
jq -r '.requests[] | [.timestamp, .\_latency_sec] | @csv' runs/<run_id>/metrics/latency-*.json
```

---

## Troubleshooting

### No evaluation report generated
- Check if `run.sh` completed successfully
- Run manually: `uv run python generate_evaluation_report.py runs/<run_id>`

### Missing metrics in report
- Ensure server processed requests (check `total_requests > 0`)
- Verify SLURM logs exist in `runs/<run_id>/logs/`

### Comparison shows no difference
- Verify one run used vLLM and the other didn't
- Check `vllm_enabled` field in JSON reports

---

## Best Practices

1. **Run multiple trials**: Run each configuration 3-5 times for statistical significance
2. **Same conditions**: Use same model, dataset, and hardware for fair comparison
3. **Warmup**: First few requests may be slower (model loading)
4. **Document runs**: Use descriptive run IDs (e.g., `2024-01-15_vllm_llama70b`)

---

## Files Reference

| File | Purpose |
|------|---------|
| `generate_evaluation_report.py` | Extracts metrics and generates report |
| `compare_runs.sh` | Compares two runs side-by-side |
| `evaluation_report.txt` | Human-readable metrics summary |
| `evaluation_report.json` | Machine-readable metrics (JSON) |
| `BACKEND_COMPARISON_GUIDE.md` | Detailed backend comparison guide |

---

## Questions?

- Configuration: See `.env` file
- vLLM options: See `BACKEND_COMPARISON_GUIDE.md`
- Run management: See `run.sh` and `server.sh`
