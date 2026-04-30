# Change Log

## April 27–29, 2026

### Model loading and CPU test fixes
- Fixed the 1B CPU test path resolution so Hugging Face snapshot symlinks resolve correctly.
- Verified the 1B CPU smoke run completed successfully and passed all 5 checks.

### vLLM adaptive decoding work
- Added adaptive speculative decoding logic to `server_vllm.py`.
- Added EWMA-based speculative acceptance tracking.
- Added request-aware speculative token budgeting.
- Recorded speculative telemetry in metrics and responses:
  - `speculative_accepted`
  - `vllm_num_speculative_tokens`
- Preserved request-specific speculative configuration so per-request settings are not overwritten by global defaults.

### New benchmarking and comparison utilities
- Added `run_vllm_adaptive.sh` for adaptive vLLM launches.
- Added `compare_decoding_runs.py` to compare baseline vs candidate metrics.
- Updated `run_five_config_tests.py` to accept custom `--port` and `--output-dir` arguments.
- Added `submit_vllm_smoke.sh` for short vLLM smoke runs.
- Patched `submit_vllm_smoke.sh` to disable server-wide default constraints for mixed workloads.

### Environment and launch fixes
- Fixed vLLM startup on V100 GPUs by forcing `VLLM_DTYPE=half` in launch scripts and `.env` handling.
- Fixed server startup OOM on V100 by adding `ENFORCE_EAGER=true` support in `server_vllm.py` and using it for benchmark launches.
- Reduced `MAX_MODEL_LEN` to `8192` for the 300-request benchmark to reduce startup pressure.

### Benchmark runs and results
- Ran a short smoke benchmark with 5 configs and compared it against the baseline.
- Ran a 500-request benchmark; it completed 424/500 requests before hitting the 4-hour SLURM timeout.
- Ran a 300-request benchmark; it failed at vLLM startup due to compile/autotune memory pressure, then was resubmitted with eager mode enabled.
- Submitted corrected benchmark jobs after each failure mode was identified.

### Generated configuration snapshots
- Multiple `.env.backup.*` files were created as configuration snapshots for different speculative and constrained decoding setups.
- Existing environment profiles include:
  - vLLM-only
  - vLLM + constrained decoding
  - vLLM + speculative decoding
  - vLLM + combined constrained/speculative decoding
  - CPU test configuration

### Current state
- Main vLLM benchmark scripts now force settings compatible with the V100 nodes used here.
- `server_vllm.py` contains the adaptive speculation changes and eager-mode support.
- Benchmark outputs and comparison artifacts are preserved under `runs/` and `slurm_logs/`.
