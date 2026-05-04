# 07 - QLoRA Gap Checklist

This checklist captures what is still missing to support **real** QLoRA fine-tuning in this repository (beyond mock artifacts).

## 1. Training Pipeline Gaps (Must Have)

- [x] Add a dedicated QLoRA training entrypoint (`scripts/train_qlora.py`).
- [x] Implement Hugging Face model loading with 4-bit quantization (`bitsandbytes`, `nf4` or chosen config).
- [x] Add PEFT LoRA adapter setup (`LoraConfig`, `get_peft_model`).
- [x] Add data ingest + tokenization pipeline for SFT data (`dataset.json`).
- [x] Add train/eval loop (`SFTTrainer`) and checkpoint save policy.
- [x] Add adapter export path (`runs/<run_id>/qlora/adapters/`).

## 2. Dependency Gaps

- [x] Add `peft` (`pyproject.toml`).
- [x] Add `bitsandbytes` (`pyproject.toml`).
- [x] Add `datasets` (`pyproject.toml`).
- [x] Add `trl` (`pyproject.toml`).
- [ ] Pin versions known to be compatible with current CUDA + cluster image.
  > **Open**: `transformers>=4.44.0`, `trl>=0.8.6`, `peft>=0.10.0`, `bitsandbytes>=0.43.0` are specified.
  > Exact pins should be locked in `uv.lock` after a successful cluster install.

## 3. Configuration Gaps

- [x] Add explicit QLoRA config block (`configs/qlora.yaml`):
  - [x] `model_name_or_path` (populated from `$MODEL_PATH`)
  - [x] `load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_compute_dtype`
  - [x] `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`
  - [x] batch size, grad accumulation, learning rate, warmup, max steps/epochs
  - [x] save/eval/logging intervals
- [x] Add environment variable contract for output directory and run ID (`$RUN_DIR`, `$RUN_ID`).
- [ ] Add optional config for distributed/multi-GPU launch.
  > **Open**: current setup targets single-GPU. Multi-GPU requires `torchrun` + `DeepSpeed`/`FSDP` config.

## 4. Run Orchestration Gaps

- [x] Add a SLURM script specifically for QLoRA fine-tuning (`scripts/run_qlora.sh`).
- [x] Add a local-dev launch command: `bash scripts/run_qlora.sh` (no SLURM prefix).
- [x] Add run directory structure for QLoRA outputs (created by `train_qlora.py` and `run_qlora.sh`):
  - [x] `runs/<run_id>/qlora/checkpoints/`
  - [x] `runs/<run_id>/qlora/logs/`
  - [x] `runs/<run_id>/qlora/metrics/`
  - [x] `runs/<run_id>/qlora/adapters/`

## 5. Validation and Success Criteria Gaps

- [x] Add training success marker (`status.json`) with terminal states: `completed`, `failed`, `interrupted`.
- [x] Add required metrics schema (`train_loss`, runtime, throughput, best checkpoint) in `metrics/train_metrics.json`.
- [ ] Add post-train evaluation script and minimum acceptance criteria.
  > **Open**: need a script that loads the adapter and runs the EMADE mutation prompts to compare outputs against a held-out baseline.
- [x] Add failure-mode handling for OOM, NaNs, and invalid data rows.

## 6. Testing Gaps

- [x] Add unit tests for dataset preprocessing and config loading (`tests/test_qlora_smoke.py`, 14/14 pass).
- [x] Add smoke test path via `--max-steps 3` CLI flag.
- [ ] Add artifact-integrity test (checkpoint, adapter, metrics, status file all present post-run).
  > **Open**: this requires an actual GPU training run to produce artifacts. Mark complete after first successful sbatch.

## 7. Documentation Gaps

- [ ] Add dedicated QLoRA setup guide (hardware assumptions, CUDA, env setup).
- [ ] Add minimal end-to-end command examples (local and SLURM).
- [ ] Add troubleshooting section for common training/runtime failures.

## 8. De-scoping / Cleanup Targets (to reduce confusion)

- [ ] Remove mock-only QLoRA artifacts from `runs/` when auditing real finetuning readiness.
- [ ] Clearly label any synthetic artifacts as `mock` in docs if retained for demo/testing.

---

## Remaining Implementation Gaps (Beyond Checklist)

The checklist items are now filled. The following gaps must still be addressed **before** QLoRA fine-tuning meaningfully improves the EMADE evolutionary loop:

### Gap A — Data volume (CRITICAL, blocks training quality)
`dataset.json` currently has **11 samples**. QLoRA SFT needs a minimum of ~200 diverse examples for meaningful generalization; 1000+ is typical. The current dataset produces adapters that are little more than overfit memorization.

**Fix**: run several full evolution cycles to accumulate data, or augment by:
1. Collecting all `prompt`/`generated_text` pairs from every SLURM eval job (they already pass through `llm_utils.py`).
2. Adding an auto-append hook in `llm_utils.generate_augmented_code()` that writes every accepted (non-fallback) LLM response to `dataset.json`.

### Gap B — Inference server integration (blocks closed-loop improvement)
After fine-tuning, `server.py` still loads the base model. There is no code path to load a LoRA adapter on top of the base model at serve time.

**Fix**: add an `ADAPTER_PATH` env var to `server.py` and wrap the model load with:
```python
from peft import PeftModel
if adapter_path := os.getenv("ADAPTER_PATH"):
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # optional: bake weights for faster inference
```

### Gap C — No eval split / post-train validation
With 11 samples there is no held-out eval. Even with more data, there is no script that re-runs the EMADE mutation prompts through the fine-tuned model and compares generated-code quality or fitness distribution to baseline.

**Fix**: write `scripts/eval_qlora.py` that:
1. Loads the fine-tuned adapter.
2. Runs N mutation prompts (drawn from dataset or a held-out set).
3. Computes average `valid_python_rate`, `fitness_percentile`, and `response_length`.
4. Writes results to `runs/<run_id>/qlora/metrics/eval_results.json`.

### Gap D — Per-model `target_modules` validation
`configs/qlora.yaml` lists Gemma-7b-it layer names. If `MODEL_PATH` is swapped for a different architecture (e.g., DeepSeek, Llama 3, Mistral), the layer names may not exist and PEFT will silently skip them or error.

**Fix**: add a pre-flight check in `train_qlora.py` that inspects `model.named_modules()` and warns if any configured `target_modules` name is absent.

### Gap E — `uv.lock` not updated with new deps
`peft`, `bitsandbytes`, `datasets`, `trl` were added to `pyproject.toml` but `uv.lock` has not been regenerated. The SLURM job will re-resolve on first run, which may fail if the cluster has no internet access.

**Fix**: run `uv lock` locally (with internet), commit the updated `uv.lock`, and let SLURM jobs use `uv sync --frozen`.

### Gap F — Multi-GPU / gradient checkpointing not configured
A Gemma-7b-it model loaded in 4-bit uses ~6–8 GB VRAM. A single A100-40GB is sufficient for batch=1, but gradient checkpointing is not enabled, which limits how long sequences can be. Long EMADE prompts (seen in dataset.json) can easily exceed 1024 tokens.

**Fix**: add `gradient_checkpointing: true` to `configs/qlora.yaml` and pass it through `SFTConfig`.
