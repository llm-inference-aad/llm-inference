"""
QLoRA fine-tuning entrypoint for the EMADE evolutionary loop LLM.

Reads dataset.json (prompt / generated_text / fitness), filters by quality,
and fine-tunes the configured base model with 4-bit quantization + LoRA.

Designed for consecutive SLURM jobs: each job resumes from the latest
checkpoint in the run directory, so the full training can span multiple
16-hour slots without losing progress.

Usage:
    uv run python scripts/train_qlora.py                          # use defaults
    uv run python scripts/train_qlora.py --config configs/qlora.yaml
    uv run python scripts/train_qlora.py --max-steps 3            # smoke test
    uv run python scripts/train_qlora.py --resume-from-checkpoint # auto-detect
    uv run python scripts/train_qlora.py --resume-from-checkpoint runs/x/qlora/checkpoints/checkpoint-50
"""

import sys
import os
import json
import logging
import argparse
import datetime
import time
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_from_env = os.environ.get("MODEL_PATH", "")
    if model_from_env:
        cfg["model_name_or_path"] = model_from_env

    run_id = os.environ.get("RUN_ID", cfg.get("run_id", "qlora_default"))
    cfg["run_id"] = run_id

    run_dir = os.environ.get("RUN_DIR", "")
    if run_dir:
        cfg["output_dir"] = str(Path(run_dir) / "qlora")

    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_and_filter_dataset(dataset_path: str, min_accuracy: float) -> Dataset:
    """
    Parses dataset.json into an HF Dataset of {text, accuracy} rows.

    Quality filter: drop any entry whose top-1 accuracy (fitness col-0) is
    below min_accuracy.  Set min_accuracy=0.0 to keep everything.
    """
    with open(dataset_path) as f:
        raw = json.load(f)

    rows = []
    skipped = 0
    for entry in raw:
        try:
            fitness_vals = [float(x) for x in entry["fitness"].split(",")]
            accuracy = fitness_vals[0]
        except (KeyError, ValueError, IndexError):
            skipped += 1
            continue

        if accuracy < min_accuracy:
            skipped += 1
            continue

        text = entry["prompt"].strip() + "\n" + entry["generated_text"].strip()
        rows.append({"text": text, "accuracy": accuracy})

    log.info(
        "Loaded %d training samples (%d skipped, min_accuracy=%.4f)",
        len(rows), skipped, min_accuracy,
    )

    if not rows:
        raise ValueError(
            f"No samples passed quality filter (min_accuracy={min_accuracy}). "
            "Lower the threshold or collect more data with evolution runs."
        )

    if len(rows) < 50:
        log.warning(
            "Dataset has only %d samples — QLoRA typically needs 200+ for meaningful "
            "generalization. Run more evolution cycles to collect data first.",
            len(rows),
        )

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model / adapter helpers
# ---------------------------------------------------------------------------

def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def validate_target_modules(model, target_modules: list[str]) -> None:
    """Warn if any configured target_modules names are absent in the model."""
    actual = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [m for m in target_modules if m not in actual]
    if missing:
        log.warning(
            "target_modules %s not found in model layers. "
            "These will be silently skipped by PEFT. "
            "Check your model architecture and update configs/qlora.yaml.",
            missing,
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(ckpt_dir: Path) -> str | None:
    """Return the path to the highest-numbered checkpoint subdir, or None."""
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


# ---------------------------------------------------------------------------
# Status / metrics
# ---------------------------------------------------------------------------

def write_status(output_dir: Path, state: str, info: dict | None = None) -> None:
    payload = {
        "status": state,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        **(info or {}),
    }
    (output_dir / "status.json").write_text(json.dumps(payload, indent=2))
    log.info("Status → %s", state)


# ---------------------------------------------------------------------------
# Time-budget callback
# ---------------------------------------------------------------------------

class TimeoutCallback(TrainerCallback):
    """
    Stop training gracefully when the wall-clock budget is almost exhausted.

    Sets should_save + should_training_stop so the Trainer writes a final
    checkpoint before the SLURM job is killed.  The next job can then resume
    from that checkpoint.
    """

    def __init__(self, budget_seconds: float):
        self.deadline = time.monotonic() + budget_seconds

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if time.monotonic() >= self.deadline:
            log.warning(
                "Time budget reached at step %d — saving checkpoint and stopping "
                "so the next SLURM job can resume.",
                state.global_step,
            )
            control.should_save = True
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# OOM / NaN guard
# ---------------------------------------------------------------------------

def _check_training_health(trainer: SFTTrainer) -> None:
    logs = getattr(trainer.state, "log_history", [])
    recent = [e for e in logs if "loss" in e]
    if recent:
        last_loss = recent[-1]["loss"]
        if last_loss != last_loss:  # NaN check
            raise RuntimeError(
                "Training loss is NaN — possible causes: too-high learning rate, "
                "bad data rows, or bf16 overflow. Lower lr or check dataset."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict, resume_from_checkpoint: str | None = None) -> None:
    output_dir  = Path(cfg["output_dir"])
    ckpt_dir    = output_dir / "checkpoints"
    log_dir     = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    adapter_dir = output_dir / "adapters"

    for d in [ckpt_dir, log_dir, metrics_dir, adapter_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Auto-detect checkpoint if caller passed "auto" or nothing but a checkpoint exists
    if resume_from_checkpoint == "auto" or resume_from_checkpoint is None:
        detected = find_latest_checkpoint(ckpt_dir)
        if detected:
            log.info("Resuming from checkpoint: %s", detected)
            resume_from_checkpoint = detected
        else:
            log.info("No existing checkpoint found — starting fresh.")
            resume_from_checkpoint = None
    else:
        log.info("Resuming from checkpoint (explicit): %s", resume_from_checkpoint)

    write_status(output_dir, "running", {"resume_from_checkpoint": resume_from_checkpoint})

    # Time-budget callback: stop training with a 30-minute safety margin
    time_budget_hours = cfg.get("time_budget_hours", 15.5)
    budget_seconds = time_budget_hours * 3600
    timeout_cb = TimeoutCallback(budget_seconds)
    log.info("Time budget: %.1f hours (%.0f s)", time_budget_hours, budget_seconds)

    try:
        # ── Dataset ──────────────────────────────────────────────────────────
        dataset = load_and_filter_dataset(
            cfg.get("dataset_path", "dataset.json"),
            cfg.get("min_accuracy", 0.0),
        )

        # ── Tokenizer ────────────────────────────────────────────────────────
        model_path = cfg["model_name_or_path"]
        if not model_path:
            raise ValueError(
                "model_name_or_path is empty. Set MODEL_PATH in .env or pass --model."
            )
        log.info("Loading tokenizer from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ── Model (4-bit) ─────────────────────────────────────────────────────
        log.info("Loading model with 4-bit quantization…")
        bnb_config = build_bnb_config(cfg)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        )

        # ── LoRA adapter ──────────────────────────────────────────────────────
        lora_config = build_lora_config(cfg)
        validate_target_modules(model, lora_config.target_modules)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── Training args (SFTConfig) ─────────────────────────────────────────
        training_args = SFTConfig(
            output_dir=str(ckpt_dir),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            learning_rate=cfg.get("learning_rate", 2e-4),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=cfg.get("warmup_ratio", 0.05),
            num_train_epochs=cfg.get("num_train_epochs", 3),
            max_steps=cfg.get("max_steps", -1),
            fp16=cfg.get("fp16", False),
            bf16=cfg.get("bf16", True),
            optim=cfg.get("optim", "paged_adamw_32bit"),
            save_steps=cfg.get("save_steps", 25),
            save_total_limit=cfg.get("save_total_limit", 3),
            eval_strategy="no",
            logging_steps=cfg.get("logging_steps", 5),
            logging_dir=str(log_dir),
            report_to="none",
            max_length=cfg.get("max_seq_length", 2048),
            dataset_text_field="text",
            packing=False,
            run_name=cfg.get("run_id", "qlora"),
        )

        # ── Train ─────────────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer,
            callbacks=[timeout_cb],
        )

        log.info("Starting training (resume=%s)…", resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        _check_training_health(trainer)

        # Determine whether we timed out or genuinely finished
        timed_out = time.monotonic() >= timeout_cb.deadline
        final_ckpt = find_latest_checkpoint(ckpt_dir)

        if timed_out:
            # Save a clean checkpoint so the next job can resume
            trainer.save_model(str(ckpt_dir / f"checkpoint-timeout-{trainer.state.global_step}"))
            write_status(output_dir, "timeout", {
                "global_step": trainer.state.global_step,
                "latest_checkpoint": final_ckpt,
                "hint": "Re-submit with same QLORA_RUN_ID to continue training.",
            })
            log.info("Job timed out at step %d. Next job will resume from %s.",
                     trainer.state.global_step, final_ckpt)
            sys.exit(0)

        # ── Save adapter ──────────────────────────────────────────────────────
        log.info("Saving LoRA adapter to %s", adapter_dir)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # ── Write metrics ─────────────────────────────────────────────────────
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime_s": train_result.metrics.get("train_runtime"),
            "samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "total_steps": train_result.global_step,
            "num_train_samples": len(dataset),
            "best_checkpoint": str(ckpt_dir),
            "adapter_dir": str(adapter_dir),
        }
        (metrics_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
        log.info("Final metrics: %s", metrics)

        write_status(output_dir, "completed", metrics)
        log.info("Training complete. Adapter saved to %s", adapter_dir)

    except KeyboardInterrupt:
        write_status(output_dir, "interrupted")
        raise
    except torch.cuda.OutOfMemoryError:
        msg = (
            "CUDA OOM — try: lower per_device_train_batch_size to 1, "
            "gradient_checkpointing is already on; try reducing max_seq_length."
        )
        log.error(msg)
        write_status(output_dir, "failed", {"error": "OOM", "detail": msg})
        sys.exit(1)
    except Exception as exc:
        write_status(output_dir, "failed", {"error": str(exc)})
        log.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for EMADE evolutionary loop LLM"
    )
    parser.add_argument(
        "--config", default="configs/qlora.yaml",
        help="Path to YAML config (default: configs/qlora.yaml)",
    )
    parser.add_argument("--dataset", help="Override dataset_path from config")
    parser.add_argument("--model",   help="Override model_name_or_path (or set MODEL_PATH env)")
    parser.add_argument("--output-dir", dest="output_dir", help="Override output directory")
    parser.add_argument(
        "--max-steps", type=int, dest="max_steps",
        help="Override max training steps — set to 1-5 for a smoke test",
    )
    parser.add_argument(
        "--resume-from-checkpoint", dest="resume_from_checkpoint",
        nargs="?", const="auto", default=None,
        help=(
            "Resume training from a checkpoint. "
            "Pass no value (or 'auto') to auto-detect the latest checkpoint in the run dir. "
            "Pass an explicit path to resume from a specific checkpoint."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset_path"] = args.dataset
    if args.model:
        cfg["model_name_or_path"] = args.model
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    main(cfg, resume_from_checkpoint=args.resume_from_checkpoint)
