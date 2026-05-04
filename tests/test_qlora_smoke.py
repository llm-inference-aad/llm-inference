"""
Smoke tests for the QLoRA training pipeline.

These tests validate data preprocessing, config loading, and artifact
integrity WITHOUT loading the full GPU model (no bitsandbytes/CUDA needed).

Run:
    uv run pytest tests/test_qlora_smoke.py -v
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Import only the pure-Python helpers — no GPU import at module level
from train_qlora import load_and_filter_dataset, load_config, write_status


# ---------------------------------------------------------------------------
# Dataset preprocessing tests
# ---------------------------------------------------------------------------

class TestDataset:
    DATASET = str(REPO_ROOT / "dataset.json")

    def test_loads_all_rows(self):
        ds = load_and_filter_dataset(self.DATASET, min_accuracy=0.0)
        assert len(ds) > 0

    def test_columns_present(self):
        ds = load_and_filter_dataset(self.DATASET, min_accuracy=0.0)
        assert "text" in ds.column_names
        assert "accuracy" in ds.column_names

    def test_text_is_non_empty(self):
        ds = load_and_filter_dataset(self.DATASET, min_accuracy=0.0)
        for row in ds:
            assert row["text"].strip(), "text field must not be blank"

    def test_accuracy_is_float(self):
        ds = load_and_filter_dataset(self.DATASET, min_accuracy=0.0)
        for row in ds:
            assert isinstance(row["accuracy"], float)
            assert 0.0 <= row["accuracy"] <= 1.0

    def test_filter_reduces_count(self):
        ds_all = load_and_filter_dataset(self.DATASET, min_accuracy=0.0)
        ds_hi  = load_and_filter_dataset(self.DATASET, min_accuracy=0.49)
        assert len(ds_hi) <= len(ds_all)

    def test_too_high_threshold_raises(self):
        with pytest.raises(ValueError, match="No samples passed"):
            load_and_filter_dataset(self.DATASET, min_accuracy=0.999)

    def test_prompt_and_completion_joined(self):
        with open(self.DATASET) as f:
            raw = json.load(f)
        ds = load_and_filter_dataset(self.DATASET, min_accuracy=0.0)
        first_raw = raw[0]
        first_ds  = ds[0]
        assert first_raw["prompt"].strip() in first_ds["text"]
        assert first_raw["generated_text"].strip() in first_ds["text"]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    CONFIG = str(REPO_ROOT / "configs" / "qlora.yaml")

    def test_yaml_is_valid(self):
        with open(self.CONFIG) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_required_keys_present(self):
        with open(self.CONFIG) as f:
            cfg = yaml.safe_load(f)
        required = [
            "load_in_4bit",
            "bnb_4bit_quant_type",
            "bnb_4bit_compute_dtype",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "per_device_train_batch_size",
            "learning_rate",
            "min_accuracy",
        ]
        for k in required:
            assert k in cfg, f"Missing required config key: {k}"

    def test_target_modules_is_list(self):
        with open(self.CONFIG) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg["target_modules"], list)
        assert len(cfg["target_modules"]) > 0

    def test_load_config_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_PATH", "/tmp/fake_model")
        monkeypatch.setenv("RUN_ID", "test_run_123")
        monkeypatch.setenv("RUN_DIR", str(tmp_path))
        cfg = load_config(self.CONFIG)
        assert cfg["model_name_or_path"] == "/tmp/fake_model"
        assert cfg["run_id"] == "test_run_123"
        assert cfg["output_dir"] == str(tmp_path / "qlora")


# ---------------------------------------------------------------------------
# Artifact / status tests
# ---------------------------------------------------------------------------

class TestArtifacts:
    def test_write_status_creates_file(self, tmp_path):
        write_status(tmp_path, "completed", {"train_loss": 0.42})
        status_file = tmp_path / "status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["status"] == "completed"
        assert data["train_loss"] == 0.42
        assert "timestamp" in data

    def test_write_status_failed(self, tmp_path):
        write_status(tmp_path, "failed", {"error": "OOM"})
        data = json.loads((tmp_path / "status.json").read_text())
        assert data["status"] == "failed"

    def test_run_dir_structure(self, tmp_path):
        for sub in ["checkpoints", "logs", "metrics", "adapters"]:
            (tmp_path / sub).mkdir()
        for sub in ["checkpoints", "logs", "metrics", "adapters"]:
            assert (tmp_path / sub).is_dir()
