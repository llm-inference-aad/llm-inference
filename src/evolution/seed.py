from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from src.cfg.constants import DATA_PATH, SOTA_ROOT, TRAIN_EPOCHS
from src.utils.print_utils import box_print


def train_seed_network_baseline(
    sota_root: str = SOTA_ROOT,
    data_path: str = DATA_PATH,
) -> Optional[Tuple[float, float]]:
    """
    Train the seed network to establish a baseline fitness value.
    This ensures fitness inheritance works correctly for genes that fall back to seed code.

    Returns
    -------
    tuple or None
        (test_accuracy, num_parameters) if training succeeds, None if results already exist
    """
    seed_results_file = os.path.join(sota_root, "results", "network_results.txt")

    if os.path.exists(seed_results_file):
        box_print("SEED NETWORK BASELINE EXISTS", print_bbox_len=60, new_line_end=False)
        with open(seed_results_file, "r") as f:
            results = f.read().strip()
        print(f"  Existing results: {results}")
        try:
            parts = results.split(",")
            test_acc = float(parts[0])
            num_params = float(parts[1])
            print(f"  ✓ Seed fitness: ({test_acc:.4f}, {num_params:.0f})")
            return (test_acc, num_params)
        except Exception as exc:  # pragma: no cover - defensive parsing
            print(f"  ⚠️  Warning: Could not parse existing results: {exc}")
            print("  Will retrain seed network...")
    else:
        box_print("TRAINING SEED NETWORK BASELINE", print_bbox_len=60, new_line_end=False)
        print(f"  Seed results file not found: {seed_results_file}")
        print("  Training seed network to establish baseline...")

    train_script = os.path.join(sota_root, "train.py")
    if not os.path.exists(train_script):
        print(f"  ✗ ERROR: Training script not found: {train_script}")
        return None

    train_cmd = [
        "python",
        train_script,
        "-data",
        data_path,
        "-epoch",
        str(TRAIN_EPOCHS),
        "-network",
        "network",
        "-save_dir",
        "weight",
    ]

    print(f"  Running: {' '.join(train_cmd)}")
    print("  This may take several minutes...")

    original_cwd = Path.cwd()
    try:
        os.chdir(sota_root)
        result = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print("  ✗ ERROR: Seed network training timed out after 30 minutes")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ✗ ERROR: Unexpected error during training: {exc}")
        return None
    finally:
        os.chdir(original_cwd)

    if result.returncode != 0:
        print("  ✗ ERROR: Seed network training failed!")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    if not os.path.exists(seed_results_file):
        print(f"  ✗ ERROR: Results file not created: {seed_results_file}")
        return None

    with open(seed_results_file, "r") as f:
        results = f.read().strip()
    print("  ✓ Seed network trained successfully!")
    print(f"  Results: {results}")

    try:
        parts = results.split(",")
        test_acc = float(parts[0])
        num_params = float(parts[1])
        print(f"  ✓ Seed fitness: ({test_acc:.4f}, {num_params:.0f})")
        return (test_acc, num_params)
    except Exception as exc:  # pragma: no cover - defensive parsing
        print(f"  ⚠️  Warning: Could not parse results: {exc}")
        return None

