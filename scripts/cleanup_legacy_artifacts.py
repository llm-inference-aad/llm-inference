#!/usr/bin/env python3
"""Remove stale experimental artifacts so new runs start from a clean slate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete redundant sweep outputs and keep only the latest artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the script's parent directory).",
    )
    parser.add_argument(
        "--sota-subdir",
        type=Path,
        default=Path("sota") / "ExquisiteNetV2",
        help="Relative path to the ExquisiteNetV2 assets.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=1,
        help="Number of recent artifacts to keep in weight/models/results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without deleting anything.",
    )
    return parser.parse_args()


def remove_path(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    action = "Would remove" if dry_run else "Removing"
    print(f"{action}: {path}")
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def prune_directory(
    base_dir: Path,
    *,
    keep: int,
    protected: Iterable[str] = (),
    dry_run: bool,
) -> None:
    if not base_dir.exists():
        print(f"Skipping missing directory: {base_dir}")
        return

    protected_set = set(protected)
    candidates = [
        entry
        for entry in base_dir.iterdir()
        if entry.name not in protected_set and not entry.name.startswith(".")
    ]

    if len(candidates) <= keep:
        print(f"No pruning needed for {base_dir} (found {len(candidates)} entries).")
        return

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for entry in candidates[keep:]:
        remove_path(entry, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sota_root = (root / args.sota_subdir).resolve()
    print(f"Repository root: {root}")
    print(f"ExquisiteNetV2 root: {sota_root}")

    targets = [
        root / "0",
        root / ".venv-backup-20251015",
        sota_root / "cifar-10-batches-py",
        sota_root / "cifar-10-python",
    ]
    for target in targets:
        remove_path(target, dry_run=args.dry_run)

    prune_directory(
        sota_root / "results",
        keep=args.keep,
        dry_run=args.dry_run,
    )
    prune_directory(
        sota_root / "models",
        keep=args.keep,
        protected=("__pycache__",),
        dry_run=args.dry_run,
    )
    prune_directory(
        sota_root / "weight",
        keep=args.keep,
        protected=("seed",),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
