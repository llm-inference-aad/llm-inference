#!/usr/bin/env python3
"""
Build a deterministic golden query dataset for text-namespace retrieval evaluation.

Outputs `src/rag/golden_queries.json` with two tiers:
- small: the original 13 hand-curated smoke queries
- medium: N auto-generated queries from `rag_corpus/pytorch.json`

This script uses only the Python standard library (no torch/deap/sentence-transformers).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


TIER_SMALL = [
    {"query": "convolutional layer 2d", "expected": ["torch.nn.Conv2d"]},
    {"query": "max pooling 2d", "expected": ["torch.nn.MaxPool2d"]},
    {"query": "cross entropy loss function", "expected": ["torch.nn.CrossEntropyLoss"]},
    {"query": "adam optimizer", "expected": ["torch.optim.Adam"]},
    {"query": "sgd optimizer", "expected": ["torch.optim.SGD"]},
    {"query": "resnet18 model architecture", "expected": ["torchvision.models.resnet18"]},
    {
        "query": "random horizontal flip data augmentation",
        "expected": ["torchvision.transforms.RandomHorizontalFlip"],
    },
    {"query": "bce with logits loss", "expected": ["torch.nn.BCEWithLogitsLoss"]},
    {"query": "linear layer fully connected", "expected": ["torch.nn.Linear"]},
    {"query": "relu activation", "expected": ["torch.nn.ReLU"]},
    {"query": "dropout regularization", "expected": ["torch.nn.Dropout"]},
    {"query": "mobile net v2", "expected": ["torchvision.models.mobilenet_v2"]},
    {"query": "cosine annealing scheduler", "expected": ["torch.optim.lr_scheduler.CosineAnnealingLR"]},
]


def _decamel(name: str) -> str:
    # Split CamelCase + digits into tokens.
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", name)
    return " ".join(p.lower() for p in parts if p)


def _keywordize(full_name: str) -> str:
    last = full_name.split(".")[-1]
    base = _decamel(last)
    out = base
    out = re.sub(r"\blr\b", "learning rate", out)
    out = re.sub(r"\bconv\b", "convolution", out)
    out = re.sub(r"\bpool\b", "pooling", out)
    out = re.sub(r"\bbn\b", "batch norm", out)
    return out


def _doc_first_line(doc: str) -> str:
    if not doc:
        return ""
    line = doc.strip().splitlines()[0].strip()
    line = re.sub(r"\s+", " ", line)
    return line[:160]


def _choose_medium_entries(raw: list[dict], limit: int) -> list[dict]:
    candidates = [e for e in raw if e.get("name")]
    candidates.sort(key=lambda e: e.get("name", ""))

    # Spread picks across top-level modules to avoid overfitting to torch.nn only.
    by_module: dict[str, list[dict]] = {}
    for entry in candidates:
        name = entry["name"]
        module = ".".join(name.split(".")[:2]) if "." in name else "misc"
        by_module.setdefault(module, []).append(entry)

    modules = sorted(by_module)
    selected: list[dict] = []
    idx = 0
    while len(selected) < limit and modules:
        module = modules[idx % len(modules)]
        if by_module[module]:
            selected.append(by_module[module].pop(0))
        else:
            modules.remove(module)
            continue
        idx += 1
    return selected


def build_golden_queries(pytorch_json: Path, *, medium_limit: int, seed: int) -> dict:
    raw = json.loads(pytorch_json.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    medium_entries = _choose_medium_entries(raw, medium_limit)
    # Add a stable shuffle for extra variety while keeping determinism.
    rng.shuffle(medium_entries)

    tier_medium: list[dict] = []
    for entry in medium_entries:
        name = entry["name"]
        doc = entry.get("docstring") or ""

        q1 = _keywordize(name)
        q2 = _decamel(name.split(".")[-1])
        q3 = _doc_first_line(doc)

        queries: list[str] = []
        for q in (q1, q2, q3):
            q = re.sub(r"\s+", " ", (q or "")).strip()
            if q and q not in queries:
                queries.append(q)

        query = queries[0] if queries else name
        if len(query.split()) < 3:
            query = f"pytorch {query}"

        tier_medium.append({"query": query, "expected": [name]})

    return {
        "version": 1,
        "source": str(pytorch_json),
        "tiers": {
            "small": TIER_SMALL,
            "medium": tier_medium,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch-json",
        default="rag_corpus/pytorch.json",
        help="Path to pytorch.json corpus (from src/rag/indexer.py).",
    )
    parser.add_argument(
        "--output",
        default="src/rag/golden_queries.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--medium-limit",
        type=int,
        default=200,
        help="Number of auto-generated medium-tier queries.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed.")
    args = parser.parse_args()

    pytorch_json = Path(args.pytorch_json)
    if not pytorch_json.exists():
        raise SystemExit(f"Missing corpus file: {pytorch_json}")

    payload = build_golden_queries(pytorch_json, medium_limit=args.medium_limit, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (small={len(payload['tiers']['small'])}, medium={len(payload['tiers']['medium'])})")


if __name__ == "__main__":
    main()

