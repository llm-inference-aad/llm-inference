"""Per-case execution for RAG isolation paired evaluation.

The harness pairs every (parent, template, augment_idx) "case" across two arms —
``no_rag`` and ``with_rag`` — keeping every other input constant. Both arms call
the production ``augment_network`` from ``src/llm_mutation.py`` so the captured
metrics reflect what users actually experience.

This module is intentionally side-effect heavy: it sets ``RUN_LOG_DIR`` /
``RUN_METRICS_DIR`` per trial, parses the ``validation_errors.csv`` that
``augment_network`` writes, and reads the per-trial network file off disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add repo "src" to sys.path so we can import production modules
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class CaseSpec:
    case_id: str
    parent: str          # repo-relative path
    template: str        # repo-relative path
    include_constant_rules: bool = True
    mutation_type: str | None = None  # for RAG retrieval; defaults to template stem


@dataclass
class TrialConfig:
    trials_per_case: int = 3
    temperature: float = 0.3
    top_p: float = 0.95
    max_new_tokens: int = 4096
    base_seed: int = 1234
    rag_use_code_context: bool = True
    rag_use_text_context: bool = True


@dataclass
class TrialResult:
    case_id: str
    trial: int
    arm: str
    gene_id: str
    parent: str
    template: str
    mutation_type: str
    augment_idx: int = -1
    syntax_valid_first_try: bool = False
    module_valid_first_try: bool = False
    n_attempts: int = 0
    fallback: bool = False
    error_types_per_attempt: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    response_chars: int = 0
    llm_latency_s: float = 0.0
    retrieved_n_code: int = 0
    retrieved_n_text: int = 0
    rag_block_chars: int = 0
    parent_changed: bool = False
    code_path: str = ""
    wall_s_total: float = 0.0
    error: str | None = None
    # Fitness columns — populated by collect_fitness.py, not by execute_trial.
    fitness_acc: float | None = None
    fitness_params: int | None = None
    fitness_inherited_from: str | None = None
    eval_job_id: str | None = None
    eval_status: str | None = None
    eval_train_seconds: float | None = None
    eval_val_acc: float | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@contextmanager
def _scoped_env(env: dict[str, str]):
    """Apply env overrides for the duration of the with-block."""
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _derive_seed(base_seed: int, case_id: str, trial: int) -> int:
    """Deterministic per-(case, trial) seed via hashing."""
    h = hashlib.sha256(f"{base_seed}|{case_id}|{trial}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_raw_template(case: CaseSpec, root: Path) -> str:
    """Read the template file and append ConstantRules.txt if requested."""
    template_text = (root / case.template).read_text(encoding="utf-8")
    if case.include_constant_rules:
        rules_text = (root / "templates/ConstantRules.txt").read_text(encoding="utf-8")
        template_text = f"{template_text}\n{rules_text}"
    return template_text


def _query_code_for_retrieval(case: CaseSpec, root: Path, augment_idx: int) -> str:
    """Pull out the parent code section that will be templated, for RAG retrieval."""
    from llm_utils import split_file  # type: ignore

    parts = split_file(str(root / case.parent))
    idx = max(1, min(augment_idx, len(parts) - 1))
    return parts[idx].strip()


def _peek_augment_idx(parent_path: Path, seed: int) -> int:
    """Predict the augment_idx that augment_network will pick under this seed."""
    from llm_utils import split_file  # type: ignore

    parts = split_file(str(parent_path))
    rng = np.random.RandomState(seed)
    return int(rng.randint(1, len(parts)))


def _parse_validation_errors(csv_path: Path, gene_id: str) -> list[str]:
    """Pull error type strings written for this gene_id during augment_network."""
    if not csv_path.exists():
        return []
    out: list[str] = []
    try:
        with csv_path.open() as f:
            for row in csv.reader(f):
                if len(row) >= 5 and row[1] == gene_id:
                    out.append(row[3])
    except Exception:
        pass
    return out


def _resolve_runtime():
    """Force-construct a RagRuntime without going through ``RAG_ENABLED`` env gating.

    We always need the runtime instantiated so we can call ``enhance_template``
    for the with_rag arm. We bypass ``get_runtime()``'s soft-disable so that
    ``RAG_ENABLED=false`` (which we toggle for the no_rag arm) doesn't kill us.
    """
    from rag.runtime import RagRuntime  # type: ignore

    return RagRuntime()


def execute_trial(
    case: CaseSpec,
    trial: int,
    arm: str,
    cfg: TrialConfig,
    root: Path,
    out_root: Path,
    runtime,
) -> TrialResult:
    """Run one (case, trial, arm) and return the recorded metrics.

    The trial-scoped env vars point ``augment_network``'s logging at our
    output dir so the per-trial validation_errors.csv stays isolated.
    """
    seed = _derive_seed(cfg.base_seed, case.case_id, trial)
    gene_id = f"{case.case_id}_{trial:02d}_{arm}"

    trial_dir = out_root / "cases" / case.case_id / f"{trial:02d}" / arm
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Predict augment_idx (augment_network seeds np.random itself; we mirror it).
    parent_path = root / case.parent
    augment_idx = _peek_augment_idx(parent_path, seed)

    # Build raw and (optionally) RAG-augmented templates.
    raw_template = _build_raw_template(case, root)
    mutation_type = case.mutation_type or Path(case.template).stem

    if arm == "with_rag":
        query_code = _query_code_for_retrieval(case, root, augment_idx)
        t0 = time.perf_counter()
        augmented_template, mutations = runtime.enhance_template(
            template=raw_template,
            mutation_type=mutation_type,
            query_code=query_code,
            gene_id=gene_id,
        )
        retrieval_s = time.perf_counter() - t0
        text_context_n = sum(1 for m in mutations if getattr(m, "kind", None) == "text")
        code_context_n = len(mutations) - text_context_n
        # PromptEnhancer returns only mutations (code retrievals); text is folded
        # into the augmented template directly. We capture the size delta as the
        # measurable signal of "how much RAG added".
        final_template = augmented_template
        rag_block_chars = max(0, len(final_template) - len(raw_template))
    else:
        final_template = raw_template
        retrieval_s = 0.0
        code_context_n = 0
        text_context_n = 0
        rag_block_chars = 0

    # Write the chosen template to a file augment_network can open.
    template_path = trial_dir / "prompt_template.txt"
    template_path.write_text(final_template, encoding="utf-8")
    # Also persist the raw template for diffing later.
    (trial_dir / "prompt_template.raw.txt").write_text(raw_template, encoding="utf-8")

    output_network = trial_dir / "network.py"

    # Scope augment_network's runtime artifacts into our trial dir.
    trial_log_dir = trial_dir / "logs"
    trial_metrics_dir = trial_dir / "metrics"
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    trial_metrics_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "RUN_LOG_DIR": str(trial_log_dir),
        "RUN_METRICS_DIR": str(trial_metrics_dir),
        # Force constants module to re-resolve dirs lazily — but it doesn't, so
        # we set them BEFORE first import. The contextmanager below is cosmetic
        # for env vars actually consulted at runtime (HOSTNAME_LOG_FILE etc.).
    }

    # Capture llm.log per-gene to estimate response length and attempt count.
    gene_log_path = trial_log_dir / "llm" / f"gene_{gene_id}.log"
    if gene_log_path.exists():
        gene_log_path.unlink()

    t_start = time.perf_counter()
    error_msg: str | None = None
    with _scoped_env(env):
        from llm_mutation import augment_network  # type: ignore

        # Reset numpy seed so augment_network picks the predicted augment_idx.
        np.random.seed(seed)
        try:
            augment_network(
                input_filename=str(parent_path),
                output_filename=str(output_network),
                template_txt=str(template_path),
                top_p=cfg.top_p,
                temperature=cfg.temperature,
                apply_quality_control=False,
                inference_submission=False,
                gene_id=gene_id,
            )
        except Exception as exc:  # pragma: no cover - upstream code is broad
            error_msg = f"{type(exc).__name__}: {exc}"
    wall_s = time.perf_counter() - t_start

    # Did augment_network mark a fallback?
    fallback_marker = output_network.with_suffix(output_network.suffix + ".fallback")
    fallback = fallback_marker.exists()

    # Pull validation errors that augment_network wrote.
    error_types = _parse_validation_errors(trial_log_dir / "validation_errors.csv", gene_id)
    n_attempts = len(error_types) + (0 if fallback and error_msg else 1)
    if error_msg and not fallback and not error_types:
        n_attempts = 0  # full failure (e.g., LLM call exception)
    syntax_valid_first_try = (not fallback) and (len(error_types) == 0) and (error_msg is None)
    module_valid_first_try = syntax_valid_first_try

    # Estimate prompt+response sizes from the per-gene LLM log.
    response_chars = 0
    prompt_chars = 0
    llm_latency_s = wall_s  # total wall as a fallback; refine below if log present
    if gene_log_path.exists():
        log_text = gene_log_path.read_text(encoding="utf-8", errors="ignore")
        # Sum all PROMPT and RAW response sections (1 per attempt).
        prompt_chars = sum(len(b) for b in _extract_blocks(log_text, "PROMPT TO LLM"))
        response_chars = sum(len(b) for b in _extract_blocks(log_text, "TEXT FROM LLM (RAW)"))

    parent_changed = False
    if output_network.exists() and not fallback:
        parent_changed = _file_sha256(output_network) != _file_sha256(parent_path)

    return TrialResult(
        case_id=case.case_id,
        trial=trial,
        arm=arm,
        gene_id=gene_id,
        parent=case.parent,
        template=case.template,
        mutation_type=mutation_type,
        augment_idx=augment_idx,
        syntax_valid_first_try=syntax_valid_first_try,
        module_valid_first_try=module_valid_first_try,
        n_attempts=max(1, n_attempts),
        fallback=fallback,
        error_types_per_attempt=error_types,
        prompt_chars=prompt_chars or len(final_template),
        response_chars=response_chars,
        llm_latency_s=llm_latency_s,
        retrieved_n_code=code_context_n,
        retrieved_n_text=text_context_n,
        rag_block_chars=rag_block_chars,
        parent_changed=parent_changed,
        code_path=str(output_network.relative_to(root)) if output_network.exists() else "",
        wall_s_total=wall_s,
        error=error_msg,
    )


def _extract_blocks(log_text: str, marker: str) -> list[str]:
    """Pull out content between log section dividers for a given marker."""
    blocks: list[str] = []
    sep = "=" * 80
    sections = log_text.split(sep)
    for section in sections:
        if marker in section:
            # Content lives after the marker line and dashed sub-divider.
            parts = section.split("-" * 80, 1)
            if len(parts) == 2:
                blocks.append(parts[1].strip())
    return blocks


def load_dataset(path: Path) -> tuple[list[CaseSpec], TrialConfig]:
    """Load a dataset JSON file into structured Python objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        CaseSpec(
            case_id=c["case_id"],
            parent=c["parent"],
            template=c["template"],
            include_constant_rules=c.get("include_constant_rules", True),
            mutation_type=c.get("mutation_type"),
        )
        for c in raw["cases"]
    ]
    cfg = TrialConfig(
        trials_per_case=raw.get("trials_per_case", 3),
        temperature=raw.get("temperature", 0.3),
        top_p=raw.get("top_p", 0.95),
        max_new_tokens=raw.get("max_new_tokens", 4096),
        base_seed=raw.get("base_seed", 1234),
        rag_use_code_context=raw.get("rag_use_code_context", True),
        rag_use_text_context=raw.get("rag_use_text_context", True),
    )
    return cases, cfg


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
