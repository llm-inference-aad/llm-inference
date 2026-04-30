"""Replay historical genes through the LLM with both arms (no_rag, with_rag).

For every row in `past_genes.csv` we:

1. Read the historical `[PROMPT TO LLM]` text from disk.
2. Build the with_rag arm by passing the prompt through `augment_via_rag(...)`,
   which calls `RagRuntime.enhance_template` in-process.
3. Send each arm's prompt to the LLM server, parse + validate the response,
   splice the result into the parent network at the augment_idx that the
   original prompt selected, and write `network_<new_gid>.py` into
   `sota/ExquisiteNetV2/models/`. On all-retries-fail we leave a `.fallback`
   marker (matching production augment_network behavior).
4. Submit one SLURM `train.py` job per arm, scoped to a fresh run dir so the
   resulting `<new_gid>_results.txt` lands under
   `experiments/rag_replay/<ts>/results/`.
5. Append a row per (orig_gene_id, arm) to `journal.jsonl`. Once all rows are
   queued, poll for results and append `test_acc/params/val_acc/train_time`
   onto the journal.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import socket
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
SOTA_ROOT = ROOT_DIR / "sota" / "ExquisiteNetV2"
SOTA_MODELS = SOTA_ROOT / "models"


def _import_sibling(module_name: str, file_name: str):
    """Load `02_rag_service.py` (numeric-prefixed, can't be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).parent / file_name
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # required so @dataclass can find the module
    spec.loader.exec_module(mod)
    return mod


def _new_gid(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_letters + string.digits, k=12))
    return f"xXx{prefix}{suffix}"[:27]


def _resolve_server(explicit: str | None) -> tuple[str, int]:
    if explicit:
        u = explicit.replace("http://", "").replace("https://", "")
        if ":" in u:
            host, port = u.split(":", 1)
            return host.strip().rstrip("/"), int(port.split("/")[0])
        return u.strip(), int(os.environ.get("SERVER_PORT", "8000"))
    hf = Path(os.environ.get("HOSTNAME_LOG_FILE", str(ROOT_DIR / "hostname.log")))
    if hf.exists():
        host = hf.read_text().strip().splitlines()[-1]
        return host, int(os.environ.get("SERVER_PORT", "8000"))
    raise SystemExit("Pass --server-url host:port or set HOSTNAME_LOG_FILE.")


def _ping(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _find_augment_idx(prompt: str, parts: list[str]) -> int:
    """Find which parent section appears verbatim inside the historical prompt.

    Each section in `parts` (starting at index 1; 0 is the import header) is a
    self-contained class block. The historical prompt embeds one of these inside
    a fenced ```python ... ``` code block. We pick the index whose section's
    `class <Name>` line appears in the prompt.
    """
    import re
    for idx in range(1, len(parts)):
        m = re.search(r"^class\s+(\w+)", parts[idx], re.MULTILINE)
        if not m:
            continue
        cls = m.group(1)
        if f"class {cls}" in prompt and parts[idx].strip()[:40] in prompt:
            return idx
    return 1  # fallback: first non-header section


def _splice_and_validate(parent_parts: list[str], augment_idx: int, code: str,
                         output_path: Path, gene_id: str) -> str | None:
    """Splice `code` at `augment_idx`, validate the assembled module.

    Returns None on success (output file written), else the validation error.
    """
    sys.path.insert(0, str(SRC_DIR))
    from llm_utils import extract_note, validate_module_source  # type: ignore

    note_txt = extract_note(parent_parts[augment_idx])
    candidate = parent_parts[:]
    candidate[augment_idx] = f"\n{note_txt}{code}\n"
    candidate_txt = "# --OPTION--".join(candidate)
    try:
        validate_module_source(
            candidate_txt, str(output_path), module_name=f"_llmge_{gene_id}"
        )
    except Exception as exc:
        return str(exc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(candidate_txt)
    return None


def _generate_for_arm(prompt: str, parent_path: Path, gene_id: str,
                      top_p: float, temperature: float) -> tuple[int, bool, str | None, Path]:
    """Run the production retry/validate/fallback loop with a pre-formatted prompt.

    Returns (n_attempts, was_fallback, last_error, output_path).
    """
    sys.path.insert(0, str(SRC_DIR))
    from llm_utils import generate_augmented_code, split_file  # type: ignore
    from cfg.constants import LLM_GENERATION_MAX_RETRIES  # type: ignore

    parts = split_file(str(parent_path))
    augment_idx = _find_augment_idx(prompt, parts)
    output_path = SOTA_MODELS / f"network_{gene_id}.py"

    last_error: str | None = None
    candidate_code: str | None = None
    n_attempts = 0
    fallback_marker = output_path.with_suffix(output_path.suffix + ".fallback")

    for attempt in range(LLM_GENERATION_MAX_RETRIES):
        n_attempts += 1
        try:
            candidate_code = generate_augmented_code(
                prompt,
                augment_idx - 1,
                False,
                top_p,
                temperature,
                inference_submission=False,
                gene_id=gene_id,
                previous_error=last_error,
                previous_code=candidate_code,
            )
        except Exception as exc:
            last_error = str(exc)
            break

        err = _splice_and_validate(parts, augment_idx, candidate_code, output_path, gene_id)
        if err is None:
            if fallback_marker.exists():
                try: fallback_marker.unlink()
                except OSError: pass
            return n_attempts, False, None, output_path
        last_error = err

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("# --OPTION--".join(parts))
    try:
        fallback_marker.write_text((last_error or "unknown").strip() or "unknown")
    except OSError:
        pass
    return n_attempts, True, last_error, output_path


def _render_sbatch(new_gid: str, run_dir: Path, slurm_log_dir: Path,
                   slurm_err_dir: Path, epochs: int, data_path: str) -> str:
    sys.path.insert(0, str(SRC_DIR))
    from cfg.constants import PYTHON_BASH_SCRIPT_TEMPLATE  # type: ignore

    python_runline = (
        f'python {ROOT_DIR}/sota/ExquisiteNetV2/train.py '
        f'-bs 216 -network "models.network_{new_gid}" '
        f'-data {data_path} -end_lr 0.001 -seed 21 -val_r 0.2 -amp '
        f'-epoch {epochs}'
    )
    return PYTHON_BASH_SCRIPT_TEMPLATE.format(
        python_runline=python_runline,
        slurm_log_dir=str(slurm_log_dir),
        slurm_error_dir=str(slurm_err_dir),
        root_dir=str(ROOT_DIR),
    )


def _submit(bash_script: str, run_dir: Path, sbatch_dir: Path,
            new_gid: str) -> tuple[str | None, str | None]:
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    script_path = sbatch_dir / f"{new_gid}.sh"
    script_path.write_text(bash_script)
    env = os.environ.copy()
    # Absolute path: train.py auto-chdir to sota/ExquisiteNetV2/ would resolve
    # a relative RUN_DIR under that subdir, hiding results from the poll loop.
    env["RUN_DIR"] = str(run_dir.resolve())
    env["LLM_INFERENCE_ROOT_DIR"] = str(ROOT_DIR)
    cmd = ["sbatch", "--parsable", "--export=ALL", str(script_path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    except FileNotFoundError:
        return None, "sbatch not on PATH"
    if out.returncode != 0:
        return None, (out.stderr or out.stdout).strip()
    return out.stdout.strip(), None


def _journal_append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _journal_iter(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _slurm_state(job_id: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", job_id, "--format=State", "-n", "-X"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    states = [s.strip() for s in out.splitlines() if s.strip()]
    return states[0] if states else None


def _parse_results(path: Path) -> tuple[float, float, float | None, float | None] | None:
    try:
        raw = path.read_text().strip()
    except Exception:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 2:
        return None
    try:
        test_acc = float(parts[0]); params = float(parts[1])
        val_acc = float(parts[2]) if len(parts) > 2 and parts[2] else None
        train_time = float(parts[3]) if len(parts) > 3 and parts[3] else None
        return test_acc, params, val_acc, train_time
    except Exception:
        return None


_CUDA_BUSY_PAT = "CUDA-capable device(s) is/are busy or unavailable"


def _is_transient_cuda(slurm_err_dir: Path, job_id: str) -> bool:
    """Detect the transient driver-busy failure that's worth one retry."""
    if not job_id:
        return False
    candidate = slurm_err_dir / f"eval-{job_id}.err"
    if not candidate.exists():
        return False
    try:
        return _CUDA_BUSY_PAT in candidate.read_text(errors="replace")
    except Exception:
        return False


def _resubmit(sbatch_dir: Path, run_dir: Path, new_gid: str) -> tuple[str | None, str | None]:
    script_path = sbatch_dir / f"{new_gid}.sh"
    if not script_path.exists():
        return None, f"missing sbatch script {script_path}"
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir.resolve())
    env["LLM_INFERENCE_ROOT_DIR"] = str(ROOT_DIR)
    out = subprocess.run(["sbatch", "--parsable", "--export=ALL", str(script_path)],
                         capture_output=True, text=True, env=env, check=False)
    if out.returncode != 0:
        return None, (out.stderr or out.stdout).strip()
    return out.stdout.strip(), None


def _poll_results(journal_path: Path, run_dir: Path,
                  results_path: Path, timeout_s: float) -> None:
    """Walk the journal, fill in test_acc/params for any pending entries."""
    slurm_err_dir = run_dir / "slurm_errors"
    sbatch_dir = run_dir / "sbatch"
    pending: dict[str, dict] = {}
    final_rows: list[dict] = []
    for row in _journal_iter(journal_path):
        if row.get("status") == "queued" and "test_acc" not in row:
            pending[row["new_gene_id"]] = row
        else:
            final_rows.append(row)
    print(f"[poll] {len(pending)} pending evals", flush=True)

    deadline = time.time() + timeout_s
    while pending and time.time() < deadline:
        for gid in list(pending):
            row = pending[gid]
            results_file = results_path / f"{gid}_results.txt"
            parsed = _parse_results(results_file)
            if parsed is not None:
                ta, pp, va, tt = parsed
                row.update(test_acc=ta, params=pp, val_acc=va, train_time_s=tt,
                           status="done", finished_at=datetime.now(timezone.utc).isoformat())
                final_rows.append(row); pending.pop(gid)
                continue
            jid = row.get("slurm_job_id")
            if jid:
                state = _slurm_state(jid)
                if state and state.upper() in {"FAILED", "TIMEOUT", "CANCELLED",
                                               "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL"}:
                    if (not row.get("cuda_retry")
                            and _is_transient_cuda(slurm_err_dir, jid)):
                        new_jid, err = _resubmit(sbatch_dir, run_dir, gid)
                        if new_jid:
                            print(f"[poll] {gid}: CUDA-busy on {jid}, resubmitted as {new_jid}", flush=True)
                            row["cuda_retry"] = True
                            row["prev_slurm_job_id"] = jid
                            row["slurm_job_id"] = new_jid
                            continue
                        else:
                            print(f"[poll] {gid}: CUDA-busy retry failed: {err}", flush=True)
                    row.update(status="failed", slurm_state=state,
                               finished_at=datetime.now(timezone.utc).isoformat())
                    final_rows.append(row); pending.pop(gid)
        if pending:
            time.sleep(60)

    for row in pending.values():
        row.setdefault("status", "timeout")
        final_rows.append(row)

    journal_path.write_text("")
    for row in final_rows:
        _journal_append(journal_path, row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=Path(__file__).parent / "datasets" / "past_genes.csv")
    ap.add_argument("--output", type=Path, required=True,
                    help="Replay run directory (will be created)")
    ap.add_argument("--server-url", default=None)
    ap.add_argument("--epochs", type=int,
                    default=int(os.environ.get("EPOCHS", "24")))
    ap.add_argument("--data-path",
                    default=os.environ.get("DATA_PATH", "cifar10"))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--eligible-only", action="store_true",
                    help="Skip genes with orig_eligible_for_rag=False")
    ap.add_argument("--skip-arm", choices=["no_rag", "with_rag"], default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-poll", action="store_true",
                    help="Submit and exit; run 03_replay.py with --poll-only later.")
    ap.add_argument("--poll-only", action="store_true")
    ap.add_argument("--poll-timeout-hours", type=float, default=12.0)
    args = ap.parse_args()

    # Load .env so VENV_PATH, DATA_PATH, LLM_INFERENCE_ROOT_DIR, etc. propagate
    # into the env that gets handed to the spawned sbatch jobs via --export=ALL.
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Don't clobber values explicitly already set in the parent env
            os.environ.setdefault(k, os.path.expandvars(v))

    args.output.mkdir(parents=True, exist_ok=True)
    journal_path = args.output / "journal.jsonl"
    results_path = args.output / "results"
    slurm_log_dir = args.output / "slurm_logs"
    slurm_err_dir = args.output / "slurm_errors"
    sbatch_dir = args.output / "sbatch"
    for d in [results_path, slurm_log_dir, slurm_err_dir, sbatch_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.poll_only:
        _poll_results(journal_path, args.output, results_path,
                      timeout_s=args.poll_timeout_hours * 3600)
        return 0

    # Resolve LLM server before importing src/ modules
    host, port = _resolve_server(args.server_url)
    if not _ping(host, port):
        raise SystemExit(f"LLM server at {host}:{port} not reachable.")
    os.environ["SERVER_PORT"] = str(port)
    hostname_log = args.output / "hostname.log"
    hostname_log.write_text(host + "\n")
    os.environ["HOSTNAME_LOG_FILE"] = str(hostname_log)
    os.environ.setdefault("USE_LOAD_BALANCER", "false")
    os.environ["RAG_ENABLED"] = "true"  # so RagRuntime won't soft-disable
    os.environ.setdefault("RUN_LOG_DIR", str(args.output / "logs"))
    os.environ.setdefault("RUN_METRICS_DIR", str(args.output / "metrics"))
    os.environ.setdefault("RUN_ERRORS_DIR", str(args.output / "logs" / "errors"))
    Path(os.environ["RUN_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["RUN_METRICS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["RUN_ERRORS_DIR"]).mkdir(parents=True, exist_ok=True)

    rag_service = _import_sibling("rag_service", "02_rag_service.py")
    print(f"[setup] warming RagRuntime...", flush=True)
    rag_service.warmup()
    print(f"[setup] runtime ready", flush=True)

    arms_to_run = ["no_rag", "with_rag"]
    if args.skip_arm:
        arms_to_run = [a for a in arms_to_run if a != args.skip_arm]

    rows = list(csv.DictReader(args.csv.open()))
    if args.eligible_only:
        rows = [r for r in rows if r["orig_eligible_for_rag"] == "True"]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    print(f"[run] {len(rows)} source genes × {len(arms_to_run)} arms = "
          f"{len(rows) * len(arms_to_run)} jobs to submit", flush=True)

    metadata_path = args.output / "run_metadata.json"
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "csv": str(args.csv), "epochs": args.epochs, "data_path": args.data_path,
        "server_url": f"http://{host}:{port}", "arms": arms_to_run,
        "eligible_only": args.eligible_only, "temperature": args.temperature,
        "top_p": args.top_p,
        "rag_env": {k: os.environ.get(k) for k in
                    ["RAG_USE_CODE_CONTEXT", "RAG_USE_TEXT_CONTEXT",
                     "RAG_TOP_K", "RAG_TEXT_TOP_K", "RAG_MIN_SIMILARITY",
                     "RAG_RERANKER_ENABLED"]},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    for i, row in enumerate(rows, start=1):
        orig_gid = row["orig_gene_id"]
        prompt = (ROOT_DIR / row["orig_prompt_path"]).read_text(encoding="utf-8")
        parent_path = ROOT_DIR / row["orig_parent_path"]
        if not parent_path.exists():
            parent_path = SOTA_ROOT / "network.py"
        parent_code = parent_path.read_text(encoding="utf-8")

        for arm in arms_to_run:
            new_gid = _new_gid(arm[0])
            if arm == "with_rag":
                req = rag_service.AugmentRequest(
                    template=prompt,
                    mutation_type=row.get("orig_mutation_op") or None,
                    query_code=parent_code,
                    gene_id=new_gid,
                )
                resp = rag_service.augment_via_rag(req)
                arm_prompt = resp.augmented_template
                retrieval = {
                    "retrieved_n_code": resp.retrieved_n_code,
                    "retrieved_n_text": resp.retrieved_n_text,
                    "rag_block_chars": resp.rag_block_chars,
                }
            else:
                arm_prompt = prompt
                retrieval = {"retrieved_n_code": 0, "retrieved_n_text": 0,
                             "rag_block_chars": 0}

            t_llm = time.perf_counter()
            n_attempts, fallback, err, network_path = _generate_for_arm(
                prompt=arm_prompt, parent_path=parent_path, gene_id=new_gid,
                top_p=args.top_p, temperature=args.temperature,
            )
            llm_wall_s = time.perf_counter() - t_llm

            bash = _render_sbatch(new_gid, args.output, slurm_log_dir,
                                  slurm_err_dir, args.epochs, args.data_path)
            job_id, submit_err = _submit(bash, args.output, sbatch_dir, new_gid)

            entry = {
                "orig_gene_id": orig_gid, "orig_run_id": row["orig_run_id"],
                "orig_was_fallback": row["orig_was_fallback"] == "True",
                "orig_test_acc": float(row["orig_test_acc"]) if row.get("orig_test_acc") else None,
                "orig_params": float(row["orig_params"]) if row.get("orig_params") else None,
                "orig_train_time_s": float(row["orig_train_time_s"]) if row.get("orig_train_time_s") else None,
                "arm": arm, "new_gene_id": new_gid,
                "n_attempts": n_attempts, "was_fallback": fallback,
                "syntax_valid_first_try": (n_attempts == 1 and not fallback),
                "llm_wall_s": llm_wall_s,
                "prompt_chars": len(arm_prompt),
                "error_msg": err, **retrieval,
                "slurm_job_id": job_id, "submit_error": submit_err,
                "network_path": str(network_path.relative_to(ROOT_DIR)),
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "status": "queued" if job_id else "submit_failed",
            }
            _journal_append(journal_path, entry)
            print(f"[{i}/{len(rows)}] {orig_gid[:12]} {arm:8s} "
                  f"new_gid={new_gid} n={n_attempts} fb={fallback} "
                  f"job={job_id or 'X'} ret={retrieval['retrieved_n_code']}/{retrieval['retrieved_n_text']}",
                  flush=True)

    metadata["queued_at"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if args.no_poll:
        print(f"[done-queue] All jobs queued. Run with --poll-only to collect.",
              flush=True)
        return 0

    _poll_results(journal_path, args.output, results_path,
                  timeout_s=args.poll_timeout_hours * 3600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
