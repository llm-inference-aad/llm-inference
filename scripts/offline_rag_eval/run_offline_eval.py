#!/usr/bin/env python3
"""Run rendered (baseline, RAG) prompt pairs against the LLM server.

Reads paired JSONL (e.g. from `poc_prompt_compare.py` or any tool emitting the
same row shape) and POSTs each row to the
`/generate` endpoint of the local LLM server. The server's response and a
local `OutputEvaluator` score are appended to a results JSONL.

Server discovery order:
1. `--server-url` CLI flag.
2. `LLM_SERVER_URL` environment variable.
3. `runs/<RUN_ID>/logs/loadbalancer.log` if `USE_LOAD_BALANCER` is set.
4. `runs/<RUN_ID>/logs/hostname.log` (or `HOSTNAME_LOG_FILE` env var).
5. The repo-level `hostname.log` file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator import OutputEvaluator  # noqa: E402  pylint: disable=wrong-import-position


@dataclass(frozen=True)
class ServerConfig:
    url: str
    max_new_tokens: int
    temperature: float
    top_p: float
    timeout_seconds: float
    max_retries: int


def _read_first_line(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text.splitlines()[0].strip() if text else None


def discover_server_url(*, cli_url: str | None, run_id: str | None, port: str) -> str:
    if cli_url:
        return cli_url.rstrip("/")
    env_url = os.environ.get("LLM_SERVER_URL")
    if env_url:
        return env_url.rstrip("/")

    candidates: list[Path] = []
    use_lb = os.environ.get("USE_LOAD_BALANCER", "false").lower() in {"1", "true", "yes"}
    if run_id:
        run_log_dir = REPO_ROOT / "runs" / run_id / "logs"
        if use_lb:
            candidates.append(run_log_dir / "loadbalancer.log")
        candidates.append(run_log_dir / "hostname.log")

    explicit_hostname_file = os.environ.get("HOSTNAME_LOG_FILE")
    if explicit_hostname_file:
        candidates.append(Path(explicit_hostname_file))

    candidates.append(REPO_ROOT / "hostname.log")

    for candidate in candidates:
        host = _read_first_line(candidate)
        if host:
            return f"http://{host}:{port}"

    raise RuntimeError(
        "Could not determine server URL. Pass --server-url, set "
        "LLM_SERVER_URL, or ensure hostname.log exists for the run."
    )


def _post_with_retries(
    url: str, payload: dict, *, timeout: float, max_retries: int
) -> tuple[dict, int]:
    """POST with backoff; returns (json_body, 1-based attempt index on success)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Server returned {response.status_code}: {response.text[:200]}"
                )
            return response.json(), attempt
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
        except Exception as exc:  # noqa: BLE001 - bubbled up below
            last_exc = exc
            break

        if attempt < max_retries:
            backoff = min(2 * attempt, 15)
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}")


def _evaluate_response(
    generated_text: str,
    code_snippet: str,
    *,
    validate_module: bool,
) -> dict:
    score = OutputEvaluator.calculate_evaluation_score(generated_text or "")
    code_blocks = OutputEvaluator.extract_code_blocks(generated_text or "")

    snippet_norm = (code_snippet or "").strip()
    differs_from_input = (
        bool(code_blocks)
        and bool(snippet_norm)
        and snippet_norm not in {block.strip() for block in code_blocks}
    )

    cheap = OutputEvaluator.cheap_mutation_metrics(
        generated_text or "", validate_module=validate_module
    )

    return {
        "eval_score": score,
        "n_code_blocks": len(code_blocks),
        "differs_from_input": differs_from_input,
        **cheap,
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run(
    rendered_path: Path,
    output_path: Path,
    server: ServerConfig,
    *,
    job_id: str,
    splits: tuple[str, ...] | None,
    limit: int | None,
    validate_module: bool = False,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_sent = 0
    n_succeeded = 0
    n_failed = 0
    by_variant: dict[str, dict] = {}
    for v in ("baseline", "rag"):
        by_variant[v] = {
            "n": 0,
            "score_sum": 0.0,
            "latency_sum": 0.0,
            "ast_ok_sum": 0,
            "compile_ok_sum": 0,
            "module_ok_sum": 0,
            "module_ok_n": 0,
        }

    endpoint = f"{server.url.rstrip('/')}/generate"

    with output_path.open("w", encoding="utf-8") as out_handle:
        for row in _iter_jsonl(rendered_path):
            if splits and row.get("split") not in splits:
                continue
            if limit is not None and n_sent >= limit * 2:  # 2 rows per pair
                break

            n_sent += 1
            payload = {
                "prompt": row["prompt"],
                "max_new_tokens": server.max_new_tokens,
                "temperature": server.temperature,
                "top_p": server.top_p,
                "job_id": job_id,
                "gene_id": f"offline_rag_{row['pair_id']}_{row['variant']}",
            }

            request_started = time.time()
            error: str | None = None
            response_payload: dict | None = None
            http_attempts_used = server.max_retries
            try:
                response_payload, http_attempts_used = _post_with_retries(
                    endpoint,
                    payload,
                    timeout=server.timeout_seconds,
                    max_retries=server.max_retries,
                )
                n_succeeded += 1
            except Exception as exc:  # noqa: BLE001 - logged + recorded
                error = str(exc)
                n_failed += 1
            wallclock_latency = time.time() - request_started

            generated_text = (response_payload or {}).get("generated_text", "")
            eval_summary = _evaluate_response(
                generated_text,
                row.get("code_snippet", ""),
                validate_module=validate_module,
            )

            result = {
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "split": row["split"],
                "mutation_label": row["mutation_label"],
                "template_path": row["template_path"],
                "augment_idx": row["augment_idx"],
                "prompt_tokens_est": row.get("prompt_tokens_est"),
                "rag": row.get("rag"),
                "request": {
                    "endpoint": endpoint,
                    "max_new_tokens": server.max_new_tokens,
                    "temperature": server.temperature,
                    "top_p": server.top_p,
                    "wallclock_latency_sec": round(wallclock_latency, 4),
                    "http_attempts_used": http_attempts_used,
                    "error": error,
                },
                "response": {
                    "generated_text": generated_text,
                    "server_latency_sec": (response_payload or {}).get("_latency_sec"),
                    "batch_size": (response_payload or {}).get("batch_size"),
                    "queue_wait_time_sec": (response_payload or {}).get("queue_wait_time_sec"),
                    "evaluationScore": (response_payload or {}).get("evaluationScore"),
                    "run_hash": (response_payload or {}).get("run_hash"),
                },
                "eval": eval_summary,
            }
            out_handle.write(json.dumps(result) + "\n")
            out_handle.flush()

            variant_stats = by_variant[row["variant"]]
            variant_stats["n"] += 1
            variant_stats["score_sum"] += eval_summary["eval_score"]
            variant_stats["latency_sum"] += wallclock_latency
            if eval_summary.get("ast_parse_ok"):
                variant_stats["ast_ok_sum"] += 1
            if eval_summary.get("compile_exec_ok"):
                variant_stats["compile_ok_sum"] += 1
            mod_ok = eval_summary.get("module_validate_ok")
            if mod_ok is not None:
                variant_stats["module_ok_n"] += 1
                if mod_ok:
                    variant_stats["module_ok_sum"] += 1

    summary = {
        "rendered_path": str(rendered_path.resolve()),
        "output_path": str(output_path.resolve()),
        "endpoint": endpoint,
        "n_sent": n_sent,
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
        "by_variant": {
            variant: {
                "n": stats["n"],
                "mean_eval_score": (stats["score_sum"] / stats["n"]) if stats["n"] else None,
                "mean_wallclock_latency_sec": (stats["latency_sum"] / stats["n"]) if stats["n"] else None,
                "mean_ast_parse_ok": (stats["ast_ok_sum"] / stats["n"]) if stats["n"] else None,
                "mean_compile_exec_ok": (stats["compile_ok_sum"] / stats["n"]) if stats["n"] else None,
                "mean_module_validate_ok": (
                    (stats["module_ok_sum"] / stats["module_ok_n"]) if stats["module_ok_n"] else None
                ),
                "module_validate_rows": stats["module_ok_n"],
            }
            for variant, stats in by_variant.items()
        },
        "validate_module": validate_module,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", type=Path, required=True, help="Path to rendered JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write results JSONL.")
    parser.add_argument("--server-url", type=str, default=None, help="Override server URL.")
    parser.add_argument("--server-port", type=str, default=os.environ.get("SERVER_PORT", "8000"))
    parser.add_argument("--run-id", type=str, default=os.environ.get("RUN_ID"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.15)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=None,
        choices=["train", "test"],
        help="Restrict evaluation to specific splits (default: both).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Hard cap on the number of pairs to evaluate (each pair is 2 requests).",
    )
    parser.add_argument(
        "--module-validate",
        action="store_true",
        help="After each response, run validate_module_source on the first ```python``` block (needs torch/repo deps).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=os.environ.get("SLURM_JOB_ID", "offline_rag_eval"),
    )
    args = parser.parse_args()

    server_url = discover_server_url(
        cli_url=args.server_url, run_id=args.run_id, port=args.server_port
    )
    server = ServerConfig(
        url=server_url,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    summary = run(
        rendered_path=args.rendered,
        output_path=args.output,
        server=server,
        job_id=args.job_id,
        splits=tuple(args.splits) if args.splits else None,
        limit=args.limit,
        validate_module=args.module_validate,
    )

    print(f"Wrote results to {args.output}")
    for key in ("endpoint", "n_sent", "n_succeeded", "n_failed"):
        print(f"  {key}: {summary[key]}")
    for variant, stats in summary["by_variant"].items():
        score = stats["mean_eval_score"]
        latency = stats["mean_wallclock_latency_sec"]
        score_s = f"{score:.4f}" if score is not None else "n/a"
        latency_s = f"{latency:.2f}s" if latency is not None else "n/a"
        ast_s = f"{stats['mean_ast_parse_ok']:.4f}" if stats["mean_ast_parse_ok"] is not None else "n/a"
        comp_s = f"{stats['mean_compile_exec_ok']:.4f}" if stats["mean_compile_exec_ok"] is not None else "n/a"
        mod = stats["mean_module_validate_ok"]
        mod_s = f"{mod:.4f}" if mod is not None else "n/a"
        print(
            f"  {variant}: n={stats['n']} fence_score={score_s} ast_ok={ast_s} "
            f"compile_ok={comp_s} module_ok={mod_s} latency={latency_s}"
        )


if __name__ == "__main__":
    main()
