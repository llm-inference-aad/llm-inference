#!/usr/bin/env python3
"""Submit a single example request to the local LLM FastAPI server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test one /generate call against the LLM server.")
    parser.add_argument(
        "--prompt",
        default="Write a Python function named add(a, b) that returns their sum.",
        help="Prompt text to send to the server.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--job-id", default=os.getenv("SLURM_JOB_ID", "llm-call-test"))
    parser.add_argument("--gene-id", default="example_gene")
    parser.add_argument("--host", default=None, help="Server host override.")
    parser.add_argument("--port", type=int, default=None, help="Server port override.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=10, help="Retry attempts for warmup/connectivity.")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Seconds to sleep between retries.")
    return parser.parse_args()


def _read_host_from_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Host file not found: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Host file is empty: {path}")
    return value


def resolve_endpoint(args: argparse.Namespace) -> tuple[str, int]:
    root_dir = Path(os.getenv("LLM_INFERENCE_ROOT_DIR", Path.cwd()))
    use_load_balancer = os.getenv("USE_LOAD_BALANCER", "false").lower() in {"1", "true", "yes"}

    if args.host:
        host = args.host
    elif use_load_balancer:
        lb_file = Path(os.getenv("LOADBALANCER_LOG_FILE", root_dir / "loadbalancer.log"))
        host = _read_host_from_file(lb_file)
    else:
        hostname_file = Path(os.getenv("HOSTNAME_LOG_FILE", root_dir / "hostname.log"))
        host = _read_host_from_file(hostname_file)

    if args.port is not None:
        port = args.port
    elif use_load_balancer:
        port = int(os.getenv("LOAD_BALANCER_PORT", "9000"))
    else:
        port = int(os.getenv("SERVER_PORT", "8000"))

    return host, port


def post_generate(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def main() -> int:
    args = parse_args()
    host, port = resolve_endpoint(args)
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "job_id": args.job_id,
        "gene_id": args.gene_id,
    }

    print(f"Target endpoint: {url}")
    print(f"Retries: {args.retries}, timeout: {args.timeout}s")

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            started = time.time()
            result = post_generate(url, payload, args.timeout)
            elapsed = time.time() - started
            text = str(result.get("generated_text", ""))
            print(f"Request succeeded on attempt {attempt} in {elapsed:.2f}s")
            print(f"evaluationScore={result.get('evaluationScore')} run_hash={result.get('run_hash')}")
            print("--- generated_text ---")
            print(text.strip() if text else "<empty>")
            return 0
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            print(f"Attempt {attempt}/{args.retries} failed: {exc}")
            if attempt < args.retries:
                time.sleep(args.retry_sleep)

    print(f"Request failed after {args.retries} attempts: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
