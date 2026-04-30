#!/usr/bin/env python3
"""Benchmark prefix-cache-aware routing through the gateway.

This intentionally uses the existing /generate API so it exercises the same
path as LLMGE. With the current worker API, ttft_sec is a first-token proxy
when max_new_tokens=1 because the vLLM offline engine returns a completed
generation rather than streaming token deltas.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def metrics_dir(root: Path) -> Path:
    run_id = os.environ.get("RUN_ID", "server-only")
    if run_id == "server-only":
        path = root / "metrics" / "data"
    else:
        path = root / "runs" / run_id / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def gateway_url(root: Path, host: str | None, port: int | None) -> str:
    if host is None:
        host_file = Path(os.environ.get("LOADBALANCER_LOG_FILE", root / "loadbalancer.log"))
        if not host_file.exists():
            raise FileNotFoundError(f"Load balancer host file not found: {host_file}")
        host = host_file.read_text().strip()
    if port is None:
        port = int(os.environ.get("LOAD_BALANCER_PORT", "9000"))
    return f"http://{host}:{port}"


def wait_for_gateway(url: str, timeout_sec: int, interval_sec: int = 5) -> dict:
    """Wait until the gateway process is accepting HTTP requests."""
    deadline = time.time() + timeout_sec
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/", timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            print(f"Waiting for gateway: {exc}")
            time.sleep(interval_sec)
    raise TimeoutError(f"Gateway did not become ready after {timeout_sec}s. Last error: {last_error}")


def llmge_like_prompt(shared_prefix: str, suffix: str) -> str:
    return f"""{shared_prefix}

Task:
Write a PyTorch model definition for CIFAR-10. Return only runnable Python code.
The model should be compact, trainable, and compatible with the LLMGE evaluator.

Variation:
{suffix}
"""


def make_shared_prefix(label: str) -> str:
    block = f"""
You are assisting an LLM-guided evolutionary search system.
The generated individual must define a neural network class that can be imported
by an evaluation job. Keep interfaces stable, avoid external downloads, and
prefer simple convolutional building blocks.
Shared cache prefix label: {label}.
"""
    return "\n".join([block.strip()] * 36)


def post_generate(url: str, prompt: str, label: str, timeout: int) -> dict:
    payload = {
        "prompt": prompt,
        "max_new_tokens": 1,
        "temperature": 0.1,
        "top_p": 0.15,
        "job_id": f"cache-benchmark-{label}",
        "gene_id": label,
    }
    started = time.time()
    response = requests.post(f"{url}/generate", json=payload, timeout=timeout)
    elapsed = time.time() - started
    response.raise_for_status()
    data = response.json()
    return {
        "label": label,
        "client_elapsed_sec": round(elapsed, 4),
        "ttft_sec": data.get("ttft_sec"),
        "ttft_measurement": data.get("ttft_measurement"),
        "server_latency_sec": data.get("e2e_latency_sec", data.get("_latency_sec")),
        "gateway_cache_hit": data.get("gateway_cache_hit"),
        "gateway_prefix_hash": data.get("gateway_prefix_hash"),
        "gateway_selected_server": data.get("gateway_selected_server"),
        "gateway_response_time_sec": data.get("gateway_response_time_sec"),
        "worker_hostname": data.get("worker_hostname"),
        "backend": data.get("backend"),
        "vllm_prefix_caching": data.get("vllm_prefix_caching"),
    }


def wait_for_healthy_workers(url: str, timeout_sec: int, interval_sec: int = 30) -> dict:
    """Wait until the gateway reports at least one healthy worker."""
    deadline = time.time() + timeout_sec
    last_status: dict = {}
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/servers", timeout=15)
            response.raise_for_status()
            status = response.json()
            last_status = status
            if status.get("healthy_servers", 0) > 0:
                return status
            print(f"Waiting for healthy workers: {status}")
        except Exception as exc:
            print(f"Waiting for gateway/workers: {exc}")
        time.sleep(interval_sec)
    raise TimeoutError(f"No healthy workers after {timeout_sec}s. Last status: {last_status}")


def summarize(results: list[dict]) -> dict:
    hit_ttfts = [r["ttft_sec"] for r in results if r.get("gateway_cache_hit") and r.get("ttft_sec") is not None]
    miss_ttfts = [r["ttft_sec"] for r in results if not r.get("gateway_cache_hit") and r.get("ttft_sec") is not None]
    summary = {
        "total_requests": len(results),
        "cache_hits": sum(1 for r in results if r.get("gateway_cache_hit")),
        "cache_misses": sum(1 for r in results if not r.get("gateway_cache_hit")),
        "hit_ttft_mean_sec": statistics.mean(hit_ttfts) if hit_ttfts else None,
        "miss_ttft_mean_sec": statistics.mean(miss_ttfts) if miss_ttfts else None,
    }
    if summary["hit_ttft_mean_sec"] is not None and summary["miss_ttft_mean_sec"]:
        summary["ttft_improvement_ratio"] = summary["miss_ttft_mean_sec"] / summary["hit_ttft_mean_sec"]
    else:
        summary["ttft_improvement_ratio"] = None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Gateway host; defaults to loadbalancer.log")
    parser.add_argument("--port", type=int, help="Gateway port; defaults to LOAD_BALANCER_PORT/9000")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--repeats", type=int, default=2, help="Warm hit repetitions after the cold request")
    parser.add_argument("--wait-healthy", type=int, default=3600, help="Seconds to wait for at least one healthy worker")
    args = parser.parse_args()

    root = repo_root()
    load_env_file(root / ".env")
    url = gateway_url(root, args.host, args.port)

    print(f"Gateway: {url}")
    gateway_health = wait_for_gateway(url, min(args.wait_healthy, 300))
    print(f"Gateway health: {gateway_health}")
    worker_status = wait_for_healthy_workers(url, args.wait_healthy)
    print(f"Worker status: {worker_status}")

    shared_a = make_shared_prefix("A")
    shared_b = make_shared_prefix("B")
    prompts = [
        ("cold_miss_a", llmge_like_prompt(shared_a, "Use residual blocks and group normalization.")),
    ]
    prompts.extend(
        (f"cache_hit_a_{idx + 1}", llmge_like_prompt(shared_a, f"Use residual blocks and group normalization. Repeat {idx}."))
        for idx in range(args.repeats)
    )
    prompts.append(("cold_miss_b", llmge_like_prompt(shared_b, "Use depthwise separable convolutions.")))

    results = []
    for label, prompt in prompts:
        result = post_generate(url, prompt, label, args.timeout)
        results.append(result)
        print(json.dumps(result, indent=2))

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gateway_url": url,
        "summary": summarize(results),
        "results": results,
    }
    out_file = metrics_dir(root) / f"cache-affinity-benchmark-{int(time.time())}.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"Wrote benchmark metrics to {out_file}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
