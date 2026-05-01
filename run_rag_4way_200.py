#!/usr/bin/env python3
"""Run a 4-way RAG benchmark suite.

Conditions:
1. RAG only
2. RAG + speculative
3. RAG + constrained
4. RAG + constrained + speculative

Default: 200 requests per condition (800 total).
This assumes the server already has RAG enabled in its environment.
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", 8001)))
parser.add_argument("--server-host", type=str, default=os.getenv("SERVER_HOST", ""))
parser.add_argument("--server-registry-file", type=Path, default=Path(os.getenv("SERVER_REGISTRY_FILE", "")) if os.getenv("SERVER_REGISTRY_FILE") else None)
parser.add_argument("--output-dir", type=Path, default=Path("runs/rag_4way_200/metrics"))
parser.add_argument("--num-requests-per-config", type=int, default=200)
parser.add_argument("--config-name", type=str, default="", help="Run only one config by name (e.g. 1_rag_only)")
args = parser.parse_args()

RESULTS_DIR = args.output_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "name": "1_rag_only",
        "prompt": "Generate a short response about artificial intelligence using the retrieved context.",
        "constraint_type": None,
        "speculative": False,
    },
    {
        "name": "2_rag_plus_spec",
        "prompt": "Generate a short response about artificial intelligence using the retrieved context.",
        "constraint_type": None,
        "speculative": True,
        "speculative_method": "suffix",
        "num_speculative_tokens": 5,
    },
    {
        "name": "3_rag_plus_constrained",
        "prompt": "Generate a JSON response summarizing the retrieved context.",
        "constraint_type": "json",
        "json_schema": {
            "type": "object",
            "required": ["topic", "summary", "score"],
            "properties": {
                "topic": {"type": "string"},
                "summary": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "speculative": False,
    },
    {
        "name": "4_rag_plus_both",
        "prompt": "Generate a JSON response summarizing the retrieved context.",
        "constraint_type": "json",
        "json_schema": {
            "type": "object",
            "required": ["topic", "summary", "score"],
            "properties": {
                "topic": {"type": "string"},
                "summary": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "speculative": True,
        "speculative_method": "suffix",
        "num_speculative_tokens": 5,
    },
]


def build_request(config: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "prompt": config["prompt"],
        "input": config["prompt"],
    }
    if config.get("constraint_type"):
        req["constraint_type"] = config["constraint_type"]
        if config["constraint_type"] == "json":
            req["json_schema"] = config.get("json_schema")
    if config.get("speculative"):
        req["enable_speculative"] = True
        req["speculative_method"] = config.get("speculative_method", "suffix")
        req["num_speculative_tokens"] = config.get("num_speculative_tokens", 5)
    return req


def resolve_server_url() -> str:
    if args.server_host:
        return f"http://{args.server_host}:{args.port}/generate"

    candidate_registries: list[Path] = []
    if args.server_registry_file is not None:
        candidate_registries.append(args.server_registry_file)

    # Search for the newest registry written by an active server run.
    for path in sorted(Path("runs").glob("*/logs/servers.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if path not in candidate_registries:
            candidate_registries.append(path)

    for _ in range(120):  # wait up to ~20 minutes in 10s intervals
        for registry in candidate_registries:
            if registry is None or not registry.exists():
                continue
            try:
                data = json.loads(registry.read_text())
                servers = data.get("servers", [])
                for server in reversed(servers):
                    if int(server.get("port", -1)) == args.port:
                        host = server.get("hostname") or server.get("host")
                        if host:
                            return f"http://{host}:{args.port}/generate"
                if servers:
                    server = servers[-1]
                    host = server.get("hostname") or server.get("host")
                    port = int(server.get("port", args.port))
                    if host:
                        return f"http://{host}:{port}/generate"
            except Exception:
                pass
        time.sleep(10)
    return f"http://127.0.0.1:{args.port}/generate"


def post_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(SERVER_URL, json=payload, timeout=360)
    resp.raise_for_status()
    return resp.json()


def run():
    global SERVER_URL
    SERVER_URL = resolve_server_url()
    configs = [cfg for cfg in CONFIGS if not args.config_name or cfg["name"] == args.config_name]
    if args.config_name and not configs:
        raise SystemExit(f"Unknown config name: {args.config_name}")

    total_reqs = args.num_requests_per_config * len(configs)
    print("=" * 80)
    print("RAG 4-Way Benchmark")
    print("=" * 80)
    print(f"Server: {SERVER_URL}")
    print(f"Requests per config: {args.num_requests_per_config}")
    print(f"Configs: {[c['name'] for c in configs]}")
    print(f"Total requests: {total_reqs}")
    print()

    all_results = []
    per_config: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in configs}
    start = time.time()

    for cfg in configs:
        print(f"[CONFIG] {cfg['name']}")
        for i in range(args.num_requests_per_config):
            t0 = time.time()
            try:
                result = post_request(build_request(cfg))
                result["config_name"] = cfg["name"]
                result["request_idx"] = i
                result["wall_time_sec"] = time.time() - t0
                all_results.append(result)
                per_config[cfg["name"]].append(result)
                if (i + 1) % 25 == 0:
                    print(f"  [{i+1}/{args.num_requests_per_config}] ok | {result.get('response_time_sec', '?')}s")
            except Exception as e:
                print(f"  [{i+1}/{args.num_requests_per_config}] failed: {e}")
        print()

    elapsed = time.time() - start

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_requests": total_reqs,
        "successful": len(all_results),
        "failed": total_reqs - len(all_results),
        "total_wall_time_sec": elapsed,
        "throughput_req_per_sec": len(all_results) / elapsed if elapsed > 0 else 0,
        "per_config": {},
    }

    for name, results in per_config.items():
        latencies = [r.get("response_time_sec", 0) for r in results if "response_time_sec" in r]
        specs = [r.get("vllm_num_speculative_tokens", 0) for r in results if "vllm_num_speculative_tokens" in r]
        scores = [r.get("evaluationScore", 0) for r in results if "evaluationScore" in r]
        if latencies:
            summary["per_config"][name] = {
                "count": len(results),
                "latency_mean": statistics.mean(latencies),
                "latency_median": statistics.median(latencies),
                "latency_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "latency_min": min(latencies),
                "latency_max": max(latencies),
                "spec_tokens_mean": statistics.mean(specs) if specs else 0,
                "score_mean": statistics.mean(scores) if scores else 0,
            }

    (RESULTS_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    with open(RESULTS_DIR / "all_requests.jsonl", "w") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

    print("=" * 80)
    print("Completed.")
    print(f"Summary: {RESULTS_DIR / 'benchmark_summary.json'}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    run()
