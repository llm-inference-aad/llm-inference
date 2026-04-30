#!/usr/bin/env python
"""
Run 500-request benchmark against vLLM server with 5-config distribution.
Produces detailed latency, acceptance, and throughput metrics.
"""
import json
import requests
import sys
import os
import argparse
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", 8002)))
parser.add_argument("--output-dir", type=Path, default=Path("runs/vllm_500request/metrics"))
parser.add_argument("--num-requests", type=int, default=500)
args = parser.parse_args()

SERVER_PORT = args.port
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}/generate"
RESULTS_DIR = args.output_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Test configurations (will be cycled across 500 requests)
CONFIGS = [
    {
        "name": "1_baseline",
        "input": "Generate a short response about artificial intelligence",
        "constraint_type": None,
        "speculative": False,
    },
    {
        "name": "2_constrained_json",
        "input": "Generate a JSON response about AI",
        "constraint_type": "json",
        "json_schema": {
            "type": "object",
            "required": ["topic", "sentiment", "score"],
            "properties": {
                "topic": {"type": "string"},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "score": {"type": "number"},
            },
        },
        "speculative": False,
    },
    {
        "name": "3_constrained_regex",
        "input": "Generate an email address",
        "constraint_type": "regex",
        "constraint": r"^[a-z]+\.[a-z]+@[a-z]+\.[a-z]+$",
        "speculative": False,
    },
    {
        "name": "4_speculative_suffix",
        "input": "Complete the sentence: The future of AI is",
        "constraint_type": None,
        "speculative": True,
        "speculative_method": "suffix",
        "num_speculative_tokens": 5,
    },
    {
        "name": "5_combined_json_spec",
        "input": "Generate a JSON response with AI insights",
        "constraint_type": "json",
        "json_schema": {
            "type": "object",
            "required": ["insight", "confidence"],
            "properties": {
                "insight": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "speculative": True,
        "speculative_method": "suffix",
        "num_speculative_tokens": 5,
    },
]


def build_request(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build LLM request from config"""
    request = {
        "input": config["input"],
        "prompt": config["input"],
    }
    
    if config.get("constraint_type"):
        request["constraint_type"] = config["constraint_type"]
        if config["constraint_type"] == "json":
            request["json_schema"] = config.get("json_schema")
        elif config["constraint_type"] == "regex":
            request["constraint"] = config.get("constraint")
    
    if config.get("speculative"):
        request["enable_speculative"] = True
        request["speculative_method"] = config.get("speculative_method", "suffix")
        request["num_speculative_tokens"] = config.get("num_speculative_tokens", 5)
        if config.get("draft_model"):
            request["draft_model"] = config["draft_model"]
    
    return request


def run_benchmark():
    """Run 500-request benchmark with config cycling"""
    print("=" * 80)
    print("500-Request Benchmark Suite - vLLM Server")
    print("=" * 80)
    print(f"Target: {args.num_requests} requests")
    print(f"Configs: cycling through {len(CONFIGS)} configurations")
    print(f"Server: http://127.0.0.1:{SERVER_PORT}/generate")
    print()
    
    all_results = []
    per_config_results = {cfg["name"]: [] for cfg in CONFIGS}
    failures = 0
    start_time_global = time.time()
    
    for i in range(args.num_requests):
        # Cycle through configs
        config = CONFIGS[i % len(CONFIGS)]
        config_name = config["name"]
        
        # Build and send request
        request_payload = build_request(config)
        request_start = time.time()
        
        try:
            response = requests.post(
                SERVER_URL,
                json=request_payload,
                timeout=180,
            )
            response.raise_for_status()
            request_end = time.time()
            
            result = response.json()
            result["request_id"] = i
            result["config_name"] = config_name
            result["wall_time_sec"] = request_end - request_start
            
            all_results.append(result)
            per_config_results[config_name].append(result)
            
            # Progress indicator every 50 requests
            if (i + 1) % 50 == 0:
                latency = result.get("response_time_sec", "?")
                tokens = result.get("completion_tokens", "?")
                print(f"[{i+1}/{args.num_requests}] {config_name:25s} | Latency: {latency:6}s | Tokens: {tokens:4}")
                
        except requests.exceptions.RequestException as e:
            print(f"[{i+1}/{args.num_requests}] {config_name:25s} | ❌ FAILED: {e}")
            failures += 1
    
    end_time_global = time.time()
    total_wall_time = end_time_global - start_time_global
    
    # Aggregate and save metrics
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Per-config stats
    per_config_stats = {}
    for config_name in per_config_results:
        results = per_config_results[config_name]
        if results:
            latencies = [r.get("response_time_sec", 0) for r in results if "response_time_sec" in r]
            spec_tokens = [r.get("vllm_num_speculative_tokens", 0) for r in results if "vllm_num_speculative_tokens" in r]
            scores = [r.get("evaluationScore", 0) for r in results if "evaluationScore" in r]
            
            per_config_stats[config_name] = {
                "count": len(results),
                "latency_mean": statistics.mean(latencies) if latencies else 0,
                "latency_median": statistics.median(latencies) if latencies else 0,
                "latency_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "latency_min": min(latencies) if latencies else 0,
                "latency_max": max(latencies) if latencies else 0,
                "spec_tokens_mean": statistics.mean(spec_tokens) if spec_tokens else 0,
                "score_mean": statistics.mean(scores) if scores else 0,
            }
    
    # Print per-config table
    print("\nPer-Config Statistics (all times in seconds):")
    print("-" * 100)
    print(f"{'Config':<25} {'Count':<8} {'Mean':<10} {'Median':<10} {'StDev':<10} {'Min':<10} {'Max':<10} {'Spec Tokens':<12}")
    print("-" * 100)
    for config_name, stats in per_config_stats.items():
        print(f"{config_name:<25} {stats['count']:<8} {stats['latency_mean']:<10.2f} {stats['latency_median']:<10.2f} {stats['latency_stdev']:<10.2f} {stats['latency_min']:<10.2f} {stats['latency_max']:<10.2f} {stats['spec_tokens_mean']:<12.1f}")
    
    # Overall stats
    all_latencies = [r.get("response_time_sec", 0) for r in all_results if "response_time_sec" in r]
    all_scores = [r.get("evaluationScore", 0) for r in all_results if "evaluationScore" in r]
    all_spec_tokens = [r.get("vllm_num_speculative_tokens", 0) for r in all_results if "vllm_num_speculative_tokens" in r]
    
    print()
    print("Overall Statistics:")
    print("-" * 100)
    print(f"Total Requests:         {args.num_requests}")
    print(f"Successful:             {len(all_results)}")
    print(f"Failed:                 {failures}")
    print(f"Success Rate:           {100 * len(all_results) / args.num_requests:.1f}%")
    print()
    print(f"Total Wall Time:        {total_wall_time:.2f}s")
    print(f"Throughput:             {len(all_results) / total_wall_time:.2f} req/s")
    print()
    print(f"Latency Mean:           {statistics.mean(all_latencies):.2f}s" if all_latencies else "N/A")
    print(f"Latency Median:         {statistics.median(all_latencies):.2f}s" if all_latencies else "N/A")
    print(f"Latency StDev:          {statistics.stdev(all_latencies):.2f}s" if len(all_latencies) > 1 else "N/A")
    print(f"Latency p95:            {sorted(all_latencies)[int(0.95 * len(all_latencies))]:.2f}s" if all_latencies else "N/A")
    print(f"Latency p99:            {sorted(all_latencies)[int(0.99 * len(all_latencies))]:.2f}s" if all_latencies else "N/A")
    print()
    print(f"Eval Score Mean:        {statistics.mean(all_scores):.2f}" if all_scores else "N/A")
    print(f"Speculative Tokens:     {statistics.mean(all_spec_tokens):.1f}" if all_spec_tokens else "N/A")
    
    # Save full results JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_requests": args.num_requests,
        "successful": len(all_results),
        "failed": failures,
        "total_wall_time_sec": total_wall_time,
        "throughput_req_per_sec": len(all_results) / total_wall_time if total_wall_time > 0 else 0,
        "per_config": per_config_stats,
        "overall": {
            "latency_mean": statistics.mean(all_latencies) if all_latencies else 0,
            "latency_median": statistics.median(all_latencies) if all_latencies else 0,
            "latency_stdev": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
            "latency_p95": sorted(all_latencies)[int(0.95 * len(all_latencies))] if all_latencies else 0,
            "latency_p99": sorted(all_latencies)[int(0.99 * len(all_latencies))] if all_latencies else 0,
            "eval_score_mean": statistics.mean(all_scores) if all_scores else 0,
            "spec_tokens_mean": statistics.mean(all_spec_tokens) if all_spec_tokens else 0,
        }
    }
    
    summary_file = RESULTS_DIR / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Summary saved to: {summary_file}")
    
    # Save all results for post-processing
    results_file = RESULTS_DIR / "all_requests.jsonl"
    with open(results_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    print(f"All results saved to: {results_file}")


if __name__ == "__main__":
    run_benchmark()
