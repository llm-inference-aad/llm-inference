#!/usr/bin/env python
"""
Run 5-configuration test suite against mock or real inference server.
Tests: baseline, constrained_json, constrained_regex, speculative_suffix, combined.
"""
import json
import requests
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", 8002)))
parser.add_argument("--output-dir", type=Path, default=Path("runs/server-only/metrics/smoke_tests"))
args = parser.parse_args()

SERVER_PORT = args.port
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}/generate"
RESULTS_DIR = args.output_dir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Test configurations
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
        "speculative_method": "draft_model",
        "draft_model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 5,
    },
]


def build_request(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build LLM request from config"""
    # Include both keys for compatibility:
    # - mock server expects `input`
    # - real server expects `prompt`
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


def run_tests():
    """Run all configuration tests"""
    print("=" * 60)
    print("5-Configuration Test Suite - Mock Server")
    print("=" * 60)
    print()
    
    results_summary = []
    
    for config in CONFIGS:
        config_name = config["name"]
        print(f"[TEST] {config_name}")
        print(f"  Input: {config['input'][:50]}...")
        
        # Build request
        request_payload = build_request(config)
        print(f"  Constraint: {config.get('constraint_type', 'none')}")
        print(f"  Speculative: {config.get('speculative', False)}")
        
        # Make request
        try:
            response = requests.post(
                SERVER_URL,
                json=request_payload,
                timeout=180,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Save result
            result_file = RESULTS_DIR / f"config_{config_name}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            
            # Display summary
            tokens = result.get("tokens_total", result.get("num_speculative_tokens", "?"))
            latency = result.get("response_time_sec", result.get("e2e_latency_sec", "?"))
            print(f"  ✅ Success | Latency: {latency}s | Tokens: {tokens}")
            results_summary.append((config_name, True, latency))
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed: {str(e)}")
            
            # Save empty result
            result_file = RESULTS_DIR / f"config_{config_name}.json"
            with open(result_file, "w") as f:
                json.dump({"error": str(e)}, f)
            
            results_summary.append((config_name, False, None))
        
        print()
    
    # Summary
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    passed = sum(1 for _, success, _ in results_summary if success)
    print(f"Passed: {passed}/{len(CONFIGS)}")
    print()
    
    for name, success, latency in results_summary:
        status = "✅" if success else "❌"
        latency_str = f"{latency:.4f}s" if latency else "N/A"
        print(f"  {status} {name}: {latency_str}")
    
    print()
    print(f"Results saved to: {RESULTS_DIR}")
    print()
    
    return passed == len(CONFIGS)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
