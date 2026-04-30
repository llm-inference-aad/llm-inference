#!/usr/bin/env python3
"""
Test client for the inference server.
Run the server first (e.g. sbatch scripts/cluster/server.sh or locally),
then run this script to verify inference works.

Usage:
  # With server on same machine (default port 8000):
  python scripts/test_inference_client.py

  # With server host/port from hostname.log (single-server mode):
  python scripts/test_inference_client.py --from-hostfile

  # Explicit host:port:
  python scripts/test_inference_client.py --host atl1-1-03-010-10-0.pace.gatech.edu --port 8000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--from-hostfile", action="store_true",
                       help="Read host from hostname.log, port from SERVER_PORT or 8000")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds)")
    args = parser.parse_args()

    if args.from_hostfile:
        repo_root = Path(__file__).resolve().parent.parent
        hostfile = repo_root / "hostname.log"
        if not hostfile.exists():
            print(f"ERROR: hostname.log not found at {hostfile}")
            print("Start the server first (e.g. sbatch scripts/cluster/server.sh)")
            sys.exit(1)
        args.host = hostfile.read_text().strip()
        args.port = int(os.environ.get("SERVER_PORT", "8000"))
        print(f"Using host:port from hostname.log: {args.host}:{args.port}")

    url = f"http://{args.host}:{args.port}"
    gen_url = f"{url}/generate"

    # 1. Health check
    print("1. Health check...")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"   WARN: GET / returned {r.status_code}")
        else:
            print(f"   OK: {r.json()}")
    except requests.exceptions.ConnectionError as e:
        print(f"   FAIL: Cannot connect to {url}")
        print(f"   {e}")
        print("   Is the server running? (Cold start: model load can take 5-10 min for Llama 70B)")
        sys.exit(1)
    except Exception as e:
        print(f"   FAIL: {e}")
        sys.exit(1)

    # 2. Inference request
    print("\n2. Inference request...")
    payload = {
        "prompt": "Write a Python function that returns the sum of two numbers.",
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
    }
    try:
        t0 = time.time()
        r = requests.post(gen_url, json=payload, timeout=args.timeout)
        elapsed = time.time() - t0
        if r.status_code != 200:
            print(f"   FAIL: POST /generate returned {r.status_code}")
            print(r.text[:500])
            sys.exit(1)
        data = r.json()
        text = data.get("generated_text", "")[:400]
        latency = data.get("e2e_latency_sec", data.get("_latency_sec", "?"))
        score = data.get("evaluationScore", "?")
        print(f"   OK (client elapsed: {elapsed:.1f}s, server latency: {latency}s)")
        print(f"   Evaluation score: {score}")
        print(f"   Generated text preview:\n   {text}...")
        print("\n✅ Inference server verified successfully.")
    except requests.exceptions.Timeout:
        print(f"   FAIL: Request timed out after {args.timeout}s")
        print("   Cold start: large models (e.g. Llama 70B) can take 5-10 min to load.")
        sys.exit(1)
    except Exception as e:
        print(f"   FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
