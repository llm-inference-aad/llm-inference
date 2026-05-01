#!/usr/bin/env python3
"""Quick test: compare constrained decoding with and without RAG context.

Behavior:
- If the server at localhost is running, send two requests (constrained JSON and unconstrained)
  with and without RAG context and print timing + outputs.
- If the server is not running, read `rag_data/metadata` and print a sample of the
  stored documents to show what would be injected.
"""
import time
import requests
import os
import json

from pathlib import Path

SERVER = os.environ.get("SERVER_URL", "http://localhost:8001")


def server_running():
    try:
        r = requests.get(SERVER + "/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def read_local_rag_sample(n=3):
    base = Path("rag_data/metadata")
    out = {"code": [], "text": []}
    if not base.exists():
        print("No local rag_data/metadata found")
        return out

    codef = base / "code.jsonl"
    textf = base / "text.jsonl"
    if codef.exists():
        with open(codef, "r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                out["code"].append(json.loads(line))
    if textf.exists():
        with open(textf, "r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                out["text"].append(json.loads(line))
    return out


def run_requests():
    from constrained_decoding_demo import ConstrainedLLMClient

    client = ConstrainedLLMClient(server_url=SERVER)

    prompt = "Generate a JSON object with fields: name, age, email"
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        },
        "required": ["name", "age", "email"]
    }

    print("\n== Running requests against server at", SERVER, "==\n")

    # Unconstrained without RAG
    t0 = time.time()
    u_no = client.generate_unconstrained(prompt, max_new_tokens=80, use_rag=False)
    t1 = time.time()

    # Unconstrained with RAG
    t2 = time.time()
    u_rag = client.generate_unconstrained(prompt, max_new_tokens=80, use_rag=True)
    t3 = time.time()

    # Constrained with JSON schema + RAG
    t4 = time.time()
    c_rag = client.generate_with_json_schema(prompt, schema, max_new_tokens=120, use_rag=True)
    t5 = time.time()

    print("Unconstrained no-RAG (%.2fs):" % (t1 - t0))
    print(u_no)
    print("\nUnconstrained with RAG (%.2fs):" % (t3 - t2))
    print(u_rag)
    print("\nConstrained (JSON) with RAG (%.2fs):" % (t5 - t4))
    print(c_rag)


def main():
    if server_running():
        try:
            run_requests()
            return
        except Exception as e:
            print("Error running requests:", e)

    print("Server not reachable — showing local RAG data sample instead.")
    sample = read_local_rag_sample(n=3)
    print(json.dumps({"sample_count_code": len(sample["code"]), "sample_count_text": len(sample["text"])}, indent=2))
    if sample["code"]:
        print("\n--- Example code document ---")
        print(sample["code"][0]["content"][:1000])
    if sample["text"]:
        print("\n--- Example text document ---")
        print(sample["text"][0]["content"][:1000])


if __name__ == "__main__":
    main()
