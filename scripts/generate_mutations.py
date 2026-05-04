"""
Generate synthetic code mutations via the local Llama server and submit SLURM eval jobs.

For each (code block, prompt template) combination, calls the local vLLM server to
produce a mutated version of the block, saves it as a network file, and submits a
SLURM eval job.  A manifest (synthetic_manifest.json) is written to the run dir so
join_metrics.py can reconstruct dataset.json entries without needing slurm logs.

Usage:
    uv run python scripts/generate_mutations.py \\
        --run-id my_run_20260428_011918 \\
        [--max-mutations 200] \\
        [--temperatures 0.3 0.7 1.0] \\
        [--no-submit]
"""

import argparse
import glob
import json
import os
import random
import re
import string
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_NETWORK = os.path.join(REPO_ROOT, "sota/ExquisiteNetV2/network.py")
MODELS_DIR = os.path.join(REPO_ROOT, "sota/ExquisiteNetV2/models")
TEMPLATES_DIR = os.path.join(REPO_ROOT, "templates/FixedPrompts")
CONSTANT_RULES = os.path.join(REPO_ROOT, "templates/ConstantRules.txt")
SLURM_RESULTS_DIR = os.path.join(REPO_ROOT, "slurm-results")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env():
    env_path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(env_path):
        sys.exit("ERROR: .env not found")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def new_gene_id():
    chars = string.ascii_letters + string.digits
    return "xXx" + "".join(random.choices(chars, k=24))


def load_code_blocks():
    with open(SEED_NETWORK) as f:
        content = f.read()
    parts = re.split(r"# --OPTION--", content)
    blocks = []
    block_to_part_idx = {}  # maps block_idx -> index in parts[]
    for i, part in enumerate(parts[1:], start=1):
        stripped = part.strip()
        if stripped and not stripped.startswith("# -- NOTE --"):
            block_to_part_idx[len(blocks)] = i
            blocks.append(stripped)
    return blocks, parts, block_to_part_idx


def load_templates():
    rules = open(CONSTANT_RULES).read().strip()
    templates = []
    for path in sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*/*.txt"))):
        tmpl = open(path).read().strip()
        full = f"{tmpl}\n{rules}"
        templates.append((os.path.basename(path), full))
    return templates


def extract_code(response_text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response_text, re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    if "class " in response_text or "def " in response_text:
        return response_text.strip()
    return ""


def validate_code(code, original_block):
    if not code or len(code) < 20:
        return False
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return False
    name_match = re.search(r"^(class|def)\s+(\w+)", original_block, re.MULTILINE)
    if name_match and name_match.group(2) not in code:
        return False
    return True


def get_server_url():
    """Read hostname.log (or load balancer log) and return the /generate URL."""
    use_lb = os.getenv("USE_LOAD_BALANCER", "false").lower() in ("true", "1", "yes")
    if use_lb:
        lb_file = os.getenv("LOADBALANCER_LOG_FILE", os.path.join(REPO_ROOT, "loadbalancer.log"))
        hostname_file = lb_file
        port = os.getenv("LOAD_BALANCER_PORT", "9000")
    else:
        hostname_file = os.getenv("HOSTNAME_LOG_FILE", os.path.join(REPO_ROOT, "hostname.log"))
        port = os.getenv("SERVER_PORT", "8000")

    if not os.path.exists(hostname_file):
        sys.exit(
            f"ERROR: server hostname file not found: {hostname_file}\n"
            "Make sure the LLM server job is running (sbatch server.sh)."
        )
    hostname = open(hostname_file).read().strip()
    return f"http://{hostname}:{port}/generate"


def call_local_server(prompt, temperature, server_url):
    payload = {
        "prompt": prompt,
        "max_new_tokens": int(os.getenv("LOCAL_SERVER_MAX_NEW_TOKENS", "1024")),
        "top_p": 0.8,
        "temperature": temperature,
        "job_id": os.getenv("SLURM_JOB_ID", "gen_mutations"),
    }
    timeout = float(os.getenv("LOCAL_SERVER_TIMEOUT", "3600"))
    max_retries = int(os.getenv("LOCAL_SERVER_MAX_RETRIES", "3"))
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(server_url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json().get("generated_text", "")
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.Timeout:
            # Server is overloaded — retrying immediately won't help, skip this combo
            print(f"  [attempt {attempt}/{max_retries}] timed out after {timeout}s — skipping")
            return None
        except requests.ConnectionError as e:
            print(f"  [attempt {attempt}/{max_retries}] connection error: {e}")
        except Exception as e:
            print(f"  [attempt {attempt}/{max_retries}] error: {e}")
            break
        if attempt < max_retries:
            time.sleep(5 * attempt)
    return None


def build_eval_script(gid, run_dir, venv_path, llm_inference_root):
    return f"""#!/bin/bash
#SBATCH --job-name=evalGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output={SLURM_RESULTS_DIR}/eval-{gid}-%j.out
#SBATCH --error={SLURM_RESULTS_DIR}/eval-{gid}-%j.err

echo "Evaluating gene: {gid}"
hostname
module load cuda

export LD_LIBRARY_PATH="{venv_path}/lib/python3.13/site-packages/nvidia/nvjitlink/lib:${{LD_LIBRARY_PATH:-}}"
source "{venv_path}/bin/activate"

export RUN_DIR="{run_dir}"
cd "{llm_inference_root}"

python ./sota/ExquisiteNetV2/train.py \\
  -bs 216 \\
  -network "models.network_{gid}" \\
  -data ./cifar10 \\
  -end_lr 0.001 \\
  -seed 21 \\
  -val_r 0.2 \\
  -amp
"""


def submit_job(script_content):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        tmp = f.name
    result = subprocess.run(["sbatch", tmp], capture_output=True, text=True)
    os.unlink(tmp)
    if result.returncode != 0:
        print(f"  sbatch error: {result.stderr.strip()}")
        return None
    m = re.search(r"(\d+)", result.stdout)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="my_run_20260428_011918")
    parser.add_argument("--max-mutations", type=int, default=200)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    load_env()

    venv_path = os.environ.get("VENV_PATH", "")
    llm_root = os.environ.get("LLM_INFERENCE_ROOT_DIR", REPO_ROOT)
    if not venv_path:
        sys.exit("ERROR: VENV_PATH not set in .env")

    server_url = get_server_url()
    print(f"Using server: {server_url}")

    # Quick connectivity check
    try:
        requests.get(server_url.replace("/generate", "/health"), timeout=5)
        print("Server health check: OK")
    except Exception:
        print("(health endpoint not available — will proceed anyway)")

    run_dir = os.path.join(REPO_ROOT, "runs", args.run_id)
    if not os.path.isdir(run_dir):
        sys.exit(f"ERROR: run dir not found: {run_dir}")

    manifest_path = os.path.join(run_dir, "synthetic_manifest.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SLURM_RESULTS_DIR, exist_ok=True)

    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    print(f"Loaded manifest: {len(manifest)} existing entries")

    code_blocks, seed_parts, block_to_part_idx = load_code_blocks()
    templates = load_templates()
    print(f"Code blocks: {len(code_blocks)}, Templates: {len(templates)}, Temperatures: {args.temperatures}")

    work = [
        (bi, ti, t)
        for bi in range(len(code_blocks))
        for ti in range(len(templates))
        for t in args.temperatures
    ]
    random.shuffle(work)

    generated = 0
    submitted = 0

    print(f"Target: {args.max_mutations} new mutations")

    for block_idx, tmpl_idx, temperature in work:
        if generated >= args.max_mutations:
            break

        block = code_blocks[block_idx]
        tmpl_name, tmpl_text = templates[tmpl_idx]
        prompt = tmpl_text.replace("{}", block, 1)

        combo_key = f"{block_idx}:{tmpl_name}:{temperature}"
        if any(e.get("combo_key") == combo_key for e in manifest.values()):
            continue

        gid = new_gene_id()
        model_path = os.path.join(MODELS_DIR, f"network_{gid}.py")

        print(f"\n[{generated+1}/{args.max_mutations}] block={block_idx} tmpl={tmpl_name} temp={temperature}")
        print(f"  gene={gid}")

        response_text = call_local_server(prompt, temperature, server_url)
        if not response_text:
            print("  No response — skipping")
            continue

        code = extract_code(response_text)
        if not validate_code(code, block):
            print("  Invalid/empty code — skipping")
            continue

        assembled_parts = seed_parts[:]
        assembled_parts[block_to_part_idx[block_idx]] = f"\n{code}\n"
        full_network = "# --OPTION--".join(assembled_parts)
        with open(model_path, "w") as f:
            f.write(full_network)

        manifest[gid] = {
            "gene_id": gid,
            "prompt": prompt,
            "generated_text": response_text,
            "block_idx": block_idx,
            "template": tmpl_name,
            "temperature": temperature,
            "combo_key": combo_key,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        if not args.no_submit:
            script = build_eval_script(gid, run_dir, venv_path, llm_root)
            job_id = submit_job(script)
            if job_id:
                manifest[gid]["eval_job_id"] = job_id
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                print(f"  Submitted eval job {job_id}")
                submitted += 1
        else:
            print("  --no-submit: skipping sbatch")

        generated += 1

    print(f"\nDone. Generated={generated}, Submitted={submitted}")
    print(f"Manifest: {manifest_path} ({len(manifest)} total entries)")
    print(f"\nOnce eval jobs finish, run:  python join_metrics.py")


if __name__ == "__main__":
    main()
