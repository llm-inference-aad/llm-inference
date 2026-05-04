import json
import os
import re
import glob

def parse_slurm_file(slurm_path):
    """
    Parses a Slurm output file to extract the Prompt and Generated Text.
    Returns (prompt, text) or (None, None) if parsing fails.
    """
    try:
        with open(slurm_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Markers used in the log file
        prompt_marker = "PROMPT TO LLM"
        text_marker = "TEXT FROM LLM"
        code_marker = "CODE FROM LLM"

        # --- Extract Prompt ---
        p_idx = content.find(prompt_marker)
        if p_idx == -1:
            return None, None

        curr = p_idx + len(prompt_marker)
        while curr < len(content):
            eol = content.find('\n', curr)
            if eol == -1: break
            line_start = eol + 1
            next_line_end = content.find('\n', line_start)
            if next_line_end == -1: next_line_end = len(content)
            line = content[line_start:next_line_end].strip()
            if not line.startswith('*') and len(line) > 0:
                curr = line_start
                break
            curr = line_start

        prompt_start_idx = curr
        t_idx = content.find(text_marker, prompt_start_idx)
        if t_idx == -1:
            return None, None

        raw_prompt = content[prompt_start_idx:t_idx].strip()
        prompt_lines = []
        for line in raw_prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith('*'): continue
            if '[INFO]' in line or '[DEBUG]' in line: continue
            prompt_lines.append(line)
        prompt = '\n'.join(prompt_lines).strip()

        # --- Extract Generated Text ---
        curr = t_idx + len(text_marker)
        while curr < len(content):
            eol = content.find('\n', curr)
            if eol == -1: break
            line_start = eol + 1
            next_line_end = content.find('\n', line_start)
            if next_line_end == -1: next_line_end = len(content)
            line = content[line_start:next_line_end].strip()
            if not line.startswith('*') and len(line) > 0:
                curr = line_start
                break
            curr = line_start

        text_start_idx = curr
        c_idx = content.find(code_marker, text_start_idx)
        if c_idx == -1:
            c_idx = len(content)

        raw_text = content[text_start_idx:c_idx].strip()
        text_lines = []
        for line in raw_text.splitlines():
            if line.strip().startswith('*'): continue
            text_lines.append(line)
        text = '\n'.join(text_lines).strip()

        return prompt, text

    except Exception as e:
        print(f"Warning: Error parsing slurm file {slurm_path}: {e}")
        return None, None


def extract_gene_id_from_slurm(slurm_path):
    """Extract gene_id from an llm-*.out file via the 'network_xXx...' pattern."""
    try:
        with open(slurm_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        m = re.search(r'network_(xXx[A-Za-z0-9]+)', content)
        return m.group(1) if m else None
    except Exception:
        return None


def collect_run_results(runs_dir):
    """
    Returns a dict: gene_id -> result_file_path
    by scanning runs/*/results/{gene_id}_results.txt
    """
    mapping = {}
    for result_file in glob.glob(os.path.join(runs_dir, "*/results/*_results.txt")):
        fname = os.path.basename(result_file)
        gene_id = fname.replace("_results.txt", "")
        if gene_id not in mapping:
            mapping[gene_id] = result_file
    return mapping


def main():
    output_file = "dataset.json"
    dataset = []
    processed_gene_ids = set()

    # ── Path 1: legacy structure ───────────────────────────────────────────────
    # metrics/data/e2e-latency-*.json  →  job_id/gene_id
    # slurm-{job_id}.out               →  prompt/text
    # sota/ExquisiteNetV2/results/     →  fitness
    metrics_dir = "metrics/data"
    legacy_results_dir = "sota/ExquisiteNetV2/results"

    json_files = glob.glob(os.path.join(metrics_dir, "e2e-latency-*.json"))
    print(f"[legacy] Found {len(json_files)} metrics files.")

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            for req in data.get("requests", []):
                job_id = req.get("job_id")
                gene_id = req.get("gene_id")
                if not job_id or not gene_id:
                    continue
                if gene_id in processed_gene_ids:
                    continue

                result_path = os.path.join(legacy_results_dir, f"{gene_id}_results.txt")
                if not os.path.exists(result_path):
                    continue

                slurm_path = f"slurm-{job_id}.out"
                if not os.path.exists(slurm_path):
                    continue

                prompt, generated_text = parse_slurm_file(slurm_path)
                if not prompt or not generated_text:
                    continue

                fitness = open(result_path).read().strip()
                dataset.append({
                    "job_id": str(job_id),
                    "gene_id": str(gene_id),
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "fitness": fitness,
                })
                processed_gene_ids.add(gene_id)

        except Exception as e:
            print(f"Warning: error reading {jf}: {e}")

    print(f"[legacy] {len(dataset)} records collected.")

    # ── Path 2: new run structure ──────────────────────────────────────────────
    # slurm-results/llm-{job_id}.out   →  gene_id (via regex) + prompt/text
    # runs/*/results/{gene_id}_results.txt  →  fitness
    run_gene_results = collect_run_results("runs")
    print(f"[new] Found {len(run_gene_results)} result files across runs/.")

    # Build job_id -> (gene_id, slurm_path) from llm-*.out files.
    # Some genes have multiple jobs (re-runs); use the lowest job_id (first attempt).
    gene_to_job = {}
    for slurm_path in sorted(glob.glob("slurm-results/llm-*.out")):
        fname = os.path.basename(slurm_path)
        m = re.match(r'llm-(\d+)\.out', fname)
        if not m:
            continue
        job_id = m.group(1)
        gene_id = extract_gene_id_from_slurm(slurm_path)
        if not gene_id:
            continue
        if gene_id not in gene_to_job:
            gene_to_job[gene_id] = (job_id, slurm_path)

    new_count = 0
    for gene_id, (job_id, slurm_path) in gene_to_job.items():
        if gene_id in processed_gene_ids:
            continue
        if gene_id not in run_gene_results:
            continue

        prompt, generated_text = parse_slurm_file(slurm_path)
        if not prompt or not generated_text:
            continue

        fitness = open(run_gene_results[gene_id]).read().strip()
        dataset.append({
            "job_id": str(job_id),
            "gene_id": str(gene_id),
            "prompt": prompt,
            "generated_text": generated_text,
            "fitness": fitness,
        })
        processed_gene_ids.add(gene_id)
        new_count += 1

    print(f"[new] {new_count} additional records collected.")

    # ── Path 3: synthetic manifest ─────────────────────────────────────────────
    # runs/*/synthetic_manifest.json  →  gene_id, prompt, generated_text
    # runs/*/results/{gene_id}_results.txt  →  fitness
    synth_count = 0
    for manifest_path in glob.glob("runs/*/synthetic_manifest.json"):
        try:
            manifest = json.load(open(manifest_path))
        except Exception as e:
            print(f"Warning: error reading {manifest_path}: {e}")
            continue

        run_dir = os.path.dirname(manifest_path)
        for gid, entry in manifest.items():
            if gid in processed_gene_ids:
                continue
            result_path = os.path.join(run_dir, "results", f"{gid}_results.txt")
            if not os.path.exists(result_path):
                continue
            fitness = open(result_path).read().strip()
            dataset.append({
                "job_id": entry.get("eval_job_id", "synthetic"),
                "gene_id": gid,
                "prompt": entry.get("prompt", ""),
                "generated_text": entry.get("generated_text", ""),
                "fitness": fitness,
            })
            processed_gene_ids.add(gid)
            synth_count += 1

    print(f"[synthetic] {synth_count} additional records collected.")
    print(f"Total: {len(dataset)} records.")

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
