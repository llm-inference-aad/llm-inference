import json
import os
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

        # Find start of actual prompt content (skip marker and decorative stars)
        # Start searching after the marker
        curr = p_idx + len(prompt_marker)
        # Skip lines that start with '*' or are empty
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

        # Find end of prompt (start of TEXT marker)
        t_idx = content.find(text_marker, prompt_start_idx)
        if t_idx == -1:
            return None, None

        raw_prompt = content[prompt_start_idx:t_idx].strip()

        # Clean up Prompt (remove trailing * lines or [INFO] logs that appear before the next marker)
        prompt_lines = []
        for line in raw_prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith('*'): continue
            if '[INFO]' in line or '[DEBUG]' in line: continue
            prompt_lines.append(line)
        prompt = '\n'.join(prompt_lines).strip()

        # --- Extract Generated Text ---
        # Find start of text content
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

        # End of text is usually the CODE marker, or end of file if CODE marker is missing
        c_idx = content.find(code_marker, text_start_idx)
        if c_idx == -1:
            c_idx = len(content)

        raw_text = content[text_start_idx:c_idx].strip()

        # Clean up Text
        text_lines = []
        for line in raw_text.splitlines():
            if line.strip().startswith('*'): continue
            text_lines.append(line)
        text = '\n'.join(text_lines).strip()

        return prompt, text

    except Exception as e:
        print(f"Warning: Error parsing slurm file {slurm_path}: {e}")
        return None, None

def main():
    # Configuration paths
    metrics_dir = "metrics/data"
    results_dir = "sota/ExquisiteNetV2/results"
    output_file = "data/dataset.json"

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dataset = []

    # We need to join 3 data sources:
    # 1. metrics/data/e2e-latency-*.json   -> Provides mapping: job_id <-> gene_id
    # 2. slurm-{job_id}.out                -> Provides: prompt, generated_text
    # 3. sota/.../results/{gene_id}_*.txt  -> Provides: fitness

    # Note: We use e2e-latency files because they align with the slurm jobs available.
    json_files = glob.glob(os.path.join(metrics_dir, "e2e-latency-*.json"))
    print(f"Found {len(json_files)} metrics files to scan.")

    processed_job_ids = set()

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)

            requests = data.get("requests", [])
            for req in requests:
                job_id = req.get("job_id")
                gene_id = req.get("gene_id")

                # Skip invalid records
                if not job_id or not gene_id:
                    continue

                # Avoid duplicates if multiple files reference the same job
                if job_id in processed_job_ids:
                    continue

                # --- Step 1: Check if Fitness Result exists ---
                # The filename format is {gene_id}_results.txt
                result_file_path = os.path.join(results_dir, f"{gene_id}_results.txt")
                if not os.path.exists(result_file_path):
                    # If no result file, we can't have a complete record
                    continue

                # Read fitness
                try:
                    with open(result_file_path, 'r') as rf:
                        fitness_content = rf.read().strip()
                except Exception as e:
                    print(f"Error reading result file {result_file_path}: {e}")
                    continue

                # --- Step 2: Check if Slurm Output exists ---
                slurm_file_path = f"slurm-{job_id}.out"
                if not os.path.exists(slurm_file_path):
                    # If no slurm file, we can't get the prompt/text
                    continue

                # Parse Prompt and Text
                prompt, generated_text = parse_slurm_file(slurm_file_path)

                if not prompt or not generated_text:
                    # Parsing failed or content missing
                    continue

                # --- Step 3: Aggregation ---
                entry = {
                    "job_id": str(job_id),
                    "gene_id": str(gene_id),
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "fitness": fitness_content
                }
                dataset.append(entry)
                processed_job_ids.add(job_id)

        except Exception as e:
            print(f"Error reading JSON file {jf}: {e}")

    print(f"Aggregation complete. Found {len(dataset)} valid records.")

    # Save to disk
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset successfully saved to {output_file}")

if __name__ == "__main__":
    main()
