#!/usr/bin/env python3
"""
Extract successful mutation examples from runs for finetuning.

This creates a dataset of (prompt, successful_code) pairs from your
genetic evolution experiments.

Usage:
    python scripts/create_finetuning_dataset.py --run-id latest
    python scripts/create_finetuning_dataset.py --run-id auto_20251017_175557 --min-accuracy 0.7
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_training_data(run_dir, min_accuracy=0.5, output_file="finetuning_dataset.jsonl"):
    """
    Extract successful mutations from a run directory.
    
    Format: JSONL with {"prompt": "...", "completion": "..."}
    """
    run_path = Path(run_dir)
    results_dir = run_path / "results"
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return
    
    training_examples = []
    
    # Read all result files
    for result_file in sorted(results_dir.glob("*_results.txt")):
        gene_id = result_file.stem.replace("_results", "")
        
        # Read fitness metrics
        try:
            with open(result_file, 'r') as f:
                line = f.read().strip()
                if not line:
                    continue
                    
                metrics = line.split(',')
                if len(metrics) < 4:
                    continue
                    
                test_acc = float(metrics[0])
                
                # Only keep successful individuals
                if test_acc < min_accuracy:
                    continue
                
                # Find corresponding network file in sota/ExquisiteNetV2/models/
                # The gene_id from results file like "xXxABC123_results.txt" has the gene ID as "xXxABC123"
                project_root = Path(__file__).parent.parent
                network_file = project_root / "sota" / "ExquisiteNetV2" / "models" / f"network_{gene_id}.py"
                if not network_file.exists():
                    print(f"[WARN] Network file not found: {network_file}")
                    continue
                
                # Read the generated code
                with open(network_file, 'r') as f:
                    code = f.read()
                
                # TODO: Extract the original prompt that generated this
                # For now, we'll create a generic prompt
                prompt = f"Optimize this neural network architecture for CIFAR10 classification."
                
                training_examples.append({
                    "prompt": prompt,
                    "completion": code,
                    "metadata": {
                        "gene_id": gene_id,
                        "test_accuracy": test_acc,
                        "run_id": run_path.name
                    }
                })
                
        except Exception as e:
            print(f"[WARN] Failed to process {result_file}: {e}")
            continue
    
    # Write to JSONL format
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n[SUCCESS] Created dataset with {len(training_examples)} examples")
    print(f"Saved to: {output_path}")
    print(f"\nTop performers:")
    sorted_examples = sorted(training_examples, key=lambda x: x['metadata']['test_accuracy'], reverse=True)
    for ex in sorted_examples[:5]:
        print(f"  - {ex['metadata']['gene_id']}: {ex['metadata']['test_accuracy']:.4f} accuracy")


def main():
    parser = argparse.ArgumentParser(description="Create finetuning dataset from runs")
    parser.add_argument("--run-id", default="latest", help="Run ID or 'latest'")
    parser.add_argument("--min-accuracy", type=float, default=0.5, 
                       help="Minimum test accuracy to include")
    parser.add_argument("--output", default="finetuning_dataset.jsonl",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Resolve run directory
    if args.run_id == "latest":
        run_dir = Path("runs/latest")
        if not run_dir.exists():
            print("[ERROR] No 'latest' symlink found in runs/")
            sys.exit(1)
    else:
        run_dir = Path("runs") / args.run_id
        if not run_dir.exists():
            print(f"[ERROR] Run directory not found: {run_dir}")
            sys.exit(1)
    
    print(f"Extracting training data from: {run_dir}")
    extract_training_data(run_dir, args.min_accuracy, args.output)


if __name__ == "__main__":
    main()

