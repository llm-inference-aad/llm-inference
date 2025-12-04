#!/usr/bin/env python3
"""
Plot average accuracy vs generation number for LLM-guided evolution runs.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def find_run_directory(run_id: str) -> Path:
    """Find the run directory given a run_id or 'latest'."""
    runs_dir = Path(__file__).parent.parent / "runs"
    
    if run_id == "latest":
        # Find the most recent run directory
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            raise ValueError("No run directories found")
        return max(run_dirs, key=lambda d: d.stat().st_mtime)
    else:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")
        return run_dir


def parse_results_file(results_file: Path) -> dict:
    """Parse a results file to extract accuracy metrics.
    
    Expected CSV format: test_acc,params,valid_acc,runtime
    """
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            content = f.read().strip()
        
        # Parse CSV format: test_acc,params,valid_acc,runtime
        parts = content.split(',')
        if len(parts) >= 1:
            test_acc = float(parts[0])
            return {
                'test_acc': test_acc,
                'valid': True
            }
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse {results_file.name}: {e}")
        return None
    
    return None


def parse_slurm_log(log_file: Path) -> dict:
    """Parse SLURM log file to extract generation information for each individual."""
    if not log_file.exists():
        return {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    gene_to_generation = {}
    current_generation = None
    
    # Find all generation markers
    gen_pattern = r'STARTING GENERATION:\s*(\d+)'
    
    # Split content by generation markers
    lines = content.split('\n')
    
    for line in lines:
        # Check for generation marker
        gen_match = re.search(gen_pattern, line)
        if gen_match:
            current_generation = int(gen_match.group(1))
            continue
        
        # Look for gene IDs (format: xXx...)
        gene_matches = re.findall(r'(xXx[A-Za-z0-9]+)', line)
        if gene_matches and current_generation is not None:
            for gene_id in gene_matches:
                # Only record the first generation we see this gene in
                if gene_id not in gene_to_generation:
                    gene_to_generation[gene_id] = current_generation
    
    return gene_to_generation


def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs generation for LLM evolution runs"
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default="latest",
        help="Run ID or 'latest' for most recent run"
    )
    args = parser.parse_args()
    
    # Find run directory
    run_dir = find_run_directory(args.run_id)
    print(f"Analyzing run: {run_dir.name}")
    
    # Find SLURM log files
    logs_dir = run_dir / "logs"
    slurm_logs = list(logs_dir.glob("slurm-main-*.out"))
    if not slurm_logs:
        raise ValueError(f"No SLURM log files found in {logs_dir}")
    
    print(f"Found {len(slurm_logs)} SLURM log file(s)")
    
    # Parse generation information from all log files
    gene_to_generation = {}
    for slurm_log in slurm_logs:
        print(f"  Parsing {slurm_log.name}...")
        log_data = parse_slurm_log(slurm_log)
        # Merge, keeping earliest generation for each gene
        for gene_id, gen in log_data.items():
            if gene_id not in gene_to_generation:
                gene_to_generation[gene_id] = gen
    
    print(f"Found {len(gene_to_generation)} unique genes across all logs")
    
    # Count total population per generation (all genes assigned to each generation)
    generation_population = defaultdict(set)
    for gene_id, gen in gene_to_generation.items():
        generation_population[gen].add(gene_id)
    
    # Parse results files
    results_dir = run_dir / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    generation_accuracies = defaultdict(list)
    generation_valid_genes = defaultdict(set)
    
    for results_file in results_dir.glob("*_results.txt"):
        # Extract gene ID from filename
        gene_id_match = re.match(r'(xXx[A-Za-z0-9]+)_results\.txt', results_file.name)
        if not gene_id_match:
            continue
        
        gene_id = gene_id_match.group(1)
        
        # Get generation for this gene
        generation = gene_to_generation.get(gene_id)
        if generation is None:
            continue
        
        # Parse results
        results = parse_results_file(results_file)
        if results and results['valid']:
            generation_accuracies[generation].append(results['test_acc'])
            generation_valid_genes[generation].add(gene_id)
    
    if not generation_accuracies:
        print("No valid results found!")
        return
    
    # Calculate statistics per generation
    generations = sorted(generation_accuracies.keys())
    avg_accuracies = []
    std_accuracies = []
    min_accuracies = []
    max_accuracies = []
    valid_counts = []
    population_counts = []
    
    for gen in generations:
        accs = generation_accuracies[gen]
        avg_accuracies.append(np.mean(accs))
        std_accuracies.append(np.std(accs))
        min_accuracies.append(np.min(accs))
        max_accuracies.append(np.max(accs))
        valid_counts.append(len(accs))
        population_counts.append(len(generation_population[gen]))
    
    # Print statistics
    print("\nGeneration Statistics:")
    print(f"{'Gen':<6} {'Pop':<8} {'Valid':<8} {'Rate':<8} {'Avg Acc':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    for gen, pop, valid, avg, std, min_acc, max_acc in zip(
        generations, population_counts, valid_counts, avg_accuracies, std_accuracies, 
        min_accuracies, max_accuracies
    ):
        rate = valid / pop * 100 if pop > 0 else 0
        print(f"{gen:<6} {pop:<8} {valid:<8} {rate:<7.1f}% {avg:<10.4f} {std:<10.4f} {min_acc:<10.4f} {max_acc:<10.4f}")
    
    # Print overall summary
    total_pop = sum(population_counts)
    total_valid = sum(valid_counts)
    overall_rate = total_valid / total_pop * 100 if total_pop > 0 else 0
    print("-" * 80)
    print(f"{'Total':<6} {total_pop:<8} {total_valid:<8} {overall_rate:<7.1f}%")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot average accuracy with error bars
    ax.errorbar(
        generations, avg_accuracies, yerr=std_accuracies,
        marker='o', linestyle='-', linewidth=2, markersize=8,
        capsize=5, capthick=2, label='Average ± Std'
    )
    
    # Plot min/max range as shaded area
    ax.fill_between(
        generations, min_accuracies, max_accuracies,
        alpha=0.2, label='Min-Max Range'
    )
    
    # Formatting
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Evolution Progress - {run_dir.name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set x-axis to show integer generations
    ax.set_xticks(generations)
    
    # Add count annotations (showing valid/total)
    for gen, pop, valid, avg in zip(generations, population_counts, valid_counts, avg_accuracies):
        ax.annotate(
            f'{valid}/{pop}', 
            xy=(gen, avg), 
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=8,
            alpha=0.7
        )
    
    plt.tight_layout()
    
    # Save plot
    output_file = run_dir / "accuracy_vs_generation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
