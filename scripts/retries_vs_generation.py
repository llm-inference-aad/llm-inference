#!/usr/bin/env python3
"""
Plot retry count vs generation number for LLM-guided evolution runs.

This script parses SLURM log files to count how many retry attempts occurred
in each generation and plots the results.
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


def parse_slurm_log_for_retries(log_file: Path) -> dict:
    """Parse SLURM log file to count retry attempts per generation.
    
    Returns a dictionary mapping generation number to retry count.
    """
    if not log_file.exists():
        return {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    generation_retries = defaultdict(int)
    current_generation = None
    
    # Pattern to match generation markers
    gen_pattern = r'STARTING GENERATION:\s*(\d+)'
    
    # Patterns to match retry attempts
    # "Attempt 1 failed validation:", "Attempt 2 validation error:", etc.
    attempt_pattern = r'Attempt (\d+).*(?:failed validation|validation error)'
    
    # Split content by lines
    lines = content.split('\n')
    
    for line in lines:
        # Check for generation marker
        gen_match = re.search(gen_pattern, line)
        if gen_match:
            current_generation = int(gen_match.group(1))
            continue
        
        # Check for retry attempts
        if current_generation is not None:
            attempt_match = re.search(attempt_pattern, line)
            if attempt_match:
                attempt_num = int(attempt_match.group(1))
                # Count retries (attempt 1 is the initial try, so retries = attempt_num - 1)
                # But we want to count each attempt line as a retry event
                generation_retries[current_generation] += 1
    
    return dict(generation_retries)


def parse_slurm_log_for_failed_mutations(log_file: Path) -> dict:
    """Parse SLURM log file to count failed mutations/matings per generation.
    
    This captures final failures after all retries are exhausted.
    Returns a dictionary mapping generation number to failure count.
    """
    if not log_file.exists():
        return {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    generation_failures = defaultdict(int)
    current_generation = None
    
    # Pattern to match generation markers
    gen_pattern = r'STARTING GENERATION:\s*(\d+)'
    
    # Patterns to match failures
    # "☠ Failed Mutated:", "☠ Failed Mated:", "‣ Failed to Submit Script."
    failure_patterns = [
        r'☠ Failed Mutated:',
        r'☠ Failed Mated:',
        r'‣ Failed to Submit Script\.',
        r'\*\s+Generated module failed validation\s+\*'
    ]
    
    lines = content.split('\n')
    
    for line in lines:
        # Check for generation marker
        gen_match = re.search(gen_pattern, line)
        if gen_match:
            current_generation = int(gen_match.group(1))
            continue
        
        # Check for failures
        if current_generation is not None:
            for pattern in failure_patterns:
                if re.search(pattern, line):
                    generation_failures[current_generation] += 1
                    break  # Only count once per line
    
    return dict(generation_failures)


def main():
    parser = argparse.ArgumentParser(
        description="Plot retry count vs generation for LLM evolution runs"
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default="latest",
        help="Run ID or 'latest' for most recent run"
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Also plot final failures (after all retries exhausted)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for the plot (default: saves to run directory)"
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
    
    # Parse retry information from all log files
    all_generation_retries = defaultdict(int)
    all_generation_failures = defaultdict(int)
    
    for slurm_log in slurm_logs:
        print(f"  Parsing {slurm_log.name}...")
        
        # Parse retries
        retry_data = parse_slurm_log_for_retries(slurm_log)
        for gen, count in retry_data.items():
            all_generation_retries[gen] += count
        
        # Parse failures if requested
        if args.include_failures:
            failure_data = parse_slurm_log_for_failed_mutations(slurm_log)
            for gen, count in failure_data.items():
                all_generation_failures[gen] += count
    
    if not all_generation_retries:
        print("No retry attempts found in the logs!")
        return
    
    # Prepare data for plotting
    generations = sorted(set(all_generation_retries.keys()) | set(all_generation_failures.keys()))
    retry_counts = [all_generation_retries.get(gen, 0) for gen in generations]
    
    # Print statistics
    print("\nRetry Statistics by Generation:")
    print(f"{'Gen':<6} {'Retries':<10}", end="")
    if args.include_failures:
        print(f" {'Failures':<10}", end="")
    print()
    print("-" * 30 if not args.include_failures else "-" * 40)
    
    for gen in generations:
        retries = all_generation_retries.get(gen, 0)
        print(f"{gen:<6} {retries:<10}", end="")
        if args.include_failures:
            failures = all_generation_failures.get(gen, 0)
            print(f" {failures:<10}", end="")
        print()
    
    print(f"\nTotal retries across all generations: {sum(retry_counts)}")
    if args.include_failures:
        total_failures = sum(all_generation_failures.values())
        print(f"Total final failures: {total_failures}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot retry counts
    bar_width = 0.35 if args.include_failures else 0.7
    x_pos = np.arange(len(generations))
    
    bars1 = ax.bar(
        x_pos - (bar_width/2 if args.include_failures else 0),
        retry_counts,
        bar_width,
        label='Retry Attempts',
        color='orange',
        alpha=0.8
    )
    
    # Plot failures if requested
    if args.include_failures:
        failure_counts = [all_generation_failures.get(gen, 0) for gen in generations]
        bars2 = ax.bar(
            x_pos + bar_width/2,
            failure_counts,
            bar_width,
            label='Final Failures',
            color='red',
            alpha=0.8
        )
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
    
    add_value_labels(bars1)
    if args.include_failures:
        add_value_labels(bars2)
    
    # Formatting
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Retry Attempts per Generation\nRun: {run_dir.name}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(generations)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save or show plot
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: save to run directory
        suffix = "_with_failures" if args.include_failures else ""
        output_path = run_dir / f"retries_vs_generation{suffix}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
