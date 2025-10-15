#!/usr/bin/env python3
"""
Enhanced Pareto Front Plotting Script with Run Directory Support
================================================================

This script analyzes results from evolutionary runs and creates visualizations.

Features:
- Automatically detects run directories
- Plots Pareto fronts for multi-objective optimization
- Generates summary reports with metadata
- Supports comparing multiple runs

Usage:
    # Plot latest run
    python scripts/plot_pareto_enhanced.py
    
    # Plot specific run
    python scripts/plot_pareto_enhanced.py --run-id run_20250113_143022
    
    # Compare multiple runs
    python scripts/plot_pareto_enhanced.py --compare run1 run2 run3
    
    # Save to custom location
    python scripts/plot_pareto_enhanced.py --output my_plot.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def parse_result_file(filepath: str) -> Optional[Tuple[float, int, float, float]]:
    """
    Parse a result file containing: test_acc,total_params,val_acc,train_time
    
    Returns:
        Tuple of (test_acc, total_params, val_acc, train_time) or None if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            line = f.read().strip()
            test_acc, total_params, val_acc, train_time = line.split(',')
            return (float(test_acc), int(total_params), float(val_acc), float(train_time))
    except Exception as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return None


def load_run_metadata(run_dir: Path) -> Dict:
    """Load run metadata from JSON file"""
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {"run_id": run_dir.name, "status": "unknown"}


def find_latest_run(base_dir: Path = Path("runs")) -> Optional[Path]:
    """Find the most recent run directory"""
    latest_link = base_dir / "latest"
    if latest_link.exists() and latest_link.is_symlink():
        return latest_link.resolve()
    
    # Fallback: find newest directory by timestamp
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def load_results_from_run(run_dir: Path) -> List[Tuple[str, float, int, float, float]]:
    """
    Load all results from a run directory.
    
    Returns:
        List of (gene_id, test_acc, total_params, val_acc, train_time)
    """
    results = []
    results_dir = run_dir / "results"
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return results
    
    for result_file in results_dir.glob("*_results.txt"):
        gene_id = result_file.stem.replace("_results", "")
        parsed = parse_result_file(str(result_file))
        if parsed:
            test_acc, total_params, val_acc, train_time = parsed
            results.append((gene_id, test_acc, total_params, val_acc, train_time))
    
    return results


def compute_pareto_front(results: List[Tuple[str, float, int, float, float]]) -> List[int]:
    """
    Compute Pareto front indices for multi-objective optimization.
    Objectives: maximize test_acc, minimize total_params
    
    Returns:
        List of indices that are on the Pareto front
    """
    if not results:
        return []
    
    n = len(results)
    pareto_indices = []
    
    for i in range(n):
        is_dominated = False
        test_acc_i = results[i][1]
        params_i = results[i][2]
        
        for j in range(n):
            if i == j:
                continue
            test_acc_j = results[j][1]
            params_j = results[j][2]
            
            # j dominates i if j is better in at least one objective and not worse in the other
            if test_acc_j >= test_acc_i and params_j <= params_i:
                if test_acc_j > test_acc_i or params_j < params_i:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    return pareto_indices


def plot_single_run(run_dir: Path, output: str, show_plot: bool = False):
    """Plot results from a single run"""
    # Load metadata
    metadata = load_run_metadata(run_dir)
    run_id = metadata.get('run_id', run_dir.name)
    
    # Load results
    results = load_results_from_run(run_dir)
    
    if not results:
        print(f"No results found in {run_dir}")
        return
    
    print(f"Loaded {len(results)} results from run: {run_id}")
    
    # Extract data
    gene_ids = [r[0] for r in results]
    test_accs = [r[1] for r in results]
    params = [r[2] for r in results]
    
    # Compute Pareto front
    pareto_indices = compute_pareto_front(results)
    pareto_gene_ids = [gene_ids[i] for i in pareto_indices]
    pareto_test_accs = [test_accs[i] for i in pareto_indices]
    pareto_params = [params[i] for i in pareto_indices]
    
    # Sort Pareto front by parameters for line plot
    sorted_pairs = sorted(zip(pareto_params, pareto_test_accs, pareto_gene_ids))
    pareto_params_sorted = [p[0] for p in sorted_pairs]
    pareto_test_accs_sorted = [p[1] for p in sorted_pairs]
    pareto_gene_ids_sorted = [p[2] for p in sorted_pairs]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    plt.scatter(np.log10(params), test_accs, alpha=0.6, s=50, label='All architectures')
    
    # Plot Pareto front
    plt.scatter(np.log10(pareto_params_sorted), pareto_test_accs_sorted, 
                color='red', s=100, marker='*', label='Pareto front', zorder=5)
    plt.plot(np.log10(pareto_params_sorted), pareto_test_accs_sorted, 
             'r--', alpha=0.5, linewidth=2, zorder=4)
    
    # Annotate Pareto front points
    for i, gene_id in enumerate(pareto_gene_ids_sorted):
        plt.annotate(gene_id[:15], 
                     (np.log10(pareto_params_sorted[i]), pareto_test_accs_sorted[i]),
                     textcoords="offset points", xytext=(0,10), ha='center',
                     fontsize=8, alpha=0.7)
    
    # Labels and title
    plt.xlabel('log10(Model Parameters)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title(f'Pareto Front Analysis\nRun: {run_id}\n{len(results)} architectures evaluated', 
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metadata text box
    metadata_text = f"Status: {metadata.get('status', 'unknown')}\n"
    metadata_text += f"Created: {metadata.get('created_at', 'N/A')}\n"
    metadata_text += f"Branch: {metadata.get('git_branch', 'N/A')}\n"
    metadata_text += f"Pareto Front: {len(pareto_indices)} architectures"
    
    plt.text(0.02, 0.02, metadata_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Run Summary: {run_id}")
    print(f"{'='*60}")
    print(f"Total architectures: {len(results)}")
    print(f"Pareto front size: {len(pareto_indices)}")
    print(f"\nTop 5 by accuracy:")
    sorted_by_acc = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    for gene_id, acc, params, val_acc, time in sorted_by_acc:
        print(f"  {gene_id}: {acc:.4f} acc, {params:,} params")
    print(f"\nTop 5 by efficiency (acc/log10(params)):")
    sorted_by_eff = sorted(results, key=lambda x: x[1]/np.log10(x[2]), reverse=True)[:5]
    for gene_id, acc, params, val_acc, time in sorted_by_eff:
        eff = acc / np.log10(params)
        print(f"  {gene_id}: {eff:.4f} efficiency, {acc:.4f} acc, {params:,} params")


def plot_comparison(run_dirs: List[Path], output: str, show_plot: bool = False):
    """Plot comparison of multiple runs"""
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))
    
    for idx, run_dir in enumerate(run_dirs):
        metadata = load_run_metadata(run_dir)
        run_id = metadata.get('run_id', run_dir.name)
        results = load_results_from_run(run_dir)
        
        if not results:
            print(f"Warning: No results found for {run_id}")
            continue
        
        test_accs = [r[1] for r in results]
        params = [r[2] for r in results]
        
        plt.scatter(np.log10(params), test_accs, alpha=0.6, s=50, 
                   color=colors[idx], label=f'{run_id} ({len(results)} archs)')
    
    plt.xlabel('log10(Model Parameters)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title(f'Multi-Run Comparison\n{len(run_dirs)} runs', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Pareto front plotting with run directory support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--run-id', type=str, help='Specific run ID to plot')
    parser.add_argument('--compare', nargs='+', help='Compare multiple run IDs')
    parser.add_argument('--output', type=str, default='pareto_analysis.png',
                       help='Output filename (default: pareto_analysis.png)')
    parser.add_argument('--show', action='store_true', help='Display plot interactively')
    parser.add_argument('--base-dir', type=str, default='runs',
                       help='Base directory containing runs (default: runs)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Handle comparison mode
    if args.compare:
        run_dirs = [base_dir / run_id for run_id in args.compare]
        for run_dir in run_dirs:
            if not run_dir.exists():
                print(f"Error: Run directory not found: {run_dir}")
                sys.exit(1)
        plot_comparison(run_dirs, args.output, args.show)
        return
    
    # Single run mode
    if args.run_id:
        run_dir = base_dir / args.run_id
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
    else:
        run_dir = find_latest_run(base_dir)
        if not run_dir:
            print(f"Error: No run directories found in {base_dir}")
            print("Create a run first with: bash scripts/create_run.sh")
            sys.exit(1)
        print(f"Using latest run: {run_dir.name}")
    
    plot_single_run(run_dir, args.output, args.show)


if __name__ == '__main__':
    main()
