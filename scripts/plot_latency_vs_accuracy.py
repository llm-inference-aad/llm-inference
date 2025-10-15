#!/usr/bin/env python3
"""
Plot latency vs accuracy for a specific run.

Correlates LLM inference latency (from metrics/data/-latency-{RUN_HASH}.json)
with model test accuracy (from runs/{run_id}/results/*_results.txt).

Usage:
    python scripts/plot_latency_vs_accuracy.py --run-id my_run_20251013_143022 --run-hash abc123def456
    python scripts/plot_latency_vs_accuracy.py --run-id latest --run-hash abc123
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def load_latency_metrics(run_id, run_hash=None, metrics_dir=None):
    """
    Load latency metrics from JSON.
    
    Search order:
    1. runs/{run_id}/metrics/latency-{run_hash}.json (new structure)
    2. metrics/data/-latency-{run_hash}.json (legacy structure)
    3. Auto-detect most recent file in run metrics dir if run_hash not provided
    """
    # Try new structure first (run-specific)
    repo_root = Path(__file__).parent.parent
    run_metrics_dir = repo_root / "runs" / run_id / "metrics"
    if run_metrics_dir.exists():
        if run_hash:
            # Look for specific hash
            metrics_file = run_metrics_dir / f"latency-{run_hash}.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    return json.load(f)
        else:
            # Auto-detect most recent metrics file
            metrics_files = sorted(run_metrics_dir.glob("latency-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if metrics_files:
                print(f"Auto-detected metrics file: {metrics_files[0].name}")
                with open(metrics_files[0], 'r') as f:
                    return json.load(f)
    
    # Fall back to legacy structure
    if metrics_dir and run_hash:
        metrics_file = Path(metrics_dir) / "data" / f"-latency-{run_hash}.json"
        if metrics_file.exists():
            print(f"Using legacy metrics file: {metrics_file}")
            with open(metrics_file, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"Metrics file not found for run_id={run_id}, run_hash={run_hash}")

def load_accuracy_results(run_dir):
    """Load accuracy results from run directory"""
    results_dir = Path(run_dir) / "results"
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    gene_accuracy = {}
    
    for result_file in results_dir.glob("*_results.txt"):
        gene_id = result_file.stem.replace("_results", "")
        try:
            with open(result_file, 'r') as f:
                test_acc, params, val_acc, train_time = f.read().strip().split(',')
                gene_accuracy[gene_id] = {
                    'test_acc': float(test_acc),
                    'params': int(params),
                    'val_acc': float(val_acc),
                    'train_time': float(train_time)
                }
        except Exception as e:
            print(f"Warning: Could not parse {result_file}: {e}")
    
    return gene_accuracy

def correlate_latency_accuracy(metrics, accuracy_map):
    """
    Correlate latency with accuracy for each gene_id.
    Aggregates multiple requests per gene_id (mean latency).
    """
    gene_latencies = defaultdict(list)
    
    # Group latencies by gene_id
    for request in metrics.get("requests", []):
        gene_id = request.get("gene_id")
        if gene_id:
            gene_latencies[gene_id].append({
                "latency": request["_latency_sec"],
                "prompt_length": request["prompt_length"],
                "batch_size": request["batch_size"]
            })
    
    # Build correlation data
    data = []
    for gene_id, requests in gene_latencies.items():
        if gene_id in accuracy_map:
            mean_latency = np.mean([r["latency"] for r in requests])
            mean_prompt = np.mean([r["prompt_length"] for r in requests])
            
            data.append({
                "gene_id": gene_id,
                "latency_mean": mean_latency,
                "latency_std": np.std([r["latency"] for r in requests]),
                "num_requests": len(requests),
                "accuracy": accuracy_map[gene_id]['test_acc'],
                "params": accuracy_map[gene_id]['params'],
                "prompt_length": mean_prompt
            })
    
    return pd.DataFrame(data)

def plot_latency_vs_accuracy(df, output_path, run_id):
    """Generate scatter plot of latency vs accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Latency vs Accuracy
    scatter1 = ax1.scatter(df['latency_mean'], df['accuracy'], 
                          c=df['prompt_length'], 
                          s=100, alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Mean LLM Inference Latency (seconds)', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'Latency vs Accuracy\n{len(df)} individuals', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for prompt length
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Avg Prompt Length (chars)', fontsize=10)
    
    # Add correlation coefficient
    corr = df['latency_mean'].corr(df['accuracy'])
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax1.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Latency vs Parameters (colored by accuracy)
    scatter2 = ax2.scatter(df['latency_mean'], np.log10(df['params']), 
                          c=df['accuracy'], 
                          s=100, alpha=0.6, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Mean LLM Inference Latency (seconds)', fontsize=12)
    ax2.set_ylabel('log10(Model Parameters)', fontsize=12)
    ax2.set_title(f'Latency vs Model Size\n(colored by accuracy)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for accuracy
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Test Accuracy', fontsize=10)
    
    plt.suptitle(f'Latency Analysis for Run: {run_id}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"LATENCY vs ACCURACY ANALYSIS")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Total individuals analyzed: {len(df)}")
    print(f"\n📊 LATENCY STATISTICS:")
    print(f"  Mean latency: {df['latency_mean'].mean():.3f}s ± {df['latency_mean'].std():.3f}s")
    print(f"  Median latency: {df['latency_mean'].median():.3f}s")
    print(f"  Range: [{df['latency_mean'].min():.3f}, {df['latency_mean'].max():.3f}]s")
    print(f"\n🎯 ACCURACY STATISTICS:")
    print(f"  Mean accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"  Range: [{df['accuracy'].min():.4f}, {df['accuracy'].max():.4f}]")
    print(f"\n📈 CORRELATION ANALYSIS:")
    print(f"  Latency vs Accuracy: {corr:.3f}")
    print(f"  Latency vs Params: {df['latency_mean'].corr(np.log10(df['params'])):.3f}")
    print(f"  Accuracy vs Params: {df['accuracy'].corr(np.log10(df['params'])):.3f}")
    
    # Top/Bottom performers by accuracy
    print(f"\n🏆 TOP 5 BY ACCURACY:")
    top_acc = df.nlargest(5, 'accuracy')[['gene_id', 'accuracy', 'latency_mean', 'params']]
    for idx, row in top_acc.iterrows():
        print(f"  {row['gene_id'][:20]}: {row['accuracy']:.4f} acc, {row['latency_mean']:.3f}s lat, {row['params']:,} params")
    
    print(f"\n⚡ TOP 5 BY SPEED (lowest latency):")
    top_speed = df.nsmallest(5, 'latency_mean')[['gene_id', 'accuracy', 'latency_mean', 'params']]
    for idx, row in top_speed.iterrows():
        print(f"  {row['gene_id'][:20]}: {row['latency_mean']:.3f}s lat, {row['accuracy']:.4f} acc, {row['params']:,} params")
    
    print(f"\n💎 BEST TRADEOFFS (high accuracy, low latency):")
    # Calculate efficiency score: accuracy / latency
    df['efficiency'] = df['accuracy'] / df['latency_mean']
    top_eff = df.nlargest(5, 'efficiency')[['gene_id', 'accuracy', 'latency_mean', 'efficiency']]
    for idx, row in top_eff.iterrows():
        print(f"  {row['gene_id'][:20]}: {row['efficiency']:.4f} eff, {row['accuracy']:.4f} acc, {row['latency_mean']:.3f}s lat")

def main():
    parser = argparse.ArgumentParser(
        description='Plot latency vs accuracy analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='If no arguments provided, defaults to the latest run with auto-detected metrics hash.'
    )
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID (e.g., my_run_20251013_143022) or "latest". Defaults to latest.')
    parser.add_argument('--run-hash', type=str, default=None,
                       help='Server run hash for metrics (e.g., abc123def456). Auto-detected if not provided.')
    parser.add_argument('--metrics-dir', type=str, default='metrics', 
                       help='Metrics directory (default: metrics)')
    parser.add_argument('--runs-dir', type=str, default='runs', 
                       help='Runs directory (default: runs)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output filename (default: scripts/plots/latency_vs_accuracy_{run_id}.png)')
    
    args = parser.parse_args()
    
    # Resolve run directory (default to latest if not specified)
    runs_dir = Path(args.runs_dir)
    
    # Make runs_dir absolute if relative
    if not runs_dir.is_absolute():
        runs_dir = Path(__file__).parent.parent / runs_dir
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        print("Make sure to run this script from the repository root or specify --runs-dir")
        sys.exit(1)
    
    if args.run_id is None or args.run_id == 'latest':
        run_dir = runs_dir / 'latest'
        if run_dir.is_symlink():
            run_dir = run_dir.resolve()
            run_id = run_dir.name
            print(f"ℹ️  Using latest run: {run_id}")
        else:
            print("Error: runs/latest symlink not found")
            print("Available runs:")
            for d in sorted(runs_dir.iterdir(), reverse=True):
                if d.is_dir() and d.name != 'latest':
                    print(f"  - {d.name}")
            sys.exit(1)
    else:
        run_dir = runs_dir / args.run_id
        run_id = args.run_id
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Set default output path if not specified
    if args.output is None:
        output_dir = Path('scripts/plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f'latency_vs_accuracy_{run_id}.png')
    
    print(f"📊 Analyzing run: {run_id}")
    if args.run_hash:
        print(f"🔍 Metrics hash: {args.run_hash}")
    print(f"💾 Output: {args.output}")
    
    # Load metrics (will auto-detect hash if not provided)
    try:
        metrics = load_latency_metrics(run_id, args.run_hash, args.metrics_dir)
        print(f"✅ Loaded {len(metrics.get('requests', []))} metric requests")
        if 'run_hash' in metrics:
            print(f"🔍 Using metrics hash: {metrics['run_hash']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Load accuracy results
    try:
        accuracy_map = load_accuracy_results(run_dir)
        print(f"✅ Loaded {len(accuracy_map)} accuracy results")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Correlate data
    df = correlate_latency_accuracy(metrics, accuracy_map)
    
    if df.empty:
        print("Error: No matching gene_ids found between metrics and results")
        print(f"  Metrics has {len(set(r.get('gene_id') for r in metrics.get('requests', [])))} unique gene_ids")
        print(f"  Results has {len(accuracy_map)} gene_ids")
        sys.exit(1)
    
    print(f"✅ Matched {len(df)} individuals with both latency and accuracy data")
    
    # Plot
    plot_latency_vs_accuracy(df, args.output, run_id)

if __name__ == '__main__':
    main()
