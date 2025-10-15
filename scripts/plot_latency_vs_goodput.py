#!/usr/bin/env python3
"""
Plot latency vs goodput across generations.

Goodput is defined as the percentage of individuals per generation
that successfully obtained fitness scores (valid evaluations).

Usage:
    python scripts/plot_latency_vs_goodput.py --run-id my_run_20251013_143022 --run-hash abc123
    python scripts/plot_latency_vs_goodput.py --run-id latest --run-hash abc123
"""

import argparse
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
import sys

def load_checkpoints(run_dir):
    """Load all checkpoint files to analyze goodput per generation"""
    checkpoints_dir = Path(run_dir) / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    generations = []
    
    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_gen_*.pkl"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    
    for checkpoint_file in checkpoint_files:
        gen_num = int(checkpoint_file.stem.split('_')[-1])
        
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file}: {e}")
            continue
        
        population = data.get("population", [])
        global_data = data.get("GLOBAL_DATA", {})
        
        # Calculate goodput: individuals with valid fitness
        valid_fitness_count = 0
        total_count = len(population)
        
        for ind in population:
            if not ind:
                continue
            gene_id = ind[0] if isinstance(ind, (list, tuple)) else None
            if gene_id and gene_id in global_data:
                fitness = global_data[gene_id].get('fitness')
                # Check if fitness is valid (not None and has valid values)
                if fitness is not None:
                    if isinstance(fitness, (tuple, list)) and len(fitness) > 0:
                        if fitness[0] is not None:
                            valid_fitness_count += 1
                    elif isinstance(fitness, (int, float)):
                        valid_fitness_count += 1
        
        goodput = (valid_fitness_count / total_count * 100) if total_count > 0 else 0
        
        generations.append({
            'generation': gen_num,
            'population_size': total_count,
            'valid_fitness_count': valid_fitness_count,
            'goodput_percent': goodput
        })
    
    return sorted(generations, key=lambda x: x['generation'])

def load_latency_metrics(metrics_dir, run_hash):
    """Load latency metrics from JSON"""
    metrics_file = Path(metrics_dir) / "data" / f"-latency-{run_hash}.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def group_latencies_by_gene(metrics):
    """Group latency measurements by gene_id"""
    gene_latencies = defaultdict(list)
    
    for request in metrics.get("requests", []):
        gene_id = request.get("gene_id")
        if gene_id:
            gene_latencies[gene_id].append(request["_latency_sec"])
    
    # Return mean latency per gene
    return {gene_id: np.mean(latencies) for gene_id, latencies in gene_latencies.items()}

def calculate_generation_latencies(checkpoints_data, gene_latencies):
    """Calculate average latency per generation based on individuals in that generation"""
    gen_latencies = []
    
    for gen_data in checkpoints_data:
        # This is a simplified approach - ideally we'd have generation number in metrics
        # For now, we'll compute overall mean latency across all measured genes
        if gene_latencies:
            gen_latencies.append(np.mean(list(gene_latencies.values())))
        else:
            gen_latencies.append(None)
    
    return gen_latencies

def plot_latency_vs_goodput(generations, gen_latencies, output_path, run_id):
    """Generate dual-axis plot of latency and goodput over generations"""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    gen_nums = [g['generation'] for g in generations]
    goodputs = [g['goodput_percent'] for g in generations]
    
    # Plot goodput on left axis
    color_goodput = 'tab:blue'
    ax1.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Goodput (%)', fontsize=14, fontweight='bold', color=color_goodput)
    line1 = ax1.plot(gen_nums, goodputs,
                     marker='o', markersize=8, color=color_goodput, 
                     linewidth=2.5, label='Goodput', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_goodput, labelsize=11)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=100, color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax1.axhline(y=50, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add goodput values as annotations
    for i, (gen, goodput) in enumerate(zip(gen_nums, goodputs)):
        ax1.annotate(f'{goodput:.1f}%', 
                    (gen, goodput),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9,
                    color=color_goodput,
                    alpha=0.7)
    
    # Plot latency on right axis (if available)
    if gen_latencies and any(lat is not None for lat in gen_latencies):
        ax2 = ax1.twinx()
        color_latency = 'tab:red'
        ax2.set_ylabel('Avg LLM Latency (s)', fontsize=14, fontweight='bold', color=color_latency)
        
        # Filter out None values
        valid_data = [(gen, lat) for gen, lat in zip(gen_nums, gen_latencies) if lat is not None]
        if valid_data:
            valid_gens, valid_lats = zip(*valid_data)
            line2 = ax2.plot(valid_gens, valid_lats,
                            marker='s', markersize=8, color=color_latency,
                            linewidth=2.5, linestyle='--', label='LLM Latency', alpha=0.8)
            ax2.tick_params(axis='y', labelcolor=color_latency, labelsize=11)
            
            # Add latency values as annotations
            for gen, lat in zip(valid_gens, valid_lats):
                ax2.annotate(f'{lat:.2f}s', 
                            (gen, lat),
                            textcoords="offset points",
                            xytext=(0, -15),
                            ha='center',
                            fontsize=9,
                            color=color_latency,
                            alpha=0.7)
    
    plt.title(f'Evolution Performance Analysis: Goodput vs Latency\nRun: {run_id}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if gen_latencies and any(lat is not None for lat in gen_latencies):
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=11)
    else:
        ax1.legend(lines1, labels1, loc='lower left', fontsize=11)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print(f"LATENCY vs GOODPUT ANALYSIS")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}\n")
    
    print(f"📊 GENERATION-BY-GENERATION BREAKDOWN:")
    print(f"{'Gen':<6} {'Pop Size':<10} {'Valid':<8} {'Goodput':<10} {'Status':<15}")
    print(f"{'-'*60}")
    
    for gen in generations:
        status = "✅ Perfect" if gen['goodput_percent'] == 100 else \
                 "⚠️  Partial" if gen['goodput_percent'] >= 50 else \
                 "❌ Poor"
        print(f"{gen['generation']:<6} {gen['population_size']:<10} "
              f"{gen['valid_fitness_count']:<8} "
              f"{gen['goodput_percent']:<10.1f} {status:<15}")
    
    print(f"\n📈 SUMMARY STATISTICS:")
    mean_goodput = np.mean(goodputs)
    print(f"  Average goodput: {mean_goodput:.1f}%")
    print(f"  Best generation: Gen {gen_nums[np.argmax(goodputs)]} ({max(goodputs):.1f}%)")
    print(f"  Worst generation: Gen {gen_nums[np.argmin(goodputs)]} ({min(goodputs):.1f}%)")
    
    if gen_latencies and any(lat is not None for lat in gen_latencies):
        valid_lats = [lat for lat in gen_latencies if lat is not None]
        print(f"\n⚡ LATENCY STATISTICS:")
        print(f"  Mean latency: {np.mean(valid_lats):.3f}s")
        print(f"  Latency range: [{min(valid_lats):.3f}, {max(valid_lats):.3f}]s")
        
        # Correlation analysis
        # Align goodput and latency data
        aligned_data = [(g, l) for g, l in zip(goodputs, gen_latencies) if l is not None]
        if len(aligned_data) > 1:
            aligned_goodputs, aligned_lats = zip(*aligned_data)
            corr = np.corrcoef(aligned_goodputs, aligned_lats)[0, 1]
            print(f"\n🔗 CORRELATION:")
            print(f"  Goodput vs Latency: {corr:.3f}")
            if abs(corr) > 0.5:
                trend = "positive" if corr > 0 else "negative"
                print(f"  ⚠️  Strong {trend} correlation detected!")
    else:
        print(f"\n⚠️  No latency data available for correlation analysis")
    
    print(f"\n💡 INSIGHTS:")
    if mean_goodput >= 90:
        print(f"  ✅ Excellent evolution stability (>90% goodput)")
    elif mean_goodput >= 70:
        print(f"  ⚠️  Moderate evolution stability (70-90% goodput)")
    else:
        print(f"  ❌ Poor evolution stability (<70% goodput)")
        print(f"     Consider: reducing mutation rate, improving validation, or checking LLM generation")

def main():
    parser = argparse.ArgumentParser(
        description='Plot latency vs goodput across generations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='If no arguments provided, defaults to the latest run.'
    )
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run ID (e.g., my_run_20251013_143022) or "latest". Defaults to latest.')
    parser.add_argument('--run-hash', type=str, default=None,
                       help='Server run hash for metrics (auto-detected if not provided)')
    parser.add_argument('--metrics-dir', type=str, default='metrics',
                       help='Metrics directory (default: metrics)')
    parser.add_argument('--runs-dir', type=str, default='runs',
                       help='Runs directory (default: runs)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: scripts/plots/latency_vs_goodput_{run_id}.png)')
    
    args = parser.parse_args()
    
    # Resolve run directory (default to latest if not specified)
    runs_dir = Path(args.runs_dir)
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
    
    # Auto-detect run_hash from metrics directory if not provided
    if args.run_hash is None:
        print("ℹ️  Auto-detecting metrics hash...")
        metrics_dir = Path(args.metrics_dir) / 'data'
        if metrics_dir.exists():
            # Find the most recent latency metrics file
            latency_files = sorted(metrics_dir.glob('-latency-*.json'), 
                                 key=lambda p: p.stat().st_mtime, reverse=True)
            if latency_files:
                # Extract hash from filename: -latency-{hash}.json
                run_hash = latency_files[0].stem.split('-latency-')[1]
                print(f"ℹ️  Detected metrics hash: {run_hash}")
            else:
                print("ℹ️  No latency metrics found (continuing without metrics overlay)")
                run_hash = None
        else:
            print(f"ℹ️  Metrics directory not found (continuing without metrics overlay)")
            run_hash = None
    else:
        run_hash = args.run_hash
    
    # Set default output path if not specified
    if args.output is None:
        output_dir = Path('scripts/plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f'latency_vs_goodput_{run_id}.png')
    
    print(f"📊 Analyzing run: {run_id}")
    if run_hash:
        print(f"🔍 Metrics hash: {run_hash}")
    print(f"💾 Output: {args.output}")
    
    # Load checkpoint data
    try:
        generations = load_checkpoints(run_dir)
        print(f"✅ Loaded {len(generations)} generations from checkpoints")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Load latency data (optional)
    gen_latencies = None
    if run_hash:
        try:
            metrics = load_latency_metrics(args.metrics_dir, run_hash)
            gene_latencies = group_latencies_by_gene(metrics)
            gen_latencies = calculate_generation_latencies(generations, gene_latencies)
            print(f"✅ Loaded latency metrics (hash: {run_hash})")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Plotting goodput only (without latency)")
            gen_latencies = None
    else:
        print("Note: No run-hash provided, plotting goodput only")
        gen_latencies = [None] * len(generations)
    
    # Plot
    plot_latency_vs_goodput(generations, gen_latencies, args.output, run_id)

if __name__ == '__main__':
    main()
