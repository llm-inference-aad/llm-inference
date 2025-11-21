#!/usr/bin/env python3
"""
Generate summary plots from a run's results and metrics.
"""

import json
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_results(results_dir):
    """Load all result files from a run."""
    results = []
    for result_file in sorted(results_dir.glob("*_results.txt")):
        gene_id = result_file.stem.replace("_results", "")
        try:
            with open(result_file, 'r') as f:
                line = f.read().strip()
                # Format: train_acc,train_samples,test_acc,time_taken
                train_acc, train_samples, test_acc, time_taken = map(float, line.split(','))
                results.append({
                    'gene_id': gene_id,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_samples': int(train_samples),
                    'time_taken': time_taken
                })
        except Exception as e:
            print(f"Warning: Could not parse {result_file}: {e}")
    return pd.DataFrame(results)

def load_metrics(metrics_dir):
    """Load metrics from a run."""
    metrics_files = list(metrics_dir.glob("latency-*.json"))
    if not metrics_files:
        return None
    
    with open(metrics_files[0], 'r') as f:
        data = json.load(f)
    
    requests_df = pd.DataFrame(data['requests'])
    return requests_df

def load_checkpoint(checkpoint_file):
    """Load evolution checkpoint."""
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None

def plot_results_summary(results_df, output_dir):
    """Generate summary plots from results."""
    if results_df.empty:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evolution Run Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Test Accuracy Distribution
    ax = axes[0, 0]
    ax.hist(results_df['test_acc'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(results_df['test_acc'].mean(), color='red', linestyle='--', 
               label=f'Mean: {results_df["test_acc"].mean():.3f}')
    ax.axvline(results_df['test_acc'].max(), color='green', linestyle='--',
               label=f'Best: {results_df["test_acc"].max():.3f}')
    ax.set_xlabel('Test Accuracy')
    ax.set_ylabel('Count')
    ax.set_title('Test Accuracy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Time vs Test Accuracy
    ax = axes[0, 1]
    scatter = ax.scatter(results_df['time_taken'], results_df['test_acc'], 
                        c=results_df['test_acc'], cmap='RdYlGn', 
                        s=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Training Time vs Test Accuracy')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Test Accuracy')
    
    # Plot 3: Train vs Test Accuracy
    ax = axes[1, 0]
    ax.scatter(results_df['train_acc'], results_df['test_acc'], 
              s=100, alpha=0.7, edgecolor='black', color='mediumpurple')
    # Add diagonal line for reference
    min_acc = min(results_df['train_acc'].min(), results_df['test_acc'].min())
    max_acc = max(results_df['train_acc'].max(), results_df['test_acc'].max())
    ax.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.5, label='Perfect correlation')
    ax.set_xlabel('Train Accuracy')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Train vs Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_data = [
        ['Total Evaluations', len(results_df)],
        ['', ''],
        ['Test Accuracy', ''],
        ['  Mean', f"{results_df['test_acc'].mean():.4f}"],
        ['  Std Dev', f"{results_df['test_acc'].std():.4f}"],
        ['  Min', f"{results_df['test_acc'].min():.4f}"],
        ['  Max', f"{results_df['test_acc'].max():.4f}"],
        ['', ''],
        ['Training Time (sec)', ''],
        ['  Mean', f"{results_df['time_taken'].mean():.2f}"],
        ['  Std Dev', f"{results_df['time_taken'].std():.2f}"],
        ['  Min', f"{results_df['time_taken'].min():.2f}"],
        ['  Max', f"{results_df['time_taken'].max():.2f}"],
    ]
    
    table = ax.table(cellText=stats_data, cellLoc='left',
                     colWidths=[0.6, 0.4],
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Style the header rows
    for i in [0, 2, 8]:
        table[(i, 0)].set_facecolor('#4472C4')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_facecolor('#4472C4')
    
    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = output_dir / 'run_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_llm_metrics(metrics_df, output_dir):
    """Generate plots from LLM metrics."""
    if metrics_df is None or metrics_df.empty:
        print("No LLM metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLM Inference Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Latency Distribution
    ax = axes[0, 0]
    ax.hist(metrics_df['_latency_sec'], bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(metrics_df['_latency_sec'].mean(), color='red', linestyle='--',
               label=f'Mean: {metrics_df["_latency_sec"].mean():.2f}s')
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('LLM Request Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Prompt Length vs Latency
    ax = axes[0, 1]
    scatter = ax.scatter(metrics_df['prompt_length'], metrics_df['_latency_sec'],
                        c=metrics_df['batch_size'], cmap='viridis',
                        s=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Prompt Length (tokens)')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Prompt Length vs Latency')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Batch Size')
    
    # Plot 3: Batch Processing Time
    ax = axes[1, 0]
    ax.hist(metrics_df['batch_processing_time_sec'], bins=20, edgecolor='black', 
           alpha=0.7, color='lightgreen')
    ax.axvline(metrics_df['batch_processing_time_sec'].mean(), color='red', linestyle='--',
               label=f'Mean: {metrics_df["batch_processing_time_sec"].mean():.2f}s')
    ax.set_xlabel('Batch Processing Time (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Batch Processing Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Requests Over Time
    ax = axes[1, 1]
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
    metrics_df = metrics_df.sort_values('timestamp')
    time_diffs = (metrics_df['timestamp'] - metrics_df['timestamp'].iloc[0]).dt.total_seconds()
    ax.plot(time_diffs, metrics_df['_latency_sec'], 'o-', alpha=0.7, markersize=6)
    ax.set_xlabel('Time Since Start (seconds)')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Request Latency Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'llm_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate summary plots from run data')
    parser.add_argument('--run-dir', type=str, default='runs/latest',
                       help='Path to run directory (default: runs/latest)')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    print(f"📊 Analyzing run: {run_dir.name}")
    
    # Create output directory
    output_dir = Path('scripts/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    results_df = load_results(run_dir / 'results')
    metrics_df = load_metrics(run_dir / 'metrics')
    
    # Generate plots
    if not results_df.empty:
        print(f"\n📈 Generating results summary plots...")
        print(f"   Found {len(results_df)} evaluation results")
        plot_results_summary(results_df, output_dir)
    
    if metrics_df is not None and not metrics_df.empty:
        print(f"\n⚡ Generating LLM metrics plots...")
        print(f"   Found {len(metrics_df)} LLM requests")
        plot_llm_metrics(metrics_df, output_dir)
    
    print(f"\n✨ All plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
