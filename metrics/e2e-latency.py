#!/usr/bin/env python3
"""
End-to-End Latency Metrics Analysis Script

This script analyzes and displays important statistics about end-to-end latency 
for LLM inference runs. It can compare metrics across different runs and display
useful visualizations.

Usage:
    python metrics/-latency.py <run_hash>
    python metrics/-latency.py --list
    python metrics/-latency.py --compare <hash1> <hash2> [<hash3> ...]
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('default')
sns.set_palette("husl")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class LatencyAnalyzer:
    def __init__(self, metrics_dir: str | None = None):
        if metrics_dir is None:
            # Default to metrics/data directory relative to script location
            script_dir = Path(__file__).parent
            self.metrics_dir = script_dir / "data"
        else:
            self.metrics_dir = Path(metrics_dir)
        
        if not self.metrics_dir.exists():
            raise FileNotFoundError(f"Metrics directory not found: {self.metrics_dir}")
    
    def list_available_runs(self) -> List[str]:
        """List all available run hashes"""
        pattern = "-latency-*.json"
        files = list(self.metrics_dir.glob(pattern))
        run_hashes = []
        
        for file in files:
            # Extract run hash from filename
            hash_part = file.stem.replace("-latency-", "")
            run_hashes.append(hash_part)
        
        return sorted(run_hashes)
    
    def load_metrics(self, run_hash: str) -> Dict[str, Any]:
        """Load metrics for a specific run hash"""
        metrics_file = self.metrics_dir / f"-latency-{run_hash}.json"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def calculate_statistics(self, requests: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive statistics from request data"""
        if not requests:
            return {}
        
        # Extract latency data
        _latencies = [req["_latency_sec"] for req in requests]
        batch_times = [req["batch_processing_time_sec"] for req in requests]
        queue_times = [req.get("queue_wait_time_sec", 0) for req in requests if req.get("queue_wait_time_sec") is not None]
        batch_sizes = [req["batch_size"] for req in requests]
        prompt_lengths = [req["prompt_length"] for req in requests]
        max_tokens = [req["max_new_tokens"] for req in requests]
        
        stats = {
            # End-to-end latency stats
            "_mean": np.mean(_latencies),
            "_median": np.median(_latencies),
            "_std": np.std(_latencies),
            "_min": np.min(_latencies),
            "_max": np.max(_latencies),
            "_p95": np.percentile(_latencies, 95),
            "_p99": np.percentile(_latencies, 99),
            
            # Batch processing time stats
            "batch_mean": np.mean(batch_times),
            "batch_median": np.median(batch_times),
            "batch_std": np.std(batch_times),
            "batch_min": np.min(batch_times),
            "batch_max": np.max(batch_times),
            
            # Queue wait time stats (if available)
            "queue_mean": np.mean(queue_times) if queue_times else 0,
            "queue_median": np.median(queue_times) if queue_times else 0,
            "queue_max": np.max(queue_times) if queue_times else 0,
            
            # Batch size stats
            "avg_batch_size": np.mean(batch_sizes),
            "max_batch_size": np.max(batch_sizes),
            "min_batch_size": np.min(batch_sizes),
            
            # Request characteristics
            "avg_prompt_length": np.mean(prompt_lengths),
            "avg_max_tokens": np.mean(max_tokens),
            
            # Throughput metrics
            "total_requests": len(requests),
            "throughput_req_per_sec": len(requests) / sum(_latencies) if sum(_latencies) > 0 else 0
        }
        
        return stats
    
    def print_single_run_analysis(self, run_hash: str):
        """Print detailed analysis for a single run"""
        try:
            metrics = self.load_metrics(run_hash)
            requests = metrics.get("requests", [])
            
            if not requests:
                print(f"No request data found for run {run_hash}")
                return
            
            stats = self.calculate_statistics(requests)
            
            print(f"\n{'='*60}")
            print(f"LATENCY ANALYSIS FOR RUN: {run_hash}")
            print(f"{'='*60}")
            
            # Session metadata
            print(f"\n📊 SESSION METADATA:")
            print(f"   Start Time: {metrics.get('session_start', 'Unknown')}")
            print(f"   Model Path: {metrics.get('model_path', 'Unknown')}")
            print(f"   Batch Size: {metrics.get('batch_size', 'Unknown')}")
            print(f"   Batch Wait Time: {metrics.get('batch_wait_time', 'Unknown')}s")
            print(f"   Total Requests: {stats['total_requests']}")
            
            # End-to-end latency analysis
            print(f"\n🚀 END-TO-END LATENCY STATISTICS:")
            print(f"   Mean:       {stats['_mean']:.3f}s")
            print(f"   Median:     {stats['_median']:.3f}s")
            print(f"   Std Dev:    {stats['_std']:.3f}s")
            print(f"   Min:        {stats['_min']:.3f}s")
            print(f"   Max:        {stats['_max']:.3f}s")
            print(f"   95th %ile:  {stats['_p95']:.3f}s")
            print(f"   99th %ile:  {stats['_p99']:.3f}s")
            
            # Batch processing analysis
            print(f"\n⚡ BATCH PROCESSING STATISTICS:")
            print(f"   Mean:       {stats['batch_mean']:.3f}s")
            print(f"   Median:     {stats['batch_median']:.3f}s")
            print(f"   Std Dev:    {stats['batch_std']:.3f}s")
            print(f"   Min:        {stats['batch_min']:.3f}s")
            print(f"   Max:        {stats['batch_max']:.3f}s")
            
            # Queue wait analysis
            if stats['queue_mean'] > 0:
                print(f"\n⏳ QUEUE WAIT TIME STATISTICS:")
                print(f"   Mean:       {stats['queue_mean']:.3f}s")
                print(f"   Median:     {stats['queue_median']:.3f}s")
                print(f"   Max:        {stats['queue_max']:.3f}s")
            
            # Batch efficiency
            print(f"\n📦 BATCH EFFICIENCY:")
            print(f"   Avg Batch Size:  {stats['avg_batch_size']:.1f}")
            print(f"   Max Batch Size:  {stats['max_batch_size']}")
            print(f"   Min Batch Size:  {stats['min_batch_size']}")
            
            # Request characteristics
            print(f"\n📝 REQUEST CHARACTERISTICS:")
            print(f"   Avg Prompt Length:   {stats['avg_prompt_length']:.0f} chars")
            print(f"   Avg Max Tokens:      {stats['avg_max_tokens']:.0f}")
            
            # Performance metrics
            print(f"\n🎯 PERFORMANCE METRICS:")
            print(f"   Throughput:     {stats['throughput_req_per_sec']:.2f} req/sec")
            
            # Gene-specific analysis
            gene_requests = {}
            for req in requests:
                gene_id = req.get("gene_id")
                if gene_id:
                    if gene_id not in gene_requests:
                        gene_requests[gene_id] = []
                    gene_requests[gene_id].append(req)
            
            if gene_requests:
                print(f"\n🧬 GENE-SPECIFIC ANALYSIS:")
                print(f"   Total Unique Genes: {len(gene_requests)}")
                
                # Sort genes by request count for better display
                sorted_genes = sorted(gene_requests.items(), key=lambda x: len(x[1]), reverse=True)
                
                for gene_id, gene_reqs in sorted_genes[:10]:  # Show top 10 genes
                    gene_latencies = [req["_latency_sec"] for req in gene_reqs]
                    avg_latency = np.mean(gene_latencies)
                    min_latency = np.min(gene_latencies)
                    max_latency = np.max(gene_latencies)
                    request_count = len(gene_reqs)
                    print(f"   {gene_id}: {request_count} reqs, {avg_latency:.3f}s avg, {min_latency:.3f}s min, {max_latency:.3f}s max")
                
                if len(gene_requests) > 10:
                    print(f"   ... and {len(gene_requests) - 10} more genes")
            else:
                print(f"\n🧬 GENE-SPECIFIC ANALYSIS:")
                print(f"   No gene_id information found in requests")
            
            # Generate visualizations
            self.create_visualizations(run_hash, requests, stats)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error analyzing run {run_hash}: {e}")
    
    def create_visualizations(self, run_hash: str, requests: List[Dict], stats: Dict):
        """Create visualization charts for the metrics"""
        
        try:
            # Extract data for plotting
            _latencies = [req["_latency_sec"] for req in requests]
            batch_times = [req["batch_processing_time_sec"] for req in requests]
            queue_times = [req.get("queue_wait_time_sec", 0) for req in requests if req.get("queue_wait_time_sec") is not None]
            batch_sizes = [req["batch_size"] for req in requests]
            timestamps = [datetime.fromisoformat(req["timestamp"]) for req in requests]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Latency Analysis for Run: {run_hash}', fontsize=16, fontweight='bold')
            
            # 1. Latency distribution histogram
            axes[0, 0].hist(_latencies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(stats['_mean'], color='red', linestyle='--', label=f'Mean: {stats["_mean"]:.3f}s')
            axes[0, 0].axvline(stats['_median'], color='green', linestyle='--', label=f'Median: {stats["_median"]:.3f}s')
            axes[0, 0].set_xlabel('End-to-End Latency (seconds)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title(' Latency Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Latency over time
            axes[0, 1].plot(timestamps, _latencies, 'o-', alpha=0.7, markersize=4)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('End-to-End Latency (seconds)')
            axes[0, 1].set_title('Latency Over Time')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Batch size vs latency scatter plot
            axes[1, 0].scatter(batch_sizes, _latencies, alpha=0.6, c=batch_times, cmap='viridis')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('End-to-End Latency (seconds)')
            axes[1, 0].set_title('Batch Size vs  Latency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Component breakdown ( vs Batch processing)
            if queue_times:
                components = ['Queue Wait', 'Batch Processing', 'Other Overhead']
                avg_queue = np.mean(queue_times)
                avg_batch = stats['batch_mean']
                avg_overhead = stats['_mean'] - avg_batch - avg_queue
                values = [avg_queue, avg_batch, max(0, avg_overhead)]
            else:
                components = ['Batch Processing', 'Other Overhead']
                avg_batch = stats['batch_mean']
                avg_overhead = stats['_mean'] - avg_batch
                values = [avg_batch, max(0, avg_overhead)]
            
            axes[1, 1].pie(values, labels=components, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Average Latency Component Breakdown')
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"latency_analysis_{run_hash}.png"
            plot_path = self.metrics_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n📈 Visualization saved to: {plot_path}")
            
            # Show the plot
            plt.show()
            
            # Create gene-specific visualization if gene data exists
            self.create_gene_analysis_plot(run_hash, requests)
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def create_gene_analysis_plot(self, run_hash: str, requests: List[Dict]):
        """Create gene-specific analysis plot"""
        try:
            # Extract gene-specific data
            gene_requests = {}
            for req in requests:
                gene_id = req.get("gene_id")
                if gene_id:
                    if gene_id not in gene_requests:
                        gene_requests[gene_id] = []
                    gene_requests[gene_id].append(req)
            
            if not gene_requests:
                print("No gene_id data found - skipping gene analysis plot")
                return
            
            # Calculate gene statistics
            gene_stats = []
            for gene_id, gene_reqs in gene_requests.items():
                latencies = [req["_latency_sec"] for req in gene_reqs]
                gene_stats.append({
                    'gene_id': gene_id,
                    'request_count': len(gene_reqs),
                    'avg_latency': np.mean(latencies),
                    'min_latency': np.min(latencies),
                    'max_latency': np.max(latencies),
                    'std_latency': np.std(latencies) if len(latencies) > 1 else 0
                })
            
            # Sort by request count
            gene_stats.sort(key=lambda x: x['request_count'], reverse=True)
            
            # Take top 15 genes for visualization
            top_genes = gene_stats[:15]
            
            if len(top_genes) < 2:
                print("Not enough genes with data for gene analysis plot")
                return
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Gene-Specific Analysis for Run: {run_hash}', fontsize=14, fontweight='bold')
            
            # Plot 1: Request count per gene
            gene_names = [g['gene_id'][:8] + '...' if len(g['gene_id']) > 8 else g['gene_id'] for g in top_genes]
            request_counts = [g['request_count'] for g in top_genes]
            
            bars1 = ax1.bar(range(len(gene_names)), request_counts, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.set_xlabel('Gene ID')
            ax1.set_ylabel('Number of Requests')
            ax1.set_title('Request Count per Gene')
            ax1.set_xticks(range(len(gene_names)))
            ax1.set_xticklabels(gene_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars1, request_counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontsize=8)
            
            # Plot 2: Average latency per gene with error bars
            avg_latencies = [g['avg_latency'] for g in top_genes]
            std_latencies = [g['std_latency'] for g in top_genes]
            
            bars2 = ax2.bar(range(len(gene_names)), avg_latencies, yerr=std_latencies, 
                           alpha=0.7, color='lightcoral', edgecolor='black', capsize=3)
            ax2.set_xlabel('Gene ID')
            ax2.set_ylabel('Average  Latency (seconds)')
            ax2.set_title('Average Latency per Gene (with std dev)')
            ax2.set_xticks(range(len(gene_names)))
            ax2.set_xticklabels(gene_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, latency in zip(bars2, avg_latencies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{latency:.3f}s', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save the gene analysis plot
            gene_plot_filename = f"gene_analysis_{run_hash}.png"
            gene_plot_path = self.metrics_dir / gene_plot_filename
            plt.savefig(gene_plot_path, dpi=300, bbox_inches='tight')
            print(f"🧬 Gene analysis plot saved to: {gene_plot_path}")
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            print(f"Error creating gene analysis plot: {e}")
    
    def compare_runs(self, run_hashes: List[str]):
        """Compare metrics across multiple runs"""
        if len(run_hashes) < 2:
            print("Need at least 2 runs to compare")
            return
        
        print(f"\n{'='*80}")
        print(f"COMPARING {len(run_hashes)} RUNS")
        print(f"{'='*80}")
        
        comparison_data = []
        
        for run_hash in run_hashes:
            try:
                metrics = self.load_metrics(run_hash)
                requests = metrics.get("requests", [])
                
                if not requests:
                    print(f"Warning: No request data found for run {run_hash}")
                    continue
                
                stats = self.calculate_statistics(requests)
                stats['run_hash'] = run_hash
                stats['session_start'] = metrics.get('session_start', 'Unknown')
                comparison_data.append(stats)
                
            except Exception as e:
                print(f"Error loading run {run_hash}: {e}")
        
        if len(comparison_data) < 2:
            print("Not enough valid runs to compare")
            return
        
        # Create comparison table
        df = pd.DataFrame(comparison_data)
        
        # Display key metrics comparison
        comparison_cols = [
            'run_hash', 'total_requests', '_mean', '_median', '_p95', 
            'batch_mean', 'avg_batch_size', 'throughput_req_per_sec'
        ]
        
        print(f"\n📊 KEY METRICS COMPARISON:")
        print("-" * 120)
        
        # Format the dataframe for better display
        display_df = df[comparison_cols].copy()
        display_df.columns = ['Run Hash', 'Requests', ' Mean', ' Median', ' P95', 'Batch Mean', 'Avg Batch', 'Throughput']
        
        # Round numeric columns
        numeric_cols = [' Mean', ' Median', ' P95', 'Batch Mean', 'Avg Batch', 'Throughput']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        
        print(display_df.to_string(index=False))
        
        # Find best and worst performing runs
        best_ = df.loc[df['_mean'].idxmin()]
        worst_ = df.loc[df['_mean'].idxmax()]
        best_throughput = df.loc[df['throughput_req_per_sec'].idxmax()]
        
        print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
        print(f"   Best  Latency:    {best_['run_hash']} ({best_['_mean']:.3f}s avg)")
        print(f"   Worst  Latency:   {worst_['run_hash']} ({worst_['_mean']:.3f}s avg)")
        print(f"   Best Throughput:     {best_throughput['run_hash']} ({best_throughput['throughput_req_per_sec']:.2f} req/sec)")
        
        # Create comparison visualization
        self.create_comparison_visualization(comparison_data)
    
    def create_comparison_visualization(self, comparison_data: List[Dict]):
        """Create comparison charts across multiple runs"""
        if len(comparison_data) < 2:
            return
        
        try:
            df = pd.DataFrame(comparison_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Run Comparison Analysis', fontsize=16, fontweight='bold')
            
            # 1. Mean latency comparison
            axes[0, 0].bar(range(len(df)), df['_mean'], alpha=0.7, color='skyblue')
            axes[0, 0].set_xlabel('Run')
            axes[0, 0].set_ylabel('Mean  Latency (seconds)')
            axes[0, 0].set_title('Mean  Latency Comparison')
            axes[0, 0].set_xticks(range(len(df)))
            axes[0, 0].set_xticklabels([h[:8] + '...' for h in df['run_hash']], rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Throughput comparison
            axes[0, 1].bar(range(len(df)), df['throughput_req_per_sec'], alpha=0.7, color='lightgreen')
            axes[0, 1].set_xlabel('Run')
            axes[0, 1].set_ylabel('Throughput (req/sec)')
            axes[0, 1].set_title('Throughput Comparison')
            axes[0, 1].set_xticks(range(len(df)))
            axes[0, 1].set_xticklabels([h[:8] + '...' for h in df['run_hash']], rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. P95 latency comparison
            axes[1, 0].bar(range(len(df)), df['_p95'], alpha=0.7, color='coral')
            axes[1, 0].set_xlabel('Run')
            axes[1, 0].set_ylabel('P95  Latency (seconds)')
            axes[1, 0].set_title('P95 Latency Comparison')
            axes[1, 0].set_xticks(range(len(df)))
            axes[1, 0].set_xticklabels([h[:8] + '...' for h in df['run_hash']], rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Average batch size comparison
            axes[1, 1].bar(range(len(df)), df['avg_batch_size'], alpha=0.7, color='gold')
            axes[1, 1].set_xlabel('Run')
            axes[1, 1].set_ylabel('Average Batch Size')
            axes[1, 1].set_title('Average Batch Size Comparison')
            axes[1, 1].set_xticks(range(len(df)))
            axes[1, 1].set_xticklabels([h[:8] + '...' for h in df['run_hash']], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save comparison plot
            plot_filename = f"run_comparison_{len(comparison_data)}_runs.png"
            plot_path = self.metrics_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n📈 Comparison visualization saved to: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating comparison visualization: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze end-to-end latency metrics for LLM inference runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics/-latency.py abc123def456      # Analyze specific run
  python metrics/-latency.py --list            # List all available runs  
  python metrics/-latency.py --compare run1 run2 run3  # Compare multiple runs
        """
    )
    
    parser.add_argument('run_hash', nargs='?', help='Run hash to analyze')
    parser.add_argument('--list', action='store_true', help='List all available run hashes')
    parser.add_argument('--compare', nargs='+', help='Compare multiple runs')
    parser.add_argument('--metrics-dir', help='Custom metrics directory path')
    
    args = parser.parse_args()
    
    try:
        analyzer = LatencyAnalyzer(args.metrics_dir)
        
        if args.list:
            runs = analyzer.list_available_runs()
            if runs:
                print(f"\n📋 Available Run Hashes ({len(runs)} total):")
                print("-" * 50)
                for i, run_hash in enumerate(runs, 1):
                    try:
                        metrics = analyzer.load_metrics(run_hash)
                        session_start = metrics.get('session_start', 'Unknown')
                        request_count = len(metrics.get('requests', []))
                        print(f"{i:2d}. {run_hash} | {session_start} | {request_count} requests")
                    except:
                        print(f"{i:2d}. {run_hash} | Error loading metadata")
            else:
                print("No metrics files found.")
        
        elif args.compare:
            if len(args.compare) < 2:
                print("Error: Need at least 2 run hashes to compare")
                sys.exit(1)
            analyzer.compare_runs(args.compare)
        
        elif args.run_hash:
            analyzer.print_single_run_analysis(args.run_hash)
        
        else:
            # If no arguments, show available runs and prompt for selection
            runs = analyzer.list_available_runs()
            if runs:
                print(f"\n📋 Available Run Hashes ({len(runs)} total):")
                print("-" * 50)
                for i, run_hash in enumerate(runs, 1):
                    try:
                        metrics = analyzer.load_metrics(run_hash)
                        session_start = metrics.get('session_start', 'Unknown')
                        request_count = len(metrics.get('requests', []))
                        print(f"{i:2d}. {run_hash} | {session_start} | {request_count} requests")
                    except:
                        print(f"{i:2d}. {run_hash} | Error loading metadata")
                
                print(f"\nUsage: python {sys.argv[0]} <run_hash>")
                print(f"       python {sys.argv[0]} --compare <hash1> <hash2> [<hash3> ...]")
            else:
                print("No metrics files found. Start the server and make some requests first.")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()