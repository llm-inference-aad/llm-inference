#!/usr/bin/env python3
"""
Enhanced Pareto Frontier Analysis with TPS Metrics

Creates multiple Pareto front visualizations:
1. TPS vs Quality (maximize both)
2. Latency vs TPS (minimize latency, maximize TPS)
3. Three-way: Latency vs TPS vs Quality
4. Original: Batch Time vs Quality

Usage:
    python metrics/plot-pareto-tps.py [data_dir]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from mpl_toolkits.mplot3d import Axes3D

class ParetoAnalyzer:
    def __init__(self, data_dir: str = "metrics/data"):
        self.data_path = Path(data_dir)
        self.runs = []
        
    def load_all_runs(self):
        """Load all run metrics from JSON files"""
        if not self.data_path.exists():
            print(f"❌ Directory '{self.data_path}' not found!")
            return False
        
        json_files = list(self.data_path.glob("e2e-latency-*.json"))
        
        if not json_files:
            print(f"❌ No latency JSON files found in '{self.data_path}'")
            return False
        
        print(f"📁 Found {len(json_files)} JSON files")
        
        for filepath in json_files:
            try:
                run_data = self._load_run_metrics(filepath)
                if run_data:
                    self.runs.append(run_data)
                    print(f"✓ {filepath.name}: {run_data['summary']}")
            except Exception as e:
                print(f"⚠️  Error processing {filepath.name}: {e}")
        
        if not self.runs:
            print("❌ No valid data points found!")
            return False
        
        print(f"\n📊 Processed {len(self.runs)} runs successfully")
        return True
    
    def _load_run_metrics(self, filepath: Path) -> Optional[Dict]:
        """Load and calculate metrics for a single run"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        requests = data.get("requests", [])
        run_hash = data.get("run_hash", filepath.stem)
        
        if not requests:
            return None
        
        # Extract all metrics
        batch_times = []
        e2e_latencies = []
        eval_scores = []
        tokens_per_second = []
        output_tokens = []
        
        for request in requests:
            if request.get("batch_processing_time_sec") is not None:
                batch_times.append(request["batch_processing_time_sec"])
            if request.get("e2e_latency_sec") is not None:
                e2e_latencies.append(request["e2e_latency_sec"])
            if request.get("evaluation_score") is not None:
                eval_scores.append(request["evaluation_score"])
            if request.get("tokens_per_second") is not None:
                tokens_per_second.append(request["tokens_per_second"])
            if request.get("output_tokens") is not None:
                output_tokens.append(request["output_tokens"])
        
        if not batch_times or not eval_scores:
            return None
        
        # Calculate averages
        metrics = {
            "run_hash": run_hash,
            "batch_time": np.mean(batch_times),
            "e2e_latency": np.mean(e2e_latencies) if e2e_latencies else None,
            "eval_score": np.mean(eval_scores),
            "tps": np.mean(tokens_per_second) if tokens_per_second else None,
            "output_tokens": np.mean(output_tokens) if output_tokens else None,
            "num_requests": len(requests),
            "has_tps": len(tokens_per_second) > 0
        }
        
        # Create summary string
        tps_str = f"TPS={metrics['tps']:.2f}" if metrics['tps'] else "No TPS"
        metrics["summary"] = f"time={metrics['batch_time']:.2f}s, score={metrics['eval_score']:.3f}, {tps_str}"
        
        return metrics
    
    def calculate_pareto_frontier(self, points: List[Tuple], objectives: List[str]) -> List[int]:
        """
        Calculate Pareto frontier indices for multi-objective optimization
        
        Args:
            points: List of tuples with values for each objective
            objectives: List of 'min' or 'max' for each objective
        
        Returns:
            List of indices that are on the Pareto frontier
        """
        pareto_indices = []
        n_objectives = len(objectives)
        
        for i, point_i in enumerate(points):
            is_dominated = False
            
            for j, point_j in enumerate(points):
                if i == j:
                    continue
                
                # Check if j dominates i
                dominates = True
                strictly_better_in_one = False
                
                for k, obj in enumerate(objectives):
                    if obj == 'min':
                        if point_j[k] > point_i[k]:  # j is worse
                            dominates = False
                            break
                        if point_j[k] < point_i[k]:  # j is better
                            strictly_better_in_one = True
                    else:  # obj == 'max'
                        if point_j[k] < point_i[k]:  # j is worse
                            dominates = False
                            break
                        if point_j[k] > point_i[k]:  # j is better
                            strictly_better_in_one = True
                
                if dominates and strictly_better_in_one:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def plot_tps_vs_quality(self, output_file: str = "pareto_tps_quality.png"):
        """Plot TPS vs Quality Pareto front"""
        # Filter runs with TPS data
        tps_runs = [r for r in self.runs if r['has_tps']]
        
        if not tps_runs:
            print("\n⚠️  No runs with TPS data found. Skipping TPS vs Quality plot.")
            return
        
        print(f"\n📈 Plotting TPS vs Quality (n={len(tps_runs)} runs with TPS data)")
        
        # Extract data
        points = [(r['tps'], r['eval_score']) for r in tps_runs]
        pareto_indices = self.calculate_pareto_frontier(points, ['max', 'max'])
        pareto_points = sorted([points[i] for i in pareto_indices], key=lambda x: x[0])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        tps_all = [p[0] for p in points]
        scores_all = [p[1] for p in points]
        tps_pareto = [p[0] for p in pareto_points]
        scores_pareto = [p[1] for p in pareto_points]
        
        # Plot all points
        ax.scatter(tps_all, scores_all,
                  alpha=0.6, s=100, c='lightblue',
                  edgecolors='steelblue', linewidth=1.5,
                  label=f'All Runs (n={len(points)})', zorder=3)
        
        # Plot Pareto frontier
        ax.scatter(tps_pareto, scores_pareto,
                  alpha=0.9, s=200, c='red',
                  edgecolors='darkred', linewidth=2,
                  marker='*', label=f'Pareto Optimal (n={len(pareto_indices)})', zorder=5)
        
        ax.plot(tps_pareto, scores_pareto, 'r--', alpha=0.6, linewidth=2, zorder=4)
        
        ax.set_xlabel('Tokens Per Second (TPS)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Evaluation Score', fontsize=13, fontweight='bold')
        ax.set_title('Pareto Frontier: Token Throughput vs Quality\nHigher is Better for Both',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotations
        for i, (tps, score) in enumerate(pareto_points):
            ax.annotate(f'P{i+1}',
                       (tps, score),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Statistics box
        stats_text = (
            f"Statistics:\n"
            f"Mean TPS: {np.mean(tps_all):.2f}\n"
            f"Mean Score: {np.mean(scores_all):.3f}\n"
            f"Pareto Efficiency: {len(pareto_indices)/len(points)*100:.1f}%"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
    
    def plot_latency_vs_tps(self, output_file: str = "pareto_latency_tps.png"):
        """Plot E2E Latency vs TPS Pareto front"""
        # Filter runs with both latency and TPS data
        valid_runs = [r for r in self.runs if r['has_tps'] and r['e2e_latency'] is not None]
        
        if not valid_runs:
            print("\n⚠️  No runs with both latency and TPS data. Skipping Latency vs TPS plot.")
            return
        
        print(f"\n📈 Plotting Latency vs TPS (n={len(valid_runs)} runs)")
        
        # Extract data (latency, TPS) - minimize latency, maximize TPS
        points = [(r['e2e_latency'], r['tps']) for r in valid_runs]
        pareto_indices = self.calculate_pareto_frontier(points, ['min', 'max'])
        pareto_points = sorted([points[i] for i in pareto_indices], key=lambda x: x[0])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        latency_all = [p[0] for p in points]
        tps_all = [p[1] for p in points]
        latency_pareto = [p[0] for p in pareto_points]
        tps_pareto = [p[1] for p in pareto_points]
        
        # Plot all points
        ax.scatter(latency_all, tps_all,
                  alpha=0.6, s=100, c='lightgreen',
                  edgecolors='darkgreen', linewidth=1.5,
                  label=f'All Runs (n={len(points)})', zorder=3)
        
        # Plot Pareto frontier
        ax.scatter(latency_pareto, tps_pareto,
                  alpha=0.9, s=200, c='red',
                  edgecolors='darkred', linewidth=2,
                  marker='*', label=f'Pareto Optimal (n={len(pareto_indices)})', zorder=5)
        
        ax.plot(latency_pareto, tps_pareto, 'r--', alpha=0.6, linewidth=2, zorder=4)
        
        ax.set_xlabel('End-to-End Latency (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Tokens Per Second (TPS)', fontsize=13, fontweight='bold')
        ax.set_title('Pareto Frontier: Latency vs Throughput\nLower Latency & Higher TPS is Better',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotations
        for i, (latency, tps) in enumerate(pareto_points):
            ax.annotate(f'P{i+1}',
                       (latency, tps),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Statistics box
        stats_text = (
            f"Statistics:\n"
            f"Mean Latency: {np.mean(latency_all):.2f}s\n"
            f"Mean TPS: {np.mean(tps_all):.2f}\n"
            f"Pareto Efficiency: {len(pareto_indices)/len(points)*100:.1f}%"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
    
    def plot_3d_pareto(self, output_file: str = "pareto_3d_tps.png"):
        """Plot 3D Pareto front: Latency vs TPS vs Quality"""
        valid_runs = [r for r in self.runs if r['has_tps'] and r['e2e_latency'] is not None]
        
        if not valid_runs:
            print("\n⚠️  No runs with complete data for 3D plot.")
            return
        
        print(f"\n📈 Plotting 3D Pareto Front (n={len(valid_runs)} runs)")
        
        # Extract data (latency, TPS, quality) - min latency, max TPS, max quality
        points = [(r['e2e_latency'], r['tps'], r['eval_score']) for r in valid_runs]
        pareto_indices = self.calculate_pareto_frontier(points, ['min', 'max', 'max'])
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        latency_all = [p[0] for p in points]
        tps_all = [p[1] for p in points]
        quality_all = [p[2] for p in points]
        
        # Plot all points
        ax.scatter(latency_all, tps_all, quality_all,
                  alpha=0.6, s=50, c='lightblue',
                  edgecolors='steelblue', linewidth=1,
                  label=f'All Runs (n={len(points)})')
        
        # Plot Pareto points
        latency_pareto = [points[i][0] for i in pareto_indices]
        tps_pareto = [points[i][1] for i in pareto_indices]
        quality_pareto = [points[i][2] for i in pareto_indices]
        
        ax.scatter(latency_pareto, tps_pareto, quality_pareto,
                  alpha=0.9, s=200, c='red',
                  edgecolors='darkred', linewidth=2,
                  marker='*', label=f'Pareto Optimal (n={len(pareto_indices)})')
        
        ax.set_xlabel('Latency (s)\n(Lower is Better)', fontsize=11, fontweight='bold')
        ax.set_ylabel('TPS\n(Higher is Better)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Quality Score\n(Higher is Better)', fontsize=11, fontweight='bold')
        ax.set_title('3D Pareto Frontier: Latency vs TPS vs Quality',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
    
    def plot_original_pareto(self, output_file: str = "pareto_baseline.png"):
        """Plot original Batch Time vs Quality Pareto front"""
        print(f"\n📈 Plotting Batch Time vs Quality (n={len(self.runs)} runs)")
        
        points = [(r['batch_time'], r['eval_score']) for r in self.runs]
        pareto_indices = self.calculate_pareto_frontier(points, ['min', 'max'])
        pareto_points = sorted([points[i] for i in pareto_indices], key=lambda x: x[0])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        times = [p[0] for p in points]
        scores = [p[1] for p in points]
        times_pareto = [p[0] for p in pareto_points]
        scores_pareto = [p[1] for p in pareto_points]
        
        ax.scatter(times, scores,
                  alpha=0.6, s=100, c='lightblue',
                  edgecolors='steelblue', linewidth=1.5,
                  label=f'All Runs (n={len(points)})', zorder=3)
        
        ax.scatter(times_pareto, scores_pareto,
                  alpha=0.9, s=200, c='red',
                  edgecolors='darkred', linewidth=2,
                  marker='*', label=f'Pareto Optimal (n={len(pareto_indices)})', zorder=5)
        
        ax.plot(times_pareto, scores_pareto, 'r--', alpha=0.6, linewidth=2, zorder=4)
        
        ax.set_xlabel('Average Batch Processing Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Evaluation Score', fontsize=13, fontweight='bold')
        ax.set_title('Pareto Frontier: Speed vs Quality\nLower Time & Higher Quality is Better',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        for i, (time, score) in enumerate(pareto_points):
            ax.annotate(f'P{i+1}',
                       (time, score),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        stats_text = (
            f"Statistics:\n"
            f"Mean Time: {np.mean(times):.3f}s\n"
            f"Mean Score: {np.mean(scores):.3f}\n"
            f"Pareto Efficiency: {len(pareto_indices)/len(points)*100:.1f}%"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
    
    def generate_all_plots(self, output_dir: str = "metrics/data"):
        """Generate all Pareto front plots"""
        output_path = Path(output_dir)
        
        print("="*70)
        print("GENERATING PARETO FRONTIER PLOTS")
        print("="*70)
        
        self.plot_original_pareto(str(output_path / "pareto_baseline.png"))
        self.plot_tps_vs_quality(str(output_path / "pareto_tps_quality.png"))
        self.plot_latency_vs_tps(str(output_path / "pareto_latency_tps.png"))
        self.plot_3d_pareto(str(output_path / "pareto_3d_tps.png"))
        
        print("\n" + "="*70)
        print("ALL PARETO PLOTS GENERATED!")
        print("="*70)
        print(f"📁 Output directory: {output_path}")
        print("\nGenerated files:")
        print("  • pareto_baseline.png      - Batch Time vs Quality")
        print("  • pareto_tps_quality.png   - TPS vs Quality")
        print("  • pareto_latency_tps.png   - Latency vs TPS")
        print("  • pareto_3d_tps.png        - 3D: Latency vs TPS vs Quality")


def main():
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "metrics/data"
    
    print("🚀 Starting Enhanced Pareto Frontier Analysis with TPS")
    print(f"📂 Data directory: {data_dir}\n")
    
    analyzer = ParetoAnalyzer(data_dir)
    
    if analyzer.load_all_runs():
        analyzer.generate_all_plots(data_dir)
        print("\n✅ Analysis complete!")
    else:
        print("\n❌ Analysis failed - no valid data found")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

