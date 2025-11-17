import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

def load_run_metrics(filepath: Path) -> Tuple[float, float, str]:
    """
    Load a single run's metrics and calculate averages
    
    Returns:
        (avg_batch_time, avg_eval_score, run_hash)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    requests = data.get("requests", [])
    run_hash = data.get("run_hash", filepath.stem)
    
    if not requests:
        return None
    
    # Extract metrics
    batch_times = []
    eval_scores = []
    
    for request in requests:
        batch_time = request.get("batch_processing_time_sec")
        eval_score = request.get("evaluation_score")
        
        if batch_time is not None:
            batch_times.append(batch_time)
        if eval_score is not None:
            eval_scores.append(eval_score)
    
    if not batch_times or not eval_scores:
        return None
    
    avg_batch_time = np.mean(batch_times)
    avg_eval_score = np.mean(eval_scores)
    
    return (avg_batch_time, avg_eval_score, run_hash)

def calculate_pareto_frontier(points: List[Tuple[float, float]]) -> List[int]:
    """
    Calculate Pareto frontier indices.
    We want to MINIMIZE batch time and MAXIMIZE eval score.
    
    Returns:
        List of indices that are on the Pareto frontier
    """
    pareto_indices = []
    
    print("\n🔍 Analyzing domination relationships...")
    
    for i, (time_i, score_i) in enumerate(points):
        is_dominated = False
        dominated_by = []
        
        for j, (time_j, score_j) in enumerate(points):
            if i == j:
                continue
            
            # Point j dominates point i if:
            # - j is at least as good in both dimensions (not worse)
            # - AND j is strictly better in at least one dimension
            faster = time_j < time_i
            better_quality = score_j > score_i
            not_worse_time = time_j <= time_i
            not_worse_quality = score_j >= score_i
            
            # j dominates i if it's better in at least one AND not worse in any
            if (faster or better_quality) and (not_worse_time and not_worse_quality):
                is_dominated = True
                dominated_by.append(j)
        
        if is_dominated:
            print(f"  Point {i} (time={time_i:.3f}, score={score_i:.3f}) is DOMINATED by points: {dominated_by}")
        else:
            print(f"  Point {i} (time={time_i:.3f}, score={score_i:.3f}) is PARETO OPTIMAL ⭐")
            pareto_indices.append(i)
    
    return pareto_indices

def plot_pareto_frontier(data_dir: str = "data", output_file: str = "pareto_baseline.png"):
    """
    Load all runs from data directory and plot Pareto frontier.
    
    Now also searches in runs/*/metrics.json for new format.
    """
    data_path = Path(data_dir)
    
    # Collect JSON files from both old and new locations
    json_files = []
    
    # Check new structure: runs/*/metrics.json
    repo_root = Path(__file__).parent.parent.parent
    runs_dir = repo_root / "runs"
    
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                metrics_file = run_dir / "metrics.json"
                if metrics_file.exists():
                    json_files.append(metrics_file)
    
    # Check old structure: data/*.json
    if data_path.exists():
        json_files.extend(list(data_path.glob("*.json")))
    
    if not json_files:
        print(f"❌ No metrics files found in '{data_dir}' or runs/*/metrics.json")
        return
    
    print(f"📁 Found {len(json_files)} metrics files")
    
    # Process each file
    all_points = []
    run_hashes = []
    
    for filepath in json_files:
        try:
            result = load_run_metrics(filepath)
            if result:
                avg_time, avg_score, run_hash = result
                all_points.append((avg_time, avg_score))
                run_hashes.append(run_hash)
                print(f"✓ {filepath.name}: avg_time={avg_time:.3f}s, avg_score={avg_score:.3f}")
        except Exception as e:
            print(f"⚠️  Error processing {filepath.name}: {e}")
    
    if not all_points:
        print("❌ No valid data points found!")
        return
    
    print(f"\n📊 Processed {len(all_points)} runs successfully")
    
    # Print all points for debugging
    print("\n" + "="*70)
    print("ALL DATA POINTS")
    print("="*70)
    print(f"{'Index':<8} {'Batch Time (s)':<18} {'Eval Score':<15} {'Run Hash':<20}")
    print("-"*70)
    for i, ((time, score), hash_val) in enumerate(zip(all_points, run_hashes)):
        print(f"{i:<8} {time:<18.4f} {score:<15.3f} {hash_val[:16]}")
    print("="*70)
    
    # Calculate Pareto frontier
    pareto_indices = calculate_pareto_frontier(all_points)
    pareto_points = [all_points[i] for i in pareto_indices]
    
    # Sort Pareto points by time for plotting
    pareto_points_sorted = sorted(pareto_points, key=lambda x: x[0])
    
    print(f"\n⭐ Found {len(pareto_indices)} Pareto optimal points (indices: {pareto_indices})")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract coordinates
    times = [p[0] for p in all_points]
    scores = [p[1] for p in all_points]
    
    pareto_times = [p[0] for p in pareto_points_sorted]
    pareto_scores = [p[1] for p in pareto_points_sorted]
    
    # Plot all points
    ax.scatter(times, scores, 
              alpha=0.6, s=100, c='lightblue', 
              edgecolors='steelblue', linewidth=1.5,
              label=f'All Runs (n={len(all_points)})', zorder=3)
    
    # Plot Pareto frontier points
    ax.scatter(pareto_times, pareto_scores, 
              alpha=0.9, s=200, c='red', 
              edgecolors='darkred', linewidth=2,
              marker='*', label=f'Pareto Optimal (n={len(pareto_indices)})', zorder=5)
    
    # Connect Pareto points with line
    ax.plot(pareto_times, pareto_scores, 
           'r--', alpha=0.6, linewidth=2, zorder=4)
    
    # Labels and formatting
    ax.set_xlabel('Average Batch Processing Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Evaluation Score', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Frontier: Speed vs Quality Baseline\nGenetic Algorithm Performance', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for Pareto points
    for i, (time, score) in enumerate(pareto_points_sorted):
        ax.annotate(f'P{i+1}', 
                   (time, score),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Calculate and display statistics
    stats_text = (
        f"Statistics:\n"
        f"Mean Time: {np.mean(times):.3f}s\n"
        f"Mean Score: {np.mean(scores):.3f}\n"
        f"Pareto Efficiency: {len(pareto_indices)/len(all_points)*100:.1f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path.absolute()}")
    
    # Also show the plot
    plt.show()
    
    # Print Pareto frontier details
    print("\n" + "="*70)
    print("PARETO FRONTIER DETAILS")
    print("="*70)
    print(f"{'Point':<8} {'Batch Time (s)':<18} {'Eval Score':<15}")
    print("-"*70)
    for i, (time, score) in enumerate(pareto_points_sorted):
        print(f"P{i+1:<7} {time:<18.4f} {score:<15.3f}")
    print("="*70)
    
    # Calculate hypervolume (simple version)
    if pareto_points_sorted:
        # Reference point: worst in both dimensions + margin
        ref_time = max(times) * 1.1
        ref_score = min(scores) * 0.9
        
        hypervolume = 0.0
        prev_time = 0.0
        
        for time, score in pareto_points_sorted:
            if score >= ref_score and time <= ref_time:
                width = time - prev_time
                height = score - ref_score
                hypervolume += width * height
                prev_time = time
        
        # Add final slice
        if pareto_points_sorted[-1][0] < ref_time:
            width = ref_time - pareto_points_sorted[-1][0]
            height = pareto_points_sorted[-1][1] - ref_score
            hypervolume += width * height
        
        print(f"\n📐 Hypervolume: {hypervolume:.4f}")
        print(f"   (Reference point: time={ref_time:.2f}s, score={ref_score:.3f})")

if __name__ == "__main__":
    import sys
    
    # Allow custom data directory as command line argument
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "pareto_baseline.png"
    
    print("🚀 Starting Pareto Frontier Analysis")
    print(f"📂 Data directory: {data_dir}")
    print(f"💾 Output file: {output_file}\n")
    
    plot_pareto_frontier(data_dir, output_file)
    
    print("\n✅ Analysis complete!")