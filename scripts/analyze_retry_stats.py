#!/usr/bin/env python3
"""
Analyze retry statistics from checkpoint files.

Usage:
    python scripts/analyze_retry_stats.py <run_id>
    python scripts/analyze_retry_stats.py runs/baseline-metrics-2025-11-17/checkpoints
"""

import argparse
import pickle
import os
from pathlib import Path
import json


def load_checkpoint(checkpoint_path):
    """Load a checkpoint file and return the data."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def analyze_retry_stats(checkpoints_dir):
    """
    Analyze retry statistics across all checkpoint files.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        dict: Summary statistics
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Find all checkpoint files
    checkpoint_files = sorted(checkpoints_path.glob("checkpoint_gen_*.pkl"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoints_dir}")
        return None
    
    print(f"Found {len(checkpoint_files)} checkpoint files\n")
    print("=" * 80)
    print("RETRY STATISTICS ANALYSIS")
    print("=" * 80)
    
    all_stats = {}
    
    for checkpoint_file in checkpoint_files:
        # Extract generation number from filename
        gen_num = int(checkpoint_file.stem.split('_')[-1])
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_file)
        
        # Get retry stats (backwards compatible with old checkpoints)
        retry_stats = checkpoint.get("RETRY_STATS", {})
        
        if gen_num in retry_stats:
            stats = retry_stats[gen_num]
            all_stats[gen_num] = stats
            
            # Calculate average
            avg_retries = (stats['total_retries'] / stats['successful_individuals'] 
                          if stats['successful_individuals'] > 0 else 0)
            
            print(f"\nGeneration {gen_num}:")
            print(f"  ✓ Successful individuals: {stats['successful_individuals']}")
            print(f"  ↻ Total retries (cumulative): {stats['total_retries']}")
            print(f"  ✗ Failed attempts (this gen): {stats['failed_attempts']}")
            print(f"  📊 Average retries per individual: {avg_retries:.2f}")
        else:
            # Fallback: calculate from GLOBAL_DATA if RETRY_STATS not available
            global_data = checkpoint.get("GLOBAL_DATA", {})
            population = checkpoint.get("population", [])
            
            total_retries = 0
            successful_count = 0
            
            for ind in population:
                gene_id = ind[0]
                if gene_id in global_data:
                    retry_count = global_data[gene_id].get('retry_count', 0)
                    total_retries += retry_count
                    successful_count += 1
            
            if successful_count > 0:
                avg_retries = total_retries / successful_count
                all_stats[gen_num] = {
                    'total_retries': total_retries,
                    'successful_individuals': successful_count,
                    'failed_attempts': 0,  # Not available in old checkpoints
                }
                
                print(f"\nGeneration {gen_num} (reconstructed):")
                print(f"  ✓ Successful individuals: {successful_count}")
                print(f"  ↻ Total retries (cumulative): {total_retries}")
                print(f"  📊 Average retries per individual: {avg_retries:.2f}")
    
    # Overall statistics
    if all_stats:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        total_successful = sum(s['successful_individuals'] for s in all_stats.values())
        total_retries = sum(s['total_retries'] for s in all_stats.values())
        total_failed = sum(s.get('failed_attempts', 0) for s in all_stats.values())
        
        overall_avg = total_retries / total_successful if total_successful > 0 else 0
        
        print(f"\nAcross all {len(all_stats)} generations:")
        print(f"  ✓ Total successful individuals: {total_successful}")
        print(f"  ↻ Total retries (cumulative): {total_retries}")
        print(f"  ✗ Total failed attempts: {total_failed}")
        print(f"  📊 Overall average retries per individual: {overall_avg:.2f}")
        
        # Per-generation averages
        gen_averages = []
        for gen_num, stats in sorted(all_stats.items()):
            if stats['successful_individuals'] > 0:
                avg = stats['total_retries'] / stats['successful_individuals']
                gen_averages.append(avg)
        
        if gen_averages:
            import statistics
            print(f"\n  📈 Min avg retries per generation: {min(gen_averages):.2f}")
            print(f"  📈 Max avg retries per generation: {max(gen_averages):.2f}")
            print(f"  📈 Median avg retries per generation: {statistics.median(gen_averages):.2f}")
        
        # Save summary to JSON
        summary_file = checkpoints_path / "retry_stats_summary.json"
        summary_data = {
            "generations": all_stats,
            "overall": {
                "total_successful_individuals": total_successful,
                "total_retries": total_retries,
                "total_failed_attempts": total_failed,
                "overall_average_retries": overall_avg,
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📁 Summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze retry statistics from checkpoint files")
    parser.add_argument("checkpoints_dir", type=str, 
                       help="Path to checkpoints directory (e.g., runs/my-run/checkpoints)")
    
    args = parser.parse_args()
    
    analyze_retry_stats(args.checkpoints_dir)


if __name__ == "__main__":
    main()
