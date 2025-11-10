#!/usr/bin/env python3
"""
Analyze Fitness Inheritance Impact

This script analyzes the impact of fitness inheritance by comparing
inheritance events, GPU hours saved, and goodput metrics.

Usage:
    python scripts/analyze_inheritance_impact.py --run-id auto_20251021_123456
    python scripts/analyze_inheritance_impact.py --run-id latest
    python scripts/analyze_inheritance_impact.py --compare-with auto_20251017_175557
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def load_run_data(run_id):
    """Load all data for a given run."""
    repo_root = Path(__file__).parent.parent
    
    if run_id == "latest":
        # Find the most recent run directory
        runs_dir = repo_root / "runs"
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if not run_dirs:
            raise FileNotFoundError("No runs found in runs/ directory")
        run_dir = run_dirs[0]
        run_id = run_dir.name
    else:
        run_dir = repo_root / "runs" / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load metadata
    metadata_file = run_dir / "run_metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Load all checkpoints
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints = {}
    if checkpoints_dir.exists():
        for checkpoint_file in checkpoints_dir.glob("checkpoint_gen_*.pkl"):
            gen_num = int(checkpoint_file.stem.split('_')[-1])
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoints[gen_num] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load {checkpoint_file}: {e}")
    
    # Load log files
    logs_dir = run_dir / "logs"
    log_content = ""
    if logs_dir.exists():
        for log_file in logs_dir.glob("slurm-main-*.out"):
            try:
                with open(log_file, 'r') as f:
                    log_content += f.read()
            except Exception as e:
                print(f"Warning: Could not read {log_file}: {e}")
    
    return {
        'run_id': run_id,
        'run_dir': run_dir,
        'metadata': metadata,
        'checkpoints': checkpoints,
        'log_content': log_content
    }

def analyze_inheritance_events(log_content):
    """Extract and analyze fitness inheritance events from logs."""
    inheritance_events = []
    fallback_events = []
    
    # Pattern for inheritance events
    inheritance_pattern = r"Gene (xXx\w+) is a fallback clone of parent (\w+).*?Inheriting fitness \(([^)]+)\)"
    for match in re.finditer(inheritance_pattern, log_content, re.MULTILINE | re.DOTALL):
        gene_id = match.group(1)
        parent_id = match.group(2)
        fitness_str = match.group(3)
        
        # Parse fitness values
        fitness_values = [float(x.strip()) for x in fitness_str.split(',')]
        
        inheritance_events.append({
            'gene_id': gene_id,
            'parent_id': parent_id,
            'fitness': tuple(fitness_values)
        })
    
    # Pattern for fallback detection (without inheritance)
    fallback_pattern = r"Gene (xXx\w+) is a fallback, but parent (\w+) not yet evaluated"
    for match in re.finditer(fallback_pattern, log_content):
        gene_id = match.group(1)
        parent_id = match.group(2)
        
        fallback_events.append({
            'gene_id': gene_id,
            'parent_id': parent_id,
            'inherited': False
        })
    
    return inheritance_events, fallback_events

def analyze_goodput(checkpoints):
    """Calculate goodput statistics from checkpoints."""
    generation_stats = {}
    
    for gen_num, checkpoint in checkpoints.items():
        population = checkpoint.get('population', [])
        global_data = checkpoint.get('GLOBAL_DATA', {})
        
        total_count = len(population)
        fallback_count = 0
        
        for ind in population:
            if not ind:
                continue
            gene_id = ind[0] if isinstance(ind, (list, tuple)) else None
            if gene_id and gene_id in global_data:
                if global_data[gene_id].get('fallback', False):
                    fallback_count += 1
        
        fallback_percent = (fallback_count / total_count * 100) if total_count > 0 else 0
        goodput = 100.0 - fallback_percent
        
        generation_stats[gen_num] = {
            'population_size': total_count,
            'fallback_count': fallback_count,
            'fallback_percent': fallback_percent,
            'goodput_percent': goodput
        }
    
    return generation_stats

def estimate_gpu_hours_saved(inheritance_events, avg_eval_time_hours=2.0):
    """Estimate GPU hours saved by fitness inheritance."""
    return len(inheritance_events) * avg_eval_time_hours

def analyze_ancestry_correctness(checkpoints):
    """Check if ancestry tracking is working correctly."""
    ancestry_stats = {
        'total_individuals': 0,
        'correct_ancestry': 0,
        'self_reference_bugs': 0,
        'missing_ancestry': 0
    }
    
    for gen_num, checkpoint in checkpoints.items():
        ancestry_data = checkpoint.get('GLOBAL_DATA_ANCESTRY', {})
        
        for gene_id, ancestry in ancestry_data.items():
            ancestry_stats['total_individuals'] += 1
            
            genes = ancestry.get('GENES', [])
            if not genes:
                ancestry_stats['missing_ancestry'] += 1
                continue
            
            # Check first parent
            parent = genes[0]
            if parent == gene_id:
                ancestry_stats['self_reference_bugs'] += 1
            elif parent == 'network' or parent.startswith('xXx'):
                ancestry_stats['correct_ancestry'] += 1
    
    return ancestry_stats

def print_analysis_report(run_data, inheritance_events, fallback_events, goodput_stats, ancestry_stats):
    """Print comprehensive analysis report."""
    run_id = run_data['run_id']
    metadata = run_data['metadata']
    
    print("=" * 80)
    print(f"FITNESS INHERITANCE IMPACT ANALYSIS")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    if 'created_at' in metadata:
        print(f"Created: {metadata['created_at']}")
    if 'status' in metadata:
        print(f"Status: {metadata['status']}")
    print()
    
    # Inheritance Analysis
    print("🎯 FITNESS INHERITANCE ANALYSIS")
    print("-" * 40)
    print(f"Inheritance events detected: {len(inheritance_events)}")
    print(f"Fallback events (no inheritance): {len(fallback_events)}")
    
    if inheritance_events:
        print("\nInheritance Details:")
        for i, event in enumerate(inheritance_events, 1):
            fitness = event['fitness']
            print(f"  {i}. {event['gene_id'][:12]}... ← {event['parent_id']} "
                  f"fitness=({fitness[0]:.4f}, {fitness[1]:.0f})")
    
    total_fallbacks = len(inheritance_events) + len(fallback_events)
    if total_fallbacks > 0:
        inheritance_rate = len(inheritance_events) / total_fallbacks * 100
        print(f"\nInheritance Success Rate: {inheritance_rate:.1f}% ({len(inheritance_events)}/{total_fallbacks})")
    
    # GPU Hours Saved
    gpu_hours_saved = estimate_gpu_hours_saved(inheritance_events)
    print(f"Estimated GPU Hours Saved: {gpu_hours_saved:.1f} hours")
    print()
    
    # Goodput Analysis
    print("📈 GOODPUT ANALYSIS")
    print("-" * 40)
    if goodput_stats:
        generations = sorted(goodput_stats.keys())
        print(f"Generations analyzed: {len(generations)} (Gen {min(generations)}-{max(generations)})")
        
        avg_goodput = sum(s['goodput_percent'] for s in goodput_stats.values()) / len(goodput_stats)
        avg_fallback = sum(s['fallback_percent'] for s in goodput_stats.values()) / len(goodput_stats)
        
        print(f"Average goodput: {avg_goodput:.1f}%")
        print(f"Average fallback rate: {avg_fallback:.1f}%")
        
        print("\nPer-Generation Breakdown:")
        for gen in generations:
            stats = goodput_stats[gen]
            print(f"  Gen {gen}: {stats['goodput_percent']:.1f}% goodput "
                  f"({stats['fallback_count']}/{stats['population_size']} fallbacks)")
    
    print()
    
    # Ancestry Analysis
    print("🔬 ANCESTRY TRACKING ANALYSIS")
    print("-" * 40)
    total = ancestry_stats['total_individuals']
    correct = ancestry_stats['correct_ancestry']
    bugs = ancestry_stats['self_reference_bugs']
    missing = ancestry_stats['missing_ancestry']
    
    print(f"Total individuals: {total}")
    print(f"Correct ancestry: {correct} ({correct/total*100:.1f}%)" if total > 0 else "Correct ancestry: 0")
    print(f"Self-reference bugs: {bugs} ({bugs/total*100:.1f}%)" if total > 0 else "Self-reference bugs: 0")
    print(f"Missing ancestry: {missing} ({missing/total*100:.1f}%)" if total > 0 else "Missing ancestry: 0")
    
    if bugs > 0:
        print("🚨 WARNING: Self-reference bugs detected! Fitness inheritance will not work correctly.")
    elif correct > 0:
        print("✅ Ancestry tracking appears to be working correctly.")
    
    print()

def compare_runs(baseline_run_id, current_run_id):
    """Compare two runs to show improvement."""
    print("🔄 COMPARISON ANALYSIS")
    print("-" * 40)
    
    try:
        baseline_data = load_run_data(baseline_run_id)
        current_data = load_run_data(current_run_id)
        
        # Analyze both runs
        baseline_inheritance, baseline_fallbacks = analyze_inheritance_events(baseline_data['log_content'])
        current_inheritance, current_fallbacks = analyze_inheritance_events(current_data['log_content'])
        
        baseline_goodput = analyze_goodput(baseline_data['checkpoints'])
        current_goodput = analyze_goodput(current_data['checkpoints'])
        
        print(f"Baseline ({baseline_run_id}):")
        print(f"  Inheritance events: {len(baseline_inheritance)}")
        print(f"  GPU hours saved: {estimate_gpu_hours_saved(baseline_inheritance):.1f}")
        
        if baseline_goodput:
            avg_baseline_goodput = sum(s['goodput_percent'] for s in baseline_goodput.values()) / len(baseline_goodput)
            print(f"  Average goodput: {avg_baseline_goodput:.1f}%")
        
        print(f"\nCurrent ({current_run_id}):")
        print(f"  Inheritance events: {len(current_inheritance)}")
        print(f"  GPU hours saved: {estimate_gpu_hours_saved(current_inheritance):.1f}")
        
        if current_goodput:
            avg_current_goodput = sum(s['goodput_percent'] for s in current_goodput.values()) / len(current_goodput)
            print(f"  Average goodput: {avg_current_goodput:.1f}%")
        
        print("\nImprovement:")
        inheritance_improvement = len(current_inheritance) - len(baseline_inheritance)
        print(f"  Inheritance events: {inheritance_improvement:+d}")
        
        gpu_improvement = estimate_gpu_hours_saved(current_inheritance) - estimate_gpu_hours_saved(baseline_inheritance)
        print(f"  GPU hours saved: {gpu_improvement:+.1f}")
        
        if baseline_goodput and current_goodput:
            goodput_improvement = avg_current_goodput - avg_baseline_goodput
            print(f"  Goodput change: {goodput_improvement:+.1f}%")
        
    except Exception as e:
        print(f"Error comparing runs: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze fitness inheritance impact')
    parser.add_argument('--run-id', default='latest', 
                      help='Run ID to analyze (default: latest)')
    parser.add_argument('--compare-with', 
                      help='Baseline run ID for comparison (e.g., auto_20251017_175557)')
    
    args = parser.parse_args()
    
    try:
        # Load and analyze current run
        run_data = load_run_data(args.run_id)
        inheritance_events, fallback_events = analyze_inheritance_events(run_data['log_content'])
        goodput_stats = analyze_goodput(run_data['checkpoints'])
        ancestry_stats = analyze_ancestry_correctness(run_data['checkpoints'])
        
        # Print analysis report
        print_analysis_report(run_data, inheritance_events, fallback_events, goodput_stats, ancestry_stats)
        
        # Comparison if requested
        if args.compare_with:
            compare_runs(args.compare_with, run_data['run_id'])
        
        # Summary recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 40)
        
        if len(inheritance_events) == 0 and len(fallback_events) > 0:
            print("• Fitness inheritance is not working despite fallbacks detected")
            print("• Check ancestry tracking for self-reference bugs")
        elif len(inheritance_events) > 0:
            hours_saved = estimate_gpu_hours_saved(inheritance_events)
            print(f"• Fitness inheritance is working! Saved ~{hours_saved:.1f} GPU hours")
            if len(fallback_events) > 0:
                print(f"• Consider investigating why {len(fallback_events)} fallbacks didn't inherit")
        else:
            print("• No fallbacks detected - excellent goodput!")
        
        if ancestry_stats['self_reference_bugs'] > 0:
            print("• CRITICAL: Fix ancestry self-reference bugs")
        
        if goodput_stats:
            avg_goodput = sum(s['goodput_percent'] for s in goodput_stats.values()) / len(goodput_stats)
            if avg_goodput < 90:
                print(f"• Consider implementing error-aware prompting to improve {avg_goodput:.1f}% goodput")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())