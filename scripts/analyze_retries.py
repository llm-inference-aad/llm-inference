#!/usr/bin/env python3
"""
Analyze retry metrics from LLM-based genetic evolution.

This script reads the retry metrics JSON files and provides statistical analysis
of how many retries individuals took before generating valid code.

Usage:
    python scripts/analyze_retries.py checkpoints/metrics/retry_metrics_all.json
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def analyze_retry_metrics(metrics_file):
    """Analyze retry statistics from JSON metrics file."""
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No retry data found.")
        return
    
    # Basic statistics
    total_individuals = len(data)
    fallback_count = sum(1 for d in data if d['is_fallback'])
    success_count = total_individuals - fallback_count
    
    # Retry statistics (for successful individuals only)
    successful = [d for d in data if not d['is_fallback']]
    if successful:
        gen_retries = [d['generation_retries'] for d in successful]
        val_attempts = [d['validation_attempts'] for d in successful]
        
        avg_gen_retries = sum(gen_retries) / len(gen_retries)
        avg_val_attempts = sum(val_attempts) / len(val_attempts)
        max_gen_retries = max(gen_retries)
        max_val_attempts = max(val_attempts)
        min_gen_retries = min(gen_retries)
        min_val_attempts = min(val_attempts)
    else:
        avg_gen_retries = avg_val_attempts = 0
        max_gen_retries = max_val_attempts = 0
        min_gen_retries = min_val_attempts = 0
    
    # Fallback reasons
    fallback_reasons = defaultdict(int)
    for d in data:
        if d['is_fallback'] and d['fallback_reason']:
            # Truncate long error messages to first line
            reason = d['fallback_reason'].split('\n')[0][:100]
            fallback_reasons[reason] += 1
    
    # Per-generation statistics
    gen_stats = defaultdict(lambda: {'total': 0, 'fallback': 0, 'retries': []})
    for d in data:
        gen = d['generation']
        gen_stats[gen]['total'] += 1
        if d['is_fallback']:
            gen_stats[gen]['fallback'] += 1
        else:
            gen_stats[gen]['retries'].append(d['generation_retries'])
    
    # Print results
    print("=" * 80)
    print("RETRY METRICS ANALYSIS")
    print("=" * 80)
    print()
    
    print(f"Total Individuals: {total_individuals}")
    print(f"Successful: {success_count} ({100*success_count/total_individuals:.1f}%)")
    print(f"Fallback to Parent: {fallback_count} ({100*fallback_count/total_individuals:.1f}%)")
    print()
    
    if successful:
        print("SUCCESSFUL INDIVIDUALS (Valid Code Generated)")
        print("-" * 80)
        print(f"Average Generation Retries: {avg_gen_retries:.2f}")
        print(f"  Min: {min_gen_retries}, Max: {max_gen_retries}")
        print(f"Average Validation Attempts: {avg_val_attempts:.2f}")
        print(f"  Min: {min_val_attempts}, Max: {max_val_attempts}")
        print()
        
        # Distribution of retries
        retry_dist = defaultdict(int)
        for r in gen_retries:
            retry_dist[r] += 1
        
        print("Generation Retry Distribution:")
        for retries in sorted(retry_dist.keys()):
            count = retry_dist[retries]
            bar = '█' * int(50 * count / max(retry_dist.values()))
            print(f"  {retries} retries: {count:4d} {bar}")
        print()
    
    if fallback_reasons:
        print("TOP FALLBACK REASONS")
        print("-" * 80)
        sorted_reasons = sorted(fallback_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
        for reason, count in sorted_reasons:
            print(f"  [{count:3d}] {reason}")
        print()
    
    print("PER-GENERATION STATISTICS")
    print("-" * 80)
    print(f"{'Gen':>4} {'Total':>6} {'Fallback':>9} {'Success Rate':>13} {'Avg Retries':>12}")
    print("-" * 80)
    for gen in sorted(gen_stats.keys()):
        stats = gen_stats[gen]
        total = stats['total']
        fallback = stats['fallback']
        success_rate = 100 * (total - fallback) / total if total > 0 else 0
        avg_retries = sum(stats['retries']) / len(stats['retries']) if stats['retries'] else 0
        print(f"{gen:4d} {total:6d} {fallback:9d} {success_rate:12.1f}% {avg_retries:12.2f}")
    print()
    
    # Fitness correlation (if fitness data is available)
    fitness_data = [(d['generation_retries'], d['fitness']) 
                    for d in successful if d.get('fitness') is not None]
    
    if fitness_data:
        print("FITNESS CORRELATION")
        print("-" * 80)
        print("Analyzing correlation between retry count and fitness...")
        
        # Group by retry count
        retry_fitness = defaultdict(list)
        for retries, fitness in fitness_data:
            if isinstance(fitness, (list, tuple)) and len(fitness) >= 1:
                retry_fitness[retries].append(fitness[0])  # Use first objective
        
        print(f"{'Retries':>8} {'Count':>6} {'Avg Fitness':>12}")
        print("-" * 80)
        for retries in sorted(retry_fitness.keys()):
            fitnesses = retry_fitness[retries]
            count = len(fitnesses)
            avg_fitness = sum(fitnesses) / count
            print(f"{retries:8d} {count:6d} {avg_fitness:12.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Analyze LLM retry metrics')
    parser.add_argument('metrics_file', type=str, 
                       help='Path to retry_metrics_all.json file')
    
    args = parser.parse_args()
    
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        return 1
    
    analyze_retry_metrics(metrics_path)
    return 0


if __name__ == '__main__':
    exit(main())
