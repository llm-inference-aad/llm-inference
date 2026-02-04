#!/usr/bin/env python3
"""
Aggregates GPU metrics (CSV) with latency data (JSON) for correlation analysis.
Outputs a combined dataset for visualization.

Usage:
    python aggregate_metrics.py --gpu metrics/gpu/server-*.csv --latency metrics/data/latency-*.json
"""

import argparse
import csv
import json
import glob
from datetime import datetime
from pathlib import Path


def parse_gpu_csv(filepath):
    """Parse nvidia-smi CSV output."""
    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Clean up column names (nvidia-smi adds spaces)
                cleaned = {k.strip(): v.strip() for k, v in row.items()}
                records.append({
                    'timestamp': cleaned.get('timestamp', ''),
                    'gpu_name': cleaned.get('name', ''),
                    'gpu_util_pct': cleaned.get('utilization.gpu [%]', '0').replace(' %', ''),
                    'mem_util_pct': cleaned.get('utilization.memory [%]', '0').replace(' %', ''),
                    'mem_total_mb': cleaned.get('memory.total [MiB]', '0').replace(' MiB', ''),
                    'mem_free_mb': cleaned.get('memory.free [MiB]', '0').replace(' MiB', ''),
                    'mem_used_mb': cleaned.get('memory.used [MiB]', '0').replace(' MiB', ''),
                })
            except Exception as e:
                print(f"Warning: Could not parse row: {e}")
    return records


def parse_latency_json(filepath):
    """Parse latency JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('requests', [])


def aggregate(gpu_files, latency_files, output_path):
    """Combine GPU and latency data."""
    all_gpu = []
    for f in gpu_files:
        all_gpu.extend(parse_gpu_csv(f))
    
    all_latency = []
    for f in latency_files:
        all_latency.extend(parse_latency_json(f))
    
    # Summary statistics
    summary = {
        'gpu_records': len(all_gpu),
        'latency_records': len(all_latency),
        'avg_gpu_util': 0,
        'avg_mem_util': 0,
        'avg_latency_sec': 0,
    }
    
    if all_gpu:
        try:
            utils = [float(r['gpu_util_pct']) for r in all_gpu if r['gpu_util_pct']]
            summary['avg_gpu_util'] = sum(utils) / len(utils) if utils else 0
            mems = [float(r['mem_util_pct']) for r in all_gpu if r['mem_util_pct']]
            summary['avg_mem_util'] = sum(mems) / len(mems) if mems else 0
        except ValueError:
            pass
    
    if all_latency:
        try:
            lats = [r.get('response_time_sec', 0) for r in all_latency]
            summary['avg_latency_sec'] = sum(lats) / len(lats) if lats else 0
        except (ValueError, TypeError):
            pass
    
    output = {
        'summary': summary,
        'gpu_samples': all_gpu[:100],  # First 100 samples
        'latency_samples': all_latency[:100],
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Aggregated {summary['gpu_records']} GPU records and {summary['latency_records']} latency records")
    print(f"Average GPU Util: {summary['avg_gpu_util']:.1f}%")
    print(f"Average Mem Util: {summary['avg_mem_util']:.1f}%")
    print(f"Average Latency: {summary['avg_latency_sec']:.1f}s")
    print(f"Output written to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate GPU and latency metrics')
    parser.add_argument('--gpu', nargs='*', default=[], help='GPU CSV files (glob patterns supported)')
    parser.add_argument('--latency', nargs='*', default=[], help='Latency JSON files (glob patterns supported)')
    parser.add_argument('--output', default='metrics/aggregated_metrics.json', help='Output file path')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    gpu_files = []
    for pattern in args.gpu:
        gpu_files.extend(glob.glob(pattern))
    
    latency_files = []
    for pattern in args.latency:
        latency_files.extend(glob.glob(pattern))
    
    if not gpu_files and not latency_files:
        # Default paths
        gpu_files = glob.glob('metrics/gpu/*.csv')
        latency_files = glob.glob('metrics/data/*.json')
    
    aggregate(gpu_files, latency_files, args.output)
