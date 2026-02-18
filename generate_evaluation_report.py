#!/usr/bin/env python3
"""
Generate a comprehensive evaluation report from a completed run.
This combines metrics from SLURM logs and JSON metrics files.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import re

def extract_metrics_from_slurm_log(log_file):
    """Extract metrics from SLURM server log."""
    if not os.path.exists(log_file):
        return None
    
    metrics = {
        'vllm_enabled': None,
        'backend': None,
        'total_requests': 0,
        'session_duration_sec': 0,
        'avg_latency_sec': 0,
        'min_latency_sec': 0,
        'max_latency_sec': 0,
        'throughput_req_per_sec': 0,
        'throughput_req_per_min': 0,
        'avg_evaluation_score': 0,
        'process_memory_mb': 0,
        'gpu_memory_used_mb': 0,
        'gpu_memory_total_mb': 0,
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract vLLM status
        if 'vLLM STATUS: ENABLED' in content:
            metrics['vllm_enabled'] = True
            metrics['backend'] = 'vLLM (continuous batching)'
        elif 'vLLM STATUS: DISABLED' in content:
            metrics['vllm_enabled'] = False
            metrics['backend'] = 'HuggingFace (standard)'
        
        # Extract metrics using regex
        patterns = {
            'total_requests': r'Total Requests:\s+(\d+)',
            'session_duration_sec': r'Session Duration:\s+([\d.]+)\s+seconds',
            'avg_latency_sec': r'Average:\s+([\d.]+)\s+sec',
            'min_latency_sec': r'Min:\s+([\d.]+)\s+sec',
            'max_latency_sec': r'Max:\s+([\d.]+)\s+sec',
            'throughput_req_per_sec': r'([\d.]+)\s+requests/sec',
            'throughput_req_per_min': r'([\d.]+)\s+requests/min',
            'avg_evaluation_score': r'Average Evaluation Score:\s+([\d.]+)',
            'process_memory_mb': r'Process Memory:\s+([\d.]+)\s+MB',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1))
        
        # Extract GPU memory (last occurrence)
        gpu_matches = re.findall(r'GPU \d+.*?:\s+([\d.]+)\s+MB\s+/\s+([\d.]+)\s+MB', content)
        if gpu_matches:
            last_gpu = gpu_matches[-1]
            metrics['gpu_memory_used_mb'] = float(last_gpu[0])
            metrics['gpu_memory_total_mb'] = float(last_gpu[1])
    
    return metrics


def extract_metrics_from_json(metrics_dir):
    """Extract detailed metrics from JSON files."""
    if not os.path.exists(metrics_dir):
        return None
    
    # Find all latency JSON files
    json_files = list(Path(metrics_dir).glob('latency-*.json'))
    if not json_files:
        return None
    
    # Use the most recent one
    latest_json = max(json_files, key=os.path.getmtime)
    
    with open(latest_json, 'r') as f:
        data = json.load(f)
    
    return data


def generate_report(run_dir):
    """Generate comprehensive evaluation report."""
    run_dir = Path(run_dir)
    
    # Find SLURM server log
    logs_dir = run_dir / 'logs'
    server_logs = list(logs_dir.glob('slurm-server-*.out'))
    
    report = {
        'run_id': run_dir.name,
        'generated_at': datetime.now().isoformat(),
        'metrics': {},
        'summary': {}
    }
    
    # Extract from SLURM log
    if server_logs:
        server_log = server_logs[0]
        slurm_metrics = extract_metrics_from_slurm_log(server_log)
        if slurm_metrics:
            report['metrics']['slurm'] = slurm_metrics
    
    # Extract from JSON
    metrics_dir = run_dir / 'metrics'
    json_metrics = extract_metrics_from_json(metrics_dir)
    if json_metrics:
        report['metrics']['json'] = {
            'backend': json_metrics.get('backend'),
            'vllm_enabled': json_metrics.get('vllm_enabled'),
            'total_requests': len(json_metrics.get('requests', [])),
            'model_path': json_metrics.get('model_path'),
        }
        
        # Calculate additional statistics from requests
        requests = json_metrics.get('requests', [])
        if requests:
            latencies = [r['_latency_sec'] for r in requests if '_latency_sec' in r]
            scores = [r['evaluation_score'] for r in requests if r.get('evaluation_score') is not None]
            
            report['metrics']['detailed'] = {
                'latency_median_sec': sorted(latencies)[len(latencies)//2] if latencies else 0,
                'latency_p95_sec': sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
                'latency_p99_sec': sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
                'score_median': sorted(scores)[len(scores)//2] if scores else 0,
            }
    
    # Create summary combining best available data
    if 'slurm' in report['metrics']:
        m = report['metrics']['slurm']
        report['summary'] = {
            'vllm_enabled': m['vllm_enabled'],
            'backend': m['backend'],
            'total_requests': int(m['total_requests']),
            'avg_latency_sec': round(m['avg_latency_sec'], 4),
            'min_latency_sec': round(m['min_latency_sec'], 4),
            'max_latency_sec': round(m['max_latency_sec'], 4),
            'throughput_req_per_sec': round(m['throughput_req_per_sec'], 4),
            'throughput_req_per_min': round(m['throughput_req_per_min'], 2),
            'avg_evaluation_score': round(m['avg_evaluation_score'], 4),
            'process_memory_mb': round(m['process_memory_mb'], 2),
            'gpu_memory_used_mb': round(m['gpu_memory_used_mb'], 2),
            'gpu_memory_total_mb': round(m['gpu_memory_total_mb'], 2),
            'gpu_memory_util_percent': round((m['gpu_memory_used_mb'] / m['gpu_memory_total_mb'] * 100), 2) if m['gpu_memory_total_mb'] > 0 else 0,
        }
    
    return report


def print_report(report):
    """Print human-readable report."""
    print("\n" + "="*80)
    print(f"{'EVALUATION REPORT':^80}")
    print("="*80)
    print(f"Run ID: {report['run_id']}")
    print(f"Generated: {report['generated_at']}")
    
    if 'summary' in report and report['summary']:
        s = report['summary']
        print("\n" + "-"*80)
        print(f"{'CONFIGURATION':^80}")
        print("-"*80)
        vllm_status = "ENABLED ✅" if s.get('vllm_enabled') else "DISABLED ❌"
        print(f"vLLM Status: {vllm_status}")
        print(f"Backend: {s.get('backend', 'Unknown')}")
        
        print("\n" + "-"*80)
        print(f"{'PERFORMANCE METRICS':^80}")
        print("-"*80)
        print(f"Total Requests:         {s.get('total_requests', 0)}")
        print(f"")
        print(f"Latency (seconds):")
        print(f"  Average:              {s.get('avg_latency_sec', 0):.4f}")
        print(f"  Min:                  {s.get('min_latency_sec', 0):.4f}")
        print(f"  Max:                  {s.get('max_latency_sec', 0):.4f}")
        print(f"")
        print(f"Throughput:")
        print(f"  Requests/sec:         {s.get('throughput_req_per_sec', 0):.4f}")
        print(f"  Requests/min:         {s.get('throughput_req_per_min', 0):.2f}")
        
        print("\n" + "-"*80)
        print(f"{'RESOURCE USAGE':^80}")
        print("-"*80)
        print(f"Process Memory:         {s.get('process_memory_mb', 0):.2f} MB")
        print(f"GPU Memory Used:        {s.get('gpu_memory_used_mb', 0):.2f} MB")
        print(f"GPU Memory Total:       {s.get('gpu_memory_total_mb', 0):.2f} MB")
        print(f"GPU Memory Util:        {s.get('gpu_memory_util_percent', 0):.2f}%")
        
        print("\n" + "-"*80)
        print(f"{'MODEL PERFORMANCE':^80}")
        print("-"*80)
        print(f"Avg Evaluation Score:   {s.get('avg_evaluation_score', 0):.4f}")
        
        if 'detailed' in report.get('metrics', {}):
            d = report['metrics']['detailed']
            print(f"Latency Median:         {d.get('latency_median_sec', 0):.4f} sec")
            print(f"Latency P95:            {d.get('latency_p95_sec', 0):.4f} sec")
            print(f"Latency P99:            {d.get('latency_p99_sec', 0):.4f} sec")
    
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_evaluation_report.py <run_directory>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Generate report
    report = generate_report(run_dir)
    
    # Save to JSON
    output_file = Path(run_dir) / 'evaluation_report.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Evaluation report saved to: {output_file}")
    
    # Print human-readable version
    print_report(report)
    
    # Also save human-readable version
    txt_output = Path(run_dir) / 'evaluation_report.txt'
    with open(txt_output, 'w') as f:
        import io
        from contextlib import redirect_stdout
        
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(report)
        
        f.write(buf.getvalue())
    
    print(f"✅ Human-readable report saved to: {txt_output}")


if __name__ == '__main__':
    main()
