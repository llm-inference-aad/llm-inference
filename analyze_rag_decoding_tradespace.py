#!/usr/bin/env python3
"""
Analyze parameter vs. latency tradespace across RAG + decoding benchmark runs.

Compares:
- RAG only (baseline latency, no constraints)
- RAG + speculative (baseline latency with speculation overhead)
- RAG + constrained (constrained latency, no speculation)
- RAG + constrained + speculative (full feature set)

Produces:
- tradespace_summary.json: per-config stats with latency, parameter estimates, and efficiency
- tradespace_report.md: human-readable tradespace analysis
- tradespace_chart_data.json: data formatted for charting (x=params, y=latency)
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_benchmark_summary(path: Path) -> Optional[Dict[str, Any]]:
    """Load a benchmark_summary.json file."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def extract_per_config_stats(summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract per-config latency and other metrics from benchmark summary."""
    result = {}
    if "per_config" not in summary:
        return result
    
    for config_name, config_data in summary["per_config"].items():
        if isinstance(config_data, dict):
            result[config_name] = {
                "count": config_data.get("count", 0),
                "latency_mean": config_data.get("latency_mean", 0),
                "latency_median": config_data.get("latency_median", 0),
                "latency_stdev": config_data.get("latency_stdev", 0),
                "latency_p95": config_data.get("latency_p95"),
                "latency_p99": config_data.get("latency_p99"),
                "spec_tokens_mean": config_data.get("spec_tokens_mean", 0),
                "score_mean": config_data.get("score_mean", 1.0),
            }
    return result


def estimate_parameter_overhead(config_name: str) -> Dict[str, Any]:
    """
    Estimate parameter count overhead for each configuration.
    
    This is a rough model:
    - baseline (1_rag_only): 100% parameters
    - speculative: +10% (for draft model, if used)
    - constrained: +5% (for constraint enforcement, minimal overhead)
    - combined: +15% (both features)
    """
    base_params = 8e9  # Assume ~8B parameter model as baseline
    
    config_lower = config_name.lower()
    
    if "both" in config_lower or ("constrained" in config_lower and "spec" in config_lower):
        scale = 1.15
        desc = "RAG + constrained + speculative"
    elif "spec" in config_lower:
        scale = 1.10
        desc = "RAG + speculative"
    elif "constrained" in config_lower:
        scale = 1.05
        desc = "RAG + constrained"
    else:
        scale = 1.0
        desc = "RAG only (baseline)"
    
    return {
        "description": desc,
        "param_overhead_percent": (scale - 1.0) * 100,
        "estimated_params": base_params * scale,
        "estimated_params_B": base_params * scale / 1e9,
    }


def compute_efficiency_metrics(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute efficiency metrics (latency per token, throughput, etc).
    
    Efficiency = tokens_per_second / parameters_in_billions
    """
    result = {}
    for config_name, data in stats.items():
        latency_mean = data.get("latency_mean", 0)
        spec_tokens_mean = data.get("spec_tokens_mean", 0)
        
        # Rough: if spec_tokens_mean is available, that's tokens speculated
        if spec_tokens_mean > 0:
            tokens_per_request = spec_tokens_mean  # Very rough estimate
            throughput = tokens_per_request / latency_mean if latency_mean > 0 else 0
        else:
            # Assume ~256 tokens generated per request (rough default)
            tokens_per_request = 256
            throughput = tokens_per_request / latency_mean if latency_mean > 0 else 0
        
        param_info = estimate_parameter_overhead(config_name)
        param_billions = param_info["estimated_params_B"]
        efficiency = throughput / param_billions if param_billions > 0 else 0
        
        result[config_name] = {
            "latency_mean_sec": latency_mean,
            "tokens_per_request": tokens_per_request,
            "throughput_tps": throughput,
            "estimated_params_B": param_billions,
            "efficiency_tps_per_B": efficiency,
            **param_info,
        }
    
    return result


def generate_tradespace_summary(
    benchmark_path: Path,
    output_dir: Path,
) -> None:
    """
    Generate tradespace analysis from a benchmark_summary.json.
    """
    summary = load_benchmark_summary(benchmark_path)
    if not summary:
        print(f"No benchmark summary found at {benchmark_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract per-config stats
    per_config_stats = extract_per_config_stats(summary)
    
    # Compute efficiency metrics
    efficiency = compute_efficiency_metrics(per_config_stats)
    
    # Generate tradespace summary JSON
    tradespace = {
        "timestamp": summary.get("timestamp"),
        "total_requests": summary.get("num_requests", 0),
        "successful": summary.get("successful", 0),
        "failed": summary.get("failed", 0),
        "benchmark_source": str(benchmark_path),
        "configs": efficiency,
    }
    
    summary_path = output_dir / "tradespace_summary.json"
    with summary_path.open("w") as f:
        json.dump(tradespace, f, indent=2)
    print(f"Wrote: {summary_path}")
    
    # Generate chart data (x=params, y=latency)
    chart_data = []
    for config_name, metrics in efficiency.items():
        chart_data.append({
            "name": config_name,
            "x": metrics["estimated_params_B"],  # x-axis: parameters
            "y": metrics["latency_mean_sec"],     # y-axis: latency
            "size": metrics["throughput_tps"],    # bubble size: throughput
            "label": metrics["description"],
        })
    
    chart_path = output_dir / "tradespace_chart_data.json"
    with chart_path.open("w") as f:
        json.dump({
            "data": chart_data,
            "x_label": "Estimated Parameters (Billions)",
            "y_label": "Mean Latency (seconds)",
            "size_label": "Throughput (tokens/sec)",
        }, f, indent=2)
    print(f"Wrote: {chart_path}")
    
    # Generate markdown report
    report_lines = [
        "# RAG + Decoding Tradespace Analysis\n",
        f"**Timestamp:** {summary.get('timestamp')}\n",
        f"**Total Requests:** {summary.get('num_requests', 0)} ({summary.get('successful', 0)} successful, {summary.get('failed', 0)} failed)\n\n",
        "## Summary\n",
        "This analysis evaluates the parameter-latency tradespace across RAG + decoding configurations.\n\n",
        "### Key Metrics\n\n",
        "| Config | Latency (s) | Est. Params (B) | Throughput (tok/s) | Efficiency |\n",
        "|--------|-------------|-----------------|-------------------|------------|\n",
    ]
    
    for config_name in sorted(efficiency.keys()):
        m = efficiency[config_name]
        latency = m["latency_mean_sec"]
        params = m["estimated_params_B"]
        throughput = m["throughput_tps"]
        eff = m["efficiency_tps_per_B"]
        report_lines.append(f"| {config_name} | {latency:.3f} | {params:.2f} | {throughput:.2f} | {eff:.4f} |\n")
    
    report_lines.append("\n## Analysis\n\n")
    
    # Compute deltas vs baseline
    baseline_config = next((c for c in efficiency.keys() if "rag_only" in c.lower()), None)
    if baseline_config:
        baseline = efficiency[baseline_config]
        report_lines.append(f"**Baseline (reference):** {baseline_config}\n")
        report_lines.append(f"- Latency: {baseline['latency_mean_sec']:.3f}s\n")
        report_lines.append(f"- Estimated parameters: {baseline['estimated_params_B']:.2f}B\n\n")
        
        report_lines.append("**Comparative Analysis:**\n\n")
        for config_name in sorted(efficiency.keys()):
            if config_name == baseline_config:
                continue
            m = efficiency[config_name]
            latency_delta = ((m["latency_mean_sec"] - baseline["latency_mean_sec"]) / baseline["latency_mean_sec"] * 100) if baseline["latency_mean_sec"] > 0 else 0
            param_delta = ((m["estimated_params_B"] - baseline["estimated_params_B"]) / baseline["estimated_params_B"] * 100) if baseline["estimated_params_B"] > 0 else 0
            
            report_lines.append(f"- **{config_name}**\n")
            report_lines.append(f"  - Latency change: {latency_delta:+.1f}%\n")
            report_lines.append(f"  - Parameter overhead: {param_delta:+.1f}%\n")
            report_lines.append(f"  - Description: {m['description']}\n\n")
    
    report_lines.append("## Interpretation\n\n")
    report_lines.append("- **Constrained decoding** adds structure enforcement at ~5% parameter overhead.\n")
    report_lines.append("- **Speculative decoding** reduces latency with ~10% parameter overhead (draft model).\n")
    report_lines.append("- **Combined** enables both safety and speed, totaling ~15% overhead.\n")
    report_lines.append("- **Efficiency metric** (throughput per billion params) shows cost-normalized throughput.\n\n")
    
    report_path = output_dir / "tradespace_report.md"
    with report_path.open("w") as f:
        f.writelines(report_lines)
    print(f"Wrote: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter vs. latency tradespace for RAG + decoding benchmarks"
    )
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=Path("runs/rag_100_jobs/metrics/benchmark_summary.json"),
        help="Path to benchmark_summary.json (default: runs/rag_100_jobs/metrics/benchmark_summary.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/rag_100_jobs/metrics"),
        help="Output directory for tradespace analysis (default: runs/rag_100_jobs/metrics)",
    )
    args = parser.parse_args()
    
    if not args.benchmark_summary.exists():
        print(f"Error: benchmark summary not found at {args.benchmark_summary}")
        print("Please ensure your benchmark has completed and produced benchmark_summary.json")
        return 1
    
    generate_tradespace_summary(args.benchmark_summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
