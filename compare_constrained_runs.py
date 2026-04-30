#!/usr/bin/env python3
"""
Compare LLM Guided Evolution with and without constrained decoding.

This script:
1. Creates two separate run directories
2. Runs run_improved.py without constraints (baseline)
3. Runs run_improved.py with JSON constraints
4. Generates evaluation reports for both
5. Compares the results

Usage:
    python compare_constrained_runs.py [num_generations]
    
Example:
    python compare_constrained_runs.py 3
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(title):
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{title.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


def print_section(title):
    """Print a formatted section."""
    print(f"\n{BOLD}{YELLOW}>>> {title}{RESET}")


def print_success(msg):
    """Print success message."""
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg):
    """Print error message."""
    print(f"{RED}❌ {msg}{RESET}")


def print_info(msg):
    """Print info message."""
    print(f"{BLUE}ℹ️  {msg}{RESET}")


def create_run_directory(name):
    """Create a new run directory."""
    print_section(f"Creating run directory: {name}")
    
    try:
        result = subprocess.run(
            f"AUTOMATED_CALL=true bash scripts/create_run.sh '{name}'",
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print_error(f"Failed to create run directory: {result.stderr}")
            return None
        
        run_id = result.stdout.strip().split('\n')[-1]
        print_success(f"Created run: {run_id}")
        return run_id
    except Exception as e:
        print_error(f"Exception creating run: {e}")
        return None


def wait_for_jobs_completion(run_dir, timeout=7200):
    """Wait for all SLURM jobs associated with a run to complete."""
    print_section(f"Waiting for all jobs to complete...")
    print_info(f"Timeout: {timeout}s ({timeout/60:.0f} minutes)")
    
    start_time = time.time()
    check_interval = 30  # Check every 30 seconds
    
    while (time.time() - start_time) < timeout:
        # Check if there are any running SLURM jobs
        try:
            result = subprocess.run(
                "squeue -u $USER -h -o '%j'",
                shell=True,
                capture_output=True,
                text=True
            )
            
            running_jobs = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Check if any jobs are related to LLM work (eval, llm, etc.)
            active_llm_jobs = [job for job in running_jobs if job and ('eval' in job.lower() or 'llm' in job.lower())]
            
            if not active_llm_jobs:
                print_success(f"All jobs completed after {time.time() - start_time:.0f}s")
                return True
            
            elapsed = time.time() - start_time
            print_info(f"Waiting... {len(active_llm_jobs)} active jobs | Elapsed: {elapsed:.0f}s / {timeout}s")
            time.sleep(check_interval)
            
        except Exception as e:
            print_error(f"Error checking job status: {e}")
            time.sleep(check_interval)
    
    print_error(f"Timeout reached after {timeout}s")
    return False


def run_llm_ge(run_id, checkpoint_dir, use_constraints=False):
    """Run LLM Guided Evolution."""
    constraint_label = "WITH" if use_constraints else "WITHOUT"
    print_section(f"Running LLM GE {constraint_label} constrained decoding")
    print_info(f"Run ID: {run_id}")
    print_info(f"Checkpoint dir: {checkpoint_dir}")
    
    env = os.environ.copy()
    env["USE_VLLM"] = "true"
    env["CONSTRAINT_TYPE"] = "json" if use_constraints else ""
    env["ENABLE_JSON_CONSTRAINTS"] = "true" if use_constraints else "false"
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            f"uv run python run_improved.py '{checkpoint_dir}'",
            shell=True,
            capture_output=False,
            env=env,
            cwd=os.getcwd()
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"LLM GE job submission completed in {elapsed:.1f}s")
            return True
        else:
            print_error(f"LLM GE failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print_error(f"Exception running LLM GE: {e}")
        return False


def generate_evaluation_report(run_dir):
    """Generate evaluation report for a run."""
    print_section(f"Generating evaluation report: {run_dir}")
    
    try:
        result = subprocess.run(
            f"uv run python generate_evaluation_report.py '{run_dir}'",
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print_success(f"Report generated for {run_dir}")
            return True
        else:
            print_error(f"Failed to generate report: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Exception generating report: {e}")
        return False


def compare_reports(baseline_dir, constrained_dir):
    """Compare evaluation reports from both runs."""
    print_header("COMPARISON RESULTS")
    
    baseline_report = Path(baseline_dir) / "evaluation_report.json"
    constrained_report = Path(constrained_dir) / "evaluation_report.json"
    
    if not baseline_report.exists():
        print_error(f"Baseline report not found: {baseline_report}")
        return
    
    if not constrained_report.exists():
        print_error(f"Constrained report not found: {constrained_report}")
        return
    
    with open(baseline_report) as f:
        baseline = json.load(f)
    
    with open(constrained_report) as f:
        constrained = json.load(f)
    
    baseline_summary = baseline.get("summary", {})
    constrained_summary = constrained.get("summary", {})
    
    if not baseline_summary or not constrained_summary:
        print_error("Could not find summary data in reports")
        return
    
    # Compare metrics
    metrics_to_compare = [
        ("Total Requests", "total_requests", None),
        ("Avg Processing Latency (primary, sec)", "avg_service_latency_sec", ".4f"),
        ("Avg Latency (E2E, sec)", "avg_latency_sec", ".4f"),
        ("Min Latency (sec)", "min_latency_sec", ".4f"),
        ("Max Latency (sec)", "max_latency_sec", ".4f"),
        ("Throughput (req/sec)", "throughput_req_per_sec", ".4f"),
        ("Throughput (req/min)", "throughput_req_per_min", ".2f"),
        ("GPU Memory Used (MB)", "gpu_memory_used_mb", ".2f"),
        ("GPU Memory Utilized (%)", "gpu_memory_util_percent", ".2f"),
        ("Avg Evaluation Score", "avg_evaluation_score", ".4f"),
    ]
    
    print(f"\n{BOLD}{'Metric':<40} {'Baseline':<20} {'Constrained':<20} {'Diff (%)':<15}{RESET}")
    print("-" * 95)
    
    for metric_name, key, fmt in metrics_to_compare:
        baseline_val = baseline_summary.get(key, 0)
        constrained_val = constrained_summary.get(key, 0)
        
        if baseline_val == 0:
            diff_pct = "N/A"
        else:
            diff_pct = f"{((constrained_val - baseline_val) / baseline_val * 100):+.1f}%"
        
        if fmt:
            baseline_str = f"{baseline_val:{fmt}}"
            constrained_str = f"{constrained_val:{fmt}}"
        else:
            baseline_str = str(baseline_val)
            constrained_str = str(constrained_val)
        
        print(f"{metric_name:<40} {baseline_str:<20} {constrained_str:<20} {diff_pct:<15}")
    
    # Calculate latency overhead using service latency (excludes queue time).
    baseline_latency = baseline_summary.get("avg_service_latency_sec", 0)
    constrained_latency = constrained_summary.get("avg_service_latency_sec", 0)
    
    if baseline_latency > 0:
        overhead = ((constrained_latency - baseline_latency) / baseline_latency) * 100
        print(f"\n{BOLD}Processing Latency Overhead (excl. queue): {overhead:+.1f}%{RESET}")
    
    # Generate comparison report
    comparison = {
        "baseline": baseline,
        "constrained": constrained,
        "comparison_timestamp": datetime.now().isoformat(),
        "latency_overhead_percent": overhead if baseline_latency > 0 else None,
        "throughput_improvement_percent": (
            (constrained_summary.get("throughput_req_per_sec", 0) - 
             baseline_summary.get("throughput_req_per_sec", 0)) / 
            baseline_summary.get("throughput_req_per_sec", 1) * 100
        ) if baseline_summary.get("throughput_req_per_sec", 0) > 0 else None,
    }
    
    # Save comparison
    comparison_file = Path("comparison_results.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print_success(f"Comparison saved to {comparison_file}")


def cleanup_previous_runs():
    """Clean up comparison run directories, results, and SLURM files from previous execution."""
    import shutil
    print_section("Cleaning up previous runs")
    
    cleanup_items = []
    
    # Remove old comparison directories from runs/
    runs_dir = Path("runs")
    if runs_dir.exists():
        for pattern in ["comparison_baseline_no_constraints_*", "comparison_with_json_constraints_*"]:
            for run_dir in runs_dir.glob(pattern):
                try:
                    shutil.rmtree(run_dir)
                    cleanup_items.append(run_dir.name)
                except Exception as e:
                    print_error(f"Failed to remove {run_dir}: {e}")
    
    # Remove comparison results file
    comparison_file = Path("comparison_results.json")
    if comparison_file.exists():
        try:
            comparison_file.unlink()
            cleanup_items.append("comparison_results.json")
        except Exception as e:
            print_error(f"Failed to remove {comparison_file}: {e}")
    
    # Remove all SLURM result files
    slurm_results_dir = Path("slurm-results")
    if slurm_results_dir.exists():
        slurm_count = 0
        for slurm_file in slurm_results_dir.glob("*"):
            try:
                if slurm_file.is_file():
                    slurm_file.unlink()
                    slurm_count += 1
            except Exception as e:
                print_error(f"Failed to remove {slurm_file}: {e}")
        
        if slurm_count > 0:
            cleanup_items.append(f"slurm-results/ ({slurm_count} files)")
    
    if cleanup_items:
        print_success(f"Cleaned up {len(cleanup_items)} item(s):")
        for item in cleanup_items:
            print_info(f"  - {item}")
    else:
        print_info("No previous runs to clean up")


def main():
    print_header("LLM GUIDED EVOLUTION - CONSTRAINED DECODING COMPARISON")
    
    # Clean up previous runs
    cleanup_previous_runs()
    
    # Parse arguments
    num_generations = 1
    if len(sys.argv) > 1:
        try:
            num_generations = int(sys.argv[1])
        except ValueError:
            print_error(f"Invalid argument: {sys.argv[1]}")
            print("Usage: python compare_constrained_runs.py [num_generations]")
            sys.exit(1)
    
    print_info(f"Generations per run: {num_generations}")
    
    # Create run directories
    baseline_id = create_run_directory("comparison_baseline_no_constraints")
    if not baseline_id:
        print_error("Failed to create baseline run directory")
        sys.exit(1)
    
    constrained_id = create_run_directory("comparison_with_json_constraints")
    if not constrained_id:
        print_error("Failed to create constrained run directory")
        sys.exit(1)
    
    baseline_dir = Path("runs") / baseline_id
    constrained_dir = Path("runs") / constrained_id
    
    # Run LLM GE without constraints
    print_header("PHASE 1: BASELINE RUN (NO CONSTRAINTS)")
    if not run_llm_ge(baseline_id, str(baseline_dir / "checkpoints"), use_constraints=False):
        print_error("Baseline run failed")
        sys.exit(1)
    
    # Wait for all baseline jobs to complete
    if not wait_for_jobs_completion(str(baseline_dir)):
        print_error("Baseline jobs did not complete in time")
        sys.exit(1)
    
    # Generate report for baseline
    if not generate_evaluation_report(str(baseline_dir)):
        print_error("Failed to generate baseline report")
        sys.exit(1)
    
    # Run LLM GE with constraints
    print_header("PHASE 2: CONSTRAINED RUN (WITH JSON CONSTRAINTS)")
    if not run_llm_ge(constrained_id, str(constrained_dir / "checkpoints"), use_constraints=True):
        print_error("Constrained run failed")
        sys.exit(1)
    
    # Wait for all constrained jobs to complete
    if not wait_for_jobs_completion(str(constrained_dir)):
        print_error("Constrained jobs did not complete in time")
        sys.exit(1)
    
    # Generate report for constrained
    if not generate_evaluation_report(str(constrained_dir)):
        print_error("Failed to generate constrained report")
        sys.exit(1)
    
    # Compare results
    compare_reports(str(baseline_dir), str(constrained_dir))
    
    print_header("COMPARISON COMPLETE")
    print_success("Both runs completed successfully!")
    print_info(f"Baseline run: {baseline_dir}")
    print_info(f"Constrained run: {constrained_dir}")


if __name__ == "__main__":
    main()
