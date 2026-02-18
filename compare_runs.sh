#!/bin/bash
# Compare two runs (with and without vLLM)

set -e

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  $0 <run_dir1> [run_dir2]"
    echo ""
    echo "Generate evaluation report for one or compare two runs."
    echo ""
    echo "Examples:"
    echo "  $0 runs/2024-01-15_vllm_test"
    echo "  $0 runs/2024-01-15_vllm_test runs/2024-01-15_no_vllm"
    echo ""
    echo "Available runs:"
    ls -lt runs/ 2>/dev/null | grep '^d' | head -5 | awk '{print "  - runs/"$9}'
    exit 0
fi

RUN1=$1
RUN2=${2:-}

# Generate report for first run
echo "Generating report for: $RUN1"
uv run python generate_evaluation_report.py "$RUN1"

if [ -n "$RUN2" ]; then
    echo ""
    echo "Generating report for: $RUN2"
    uv run python generate_evaluation_report.py "$RUN2"
    
    echo ""
    echo "================================================================================"
    echo "                            SIDE-BY-SIDE COMPARISON"
    echo "================================================================================"
    
    # Extract key metrics from both reports
    if [ -f "$RUN1/evaluation_report.json" ] && [ -f "$RUN2/evaluation_report.json" ]; then
        python3 << EOF
import json

with open('$RUN1/evaluation_report.json', 'r') as f:
    run1 = json.load(f)
with open('$RUN2/evaluation_report.json', 'r') as f:
    run2 = json.load(f)

s1 = run1.get('summary', {})
s2 = run2.get('summary', {})

def compare_metric(name, key, lower_is_better=False, unit='', format_str='.4f'):
    val1 = s1.get(key, 0)
    val2 = s2.get(key, 0)
    
    if val1 == 0 or val2 == 0:
        diff_pct = 0
        diff_symbol = ''
    else:
        diff_pct = ((val2 - val1) / val1) * 100
        if lower_is_better:
            diff_symbol = '✅' if diff_pct < 0 else '❌'
        else:
            diff_symbol = '✅' if diff_pct > 0 else '❌'
    
    format_spec = f'{{:{format_str}}}'
    val1_str = format_spec.format(val1)
    val2_str = format_spec.format(val2)
    diff_str = f'{diff_pct:+.1f}%' if diff_pct != 0 else 'same'
    
    print(f'{name:30s} {val1_str:>12s}{unit:4s} {val2_str:>12s}{unit:4s} {diff_str:>10s} {diff_symbol}')

print(f'Run 1: {run1["run_id"]}')
print(f'Run 2: {run2["run_id"]}')
print()
print(f'{"Metric":<30s} {"Run 1":>16s} {"Run 2":>16s} {"Change":>10s}')
print('-' * 80)
print(f'Backend: {s1.get("backend", "Unknown"):>46s} {s2.get("backend", "Unknown"):>16s}')
print()
compare_metric('Total Requests', 'total_requests', format_str='.0f')
print()
compare_metric('Avg Latency', 'avg_latency_sec', lower_is_better=True, unit='sec')
compare_metric('Min Latency', 'min_latency_sec', lower_is_better=True, unit='sec')
compare_metric('Max Latency', 'max_latency_sec', lower_is_better=True, unit='sec')
print()
compare_metric('Throughput (req/sec)', 'throughput_req_per_sec', unit='req')
compare_metric('Throughput (req/min)', 'throughput_req_per_min', unit='req', format_str='.2f')
print()
compare_metric('GPU Memory Used', 'gpu_memory_used_mb', unit='MB', format_str='.2f')
compare_metric('GPU Memory Util', 'gpu_memory_util_percent', unit='%', format_str='.2f')
print()
compare_metric('Evaluation Score', 'avg_evaluation_score')
print('=' * 80)
EOF
    fi
fi

echo ""
echo "Reports saved in:"
echo "  - $RUN1/evaluation_report.txt"
echo "  - $RUN1/evaluation_report.json"
if [ -n "$RUN2" ]; then
    echo "  - $RUN2/evaluation_report.txt"
    echo "  - $RUN2/evaluation_report.json"
fi
