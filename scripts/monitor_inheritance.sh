#!/bin/bash
# Real-time Fitness Inheritance Monitor
# Usage: ./scripts/monitor_inheritance.sh [run_id]

set -euo pipefail

# Default to monitoring the latest run
RUN_ID=${1:-"latest"}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${REPO_ROOT}/runs"

# Determine the run directory
if [[ "$RUN_ID" == "latest" ]]; then
    RUN_DIR=$(ls -td "${RUNS_ROOT}"/*/ 2>/dev/null | head -1 || true)
    if [[ -z "$RUN_DIR" ]]; then
        echo "❌ No runs found in runs/ directory"
        exit 1
    fi
    RUN_ID=$(basename "$RUN_DIR")
else
    RUN_DIR="${RUNS_ROOT}/$RUN_ID"
    if [[ ! -d "$RUN_DIR" ]]; then
        echo "❌ Run directory not found: $RUN_DIR"
        exit 1
    fi
fi

RUN_LOG_DIR="${RUN_DIR}/logs"

echo "🔍 Monitoring fitness inheritance for run: $RUN_ID"
echo "📁 Run directory: $RUN_DIR"
echo "⏰ Started at: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Initialize counters
inheritance_count=0
fallback_count=0

# Function to check ancestry structure (called after Gen 0)
check_ancestry_structure() {
    local checkpoint_file="$RUN_DIR/checkpoints/checkpoint_gen_0.pkl"
    if [[ -f "$checkpoint_file" ]]; then
        echo ""
        echo "🔬 Checking ancestry structure in Gen 0 checkpoint..."
        
        python3 << EOF
import pickle
import sys

try:
    with open('$checkpoint_file', 'rb') as f:
        data = pickle.load(f)
    
    ancestry_data = data.get('GLOBAL_DATA_ANCESTRY', {})
    if not ancestry_data:
        print("⚠️  No ancestry data found in checkpoint")
        sys.exit(1)
    
    print(f"📊 Found {len(ancestry_data)} individuals in ancestry data")
    
    # Check first few individuals
    correct_count = 0
    incorrect_count = 0
    
    for gene_id, ancestry in list(ancestry_data.items())[:5]:
        genes = ancestry.get('GENES', [])
        if len(genes) > 0:
            parent = genes[0]
            if parent == 'network':
                correct_count += 1
                print(f"✅ {gene_id}: parent = 'network' (CORRECT)")
            elif parent == gene_id:
                incorrect_count += 1
                print(f"❌ {gene_id}: parent = '{gene_id}' (BUG STILL EXISTS!)")
            else:
                print(f"🤔 {gene_id}: parent = '{parent}' (UNEXPECTED)")
        else:
            print(f"⚠️  {gene_id}: No parent genes found")
    
    if incorrect_count > 0:
        print(f"🚨 CRITICAL: Ancestry bug still exists! {incorrect_count} individuals have self-reference")
        sys.exit(1)
    elif correct_count > 0:
        print(f"✅ GOOD: Ancestry fix is working! {correct_count} individuals correctly reference 'network'")
    else:
        print("⚠️  Unclear ancestry state - manual inspection recommended")

except Exception as e:
    print(f"❌ Error checking ancestry: {e}")
    sys.exit(1)
EOF
        
        if [[ $? -eq 0 ]]; then
            echo "✅ Ancestry structure verification PASSED"
        else
            echo "❌ Ancestry structure verification FAILED"
            return 1
        fi
    fi
}

# Function to process log lines
process_log_line() {
    local line="$1"
    local timestamp=$(date '+%H:%M:%S')
    
    # Check for fitness inheritance events
    if echo "$line" | grep -q "Inheriting fitness"; then
        inheritance_count=$((inheritance_count + 1))
        echo "[$timestamp] 🎯 INHERITANCE #$inheritance_count: $line"
        
        # Extract gene ID and parent fitness if possible
        if echo "$line" | grep -oE "Gene xXx[A-Za-z0-9]+ is a fallback clone" >/dev/null; then
            gene_id=$(echo "$line" | grep -oE "xXx[A-Za-z0-9]+")
            echo "         └── Gene: $gene_id"
        fi
    fi
    
    # Check for fallback detection
    if echo "$line" | grep -q "is a fallback, but parent.*not yet evaluated"; then
        fallback_count=$((fallback_count + 1))
        echo "[$timestamp] ⚠️  FALLBACK #$fallback_count: $line"
    fi
    
    # Check for generation completion
    if echo "$line" | grep -qE "Generation [0-9]+ completed"; then
        gen_num=$(echo "$line" | grep -oE "Generation [0-9]+" | grep -oE "[0-9]+")
        echo "[$timestamp] 📈 GENERATION $gen_num COMPLETED"
        
        # Check ancestry after Gen 0 completes
        if [[ "$gen_num" == "0" ]]; then
            sleep 2  # Wait for checkpoint to be written
            check_ancestry_structure || echo "⚠️  Ancestry check failed, continuing monitoring..."
        fi
    fi
    
    # Check for run completion
    if echo "$line" | grep -q "Job complete\|=== Job complete ==="; then
        echo "[$timestamp] 🏁 RUN COMPLETED"
        return 1  # Signal to stop monitoring
    fi
}

# Monitor the log file
LOG_PATTERN="$RUN_LOG_DIR/slurm-main-*.out"
LOG_FILE=$(ls $LOG_PATTERN 2>/dev/null | head -1)

if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    echo "📄 Monitoring existing log: $(basename "$LOG_FILE")"
    echo "   Use Ctrl+C to stop monitoring"
    echo ""
    
    # Process existing content first
    while IFS= read -r line; do
        process_log_line "$line"
    done < "$LOG_FILE"
    
    # Then follow new content
    tail -f "$LOG_FILE" | while IFS= read -r line; do
        if ! process_log_line "$line"; then
            break
        fi
    done
else
    echo "⏳ Waiting for log file to appear: $LOG_PATTERN"
    
    # Wait for log file to be created
    while ! ls $LOG_PATTERN >/dev/null 2>&1; do
        sleep 5
        echo "   Still waiting... ($(date '+%H:%M:%S'))"
    done
    
    LOG_FILE=$(ls $LOG_PATTERN | head -1)
    echo "📄 Log file found: $(basename "$LOG_FILE")"
    echo "🔄 Starting real-time monitoring..."
    echo ""
    
    tail -f "$LOG_FILE" | while IFS= read -r line; do
        if ! process_log_line "$line"; then
            break
        fi
    done
fi

# Final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 MONITORING SUMMARY"
echo "   Total inheritance events: $inheritance_count"
echo "   Total fallback events: $fallback_count"
echo "   Monitoring ended at: $(date)"

if [[ $inheritance_count -gt 0 ]]; then
    echo "✅ SUCCESS: Fitness inheritance is working! ($inheritance_count events detected)"
else
    echo "⚠️  No inheritance events detected - this could indicate:"
    echo "   • No fallbacks occurred (very good goodput)"
    echo "   • Ancestry bug still exists (check logs manually)"
    echo "   • Run still in progress (monitor longer)"
fi
