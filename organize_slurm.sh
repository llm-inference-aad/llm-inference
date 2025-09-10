#!/bin/bash
# Simple SLURM file organizer by date and status

echo "SLURM File Organizer"
echo "==================="

# Create base directory
mkdir -p slurm_logs

# Process each SLURM file
for file in slurm-*.out; do
    if [ ! -f "$file" ]; then
        echo "No SLURM files found!"
        exit 1
    fi
    
    echo "Processing: $file"
    
    # Extract date (from prolog line)
    date_str=$(grep "Begin Slurm Prolog:" "$file" | head -1 | sed 's/.*Begin Slurm Prolog: //' | cut -d' ' -f1)
    # Convert Sep-02-2025 to 2025-09-02
    date_folder=$(date -d "$date_str" +%Y-%m-%d 2>/dev/null || echo "unknown_date")
    
    # Determine status
    if grep -q "Traceback\|Error\|Exception" "$file"; then
        status="failed"
    else
        status="completed"
    fi
    
    # Create target directory
    target_dir="slurm_logs/$date_folder/$status"
    mkdir -p "$target_dir"
    
    # Move file
    mv "$file" "$target_dir/"
    echo "  → Moved to: $target_dir/$file"
done

echo ""
echo "Organization complete!"
echo "Directory structure:"
tree slurm_logs 2>/dev/null || find slurm_logs -type f | sort
