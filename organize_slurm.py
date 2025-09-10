#!/usr/bin/env python3
"""
SLURM File Organizer
Organizes SLURM output files by UV usage, date, and status.
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

def extract_job_info(filename):
    """Extract job information from SLURM file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract date from prolog
        date_match = re.search(r'Begin Slurm Prolog:\s+(\w+-\d+-\d+\s+\d+:\d+:\d+)', content)
        if date_match:
            date_str = date_match.group(1)
            # Parse date (e.g., "Sep-02-2025 13:24:54")
            try:
                parsed_date = datetime.strptime(date_str, '%b-%d-%Y %H:%M:%S')
                date_folder = parsed_date.strftime('%Y-%m-%d')
            except:
                date_folder = 'unknown_date'
        else:
            date_folder = 'unknown_date'
        
        # Determine status
        has_epilog = 'Begin Slurm Epilog' in content
        has_traceback = 'Traceback' in content
        has_errors = bool(re.search(r'(Error|Exception|Failed)', content, re.IGNORECASE))
        
        if has_traceback or has_errors:
            status = 'failed'
        elif has_epilog:
            status = 'completed'
        else:
            status = 'unknown'
        
        return {
            'date_folder': date_folder,
            'status': status
        }
    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return {
            'date_folder': 'unknown_date',
            'status': 'unknown'
        }

def organize_slurm_files(dry_run=True):
    """Organize SLURM files into directory structure."""
    
    # Find all SLURM files
    slurm_files = [f for f in os.listdir('.') if f.startswith('slurm-') and f.endswith('.out')]
    
    if not slurm_files:
        print("No SLURM files found!")
        return
    
    print(f"Found {len(slurm_files)} SLURM files")
    print("=" * 50)
    
    # Create directory structure
    base_dir = Path('slurm_logs')
    
    for filename in slurm_files:
        info = extract_job_info(filename)
        
        # Create directory path: slurm_logs/date/status/
        target_dir = base_dir / info['date_folder'] / info['status']
        
        if dry_run:
            print(f"Would move: {filename}")
            print(f"  → {target_dir / filename}")
            print(f"  Date: {info['date_folder']}, Status: {info['status']}")
            print()
        else:
            # Actually move the file
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(filename, target_dir / filename)
            print(f"Moved: {filename} → {target_dir / filename}")
    
    if dry_run:
        print("=" * 50)
        print("This was a DRY RUN. To actually organize files, run:")
        print("python organize_slurm.py --execute")

def create_summary():
    """Create a summary of all SLURM files."""
    slurm_files = [f for f in os.listdir('.') if f.startswith('slurm-') and f.endswith('.out')]
    
    print("SLURM Files Summary")
    print("=" * 50)
    
    failed_count = 0
    completed_count = 0
    
    for filename in sorted(slurm_files):
        info = extract_job_info(filename)
        
        status_icon = '❌' if info['status'] == 'failed' else '✅'
        
        print(f"{status_icon} {filename:<20} | {info['date_folder']:<12} | {info['status']}")
        
        if info['status'] == 'failed':
            failed_count += 1
        else:
            completed_count += 1
    
    print("=" * 50)
    print(f"Total files: {len(slurm_files)}")
    print(f"Failed: {failed_count}, Completed: {completed_count}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        print("Organizing SLURM files...")
        organize_slurm_files(dry_run=False)
    elif len(sys.argv) > 1 and sys.argv[1] == '--summary':
        create_summary()
    else:
        print("SLURM File Organizer")
        print("=" * 30)
        create_summary()
        print("\n" + "=" * 30)
        print("DRY RUN - Organization Preview:")
        organize_slurm_files(dry_run=True)
