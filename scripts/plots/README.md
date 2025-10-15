# Generated Plots

This directory contains visualization outputs from analysis scripts.

## Generated Files

- `latency_vs_accuracy_{run_id}.png` - Latency vs accuracy correlation plots
- `latency_vs_goodput_{run_id}.png` - Goodput over generations plots  
- `latency_analysis_{hash}.png` - E2E latency analysis from analyze_e2e_latency.py
- `gene_analysis_{hash}.png` - Gene-specific latency breakdown
- `run_comparison_{N}_runs.png` - Multi-run comparison charts

## Organization

- Plots are named by run_id or hash for easy identification
- High DPI (300) for publication quality
- Old plots are not automatically deleted (manual cleanup as needed)

## Usage

Plots are automatically generated when running analysis scripts from the parent directory.

See `../README.md` for script usage details.
