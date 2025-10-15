# Plots Directory

This directory contains generated visualization outputs from analysis scripts.

## Generated Files

- `latency_vs_accuracy_{run_id}.png` - Latency vs accuracy scatter plots
- `latency_vs_goodput_{run_id}.png` - Goodput over generations plots

## Usage

Plots are automatically saved here when running the analysis scripts:

```bash
# Plot latency vs accuracy (defaults to latest run)
python scripts/plot_latency_vs_accuracy.py

# Plot goodput over generations (defaults to latest run)
python scripts/plot_latency_vs_goodput.py

# Specify a particular run
python scripts/plot_latency_vs_accuracy.py --run-id auto_20251014_191652

# Custom output location
python scripts/plot_latency_vs_accuracy.py --output my_custom_plot.png
```

## Organization

- Plots are organized by run_id in their filenames
- Old plots are not automatically deleted (manual cleanup if needed)
- All plots use high DPI (300) for publication quality
