# Load Balancer Installation Guide

## For New Users

When you clone this repository, follow these steps to set up the load balancer:

### 1. Install Dependencies

```bash
# Install all dependencies including aiohttp
uv sync

# Verify aiohttp is installed
uv run python -c "import aiohttp; print('aiohttp version:', aiohttp.__version__)"
```

### 2. Set Up Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your paths
nano .env
```

### 3. Enable Load Balancer Mode

```bash
# Enable load balancer in .env
sed -i 's/USE_LOAD_BALANCER=false/USE_LOAD_BALANCER=true/' .env
```

### 4. Test Load Balancer

```bash
# Test load balancer startup
uv run python load_balancer.py
```

### 5. Start Cluster

```bash
# Start cluster with 3 servers
./scripts/start_cluster.sh 3

# Monitor cluster status
./scripts/monitor_cluster.sh
```

## Dependencies Added

The following dependency was added to `pyproject.toml`:

```toml
"aiohttp>=3.8.0"
```

This is required for the load balancer to make HTTP requests to inference servers.

## Troubleshooting

### If aiohttp is missing:

```bash
# Reinstall dependencies
uv sync

# Or install aiohttp directly
uv add aiohttp
```

### If load balancer fails to start:

```bash
# Check logs
tail -f slurm-<JOB_ID>.out

# Test manually
uv run python load_balancer.py
```

## Files to Commit

Make sure to commit these files:
- `pyproject.toml` (updated with aiohttp)
- `uv.lock` (updated with aiohttp)
- All load balancer files (load_balancer.py, *.sh, *.md)

## Backward Compatibility

The load balancer is opt-in. Existing users can continue using single-server mode by keeping `USE_LOAD_BALANCER=false` in their `.env` file.
