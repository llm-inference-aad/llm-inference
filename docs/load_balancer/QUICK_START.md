# Load Balancer Quick Start

## TL;DR

```bash
# 1. Enable load balancer in .env
echo "USE_LOAD_BALANCER=true" >> .env

# 2. Start cluster with N servers
./scripts/start_cluster.sh 5

# 3. Wait ~3 minutes for model loading

# 4. Monitor status
./scripts/monitor_cluster.sh

# 5. Submit jobs (they auto-use load balancer)
sbatch scripts/run.sh
```

## Essential Commands

### Start Cluster
```bash
./scripts/start_cluster.sh [NUM_SERVERS]    # Default: 3 servers
```

### Monitor
```bash
./scripts/monitor_cluster.sh                # One-time check
./scripts/monitor_cluster.sh --continuous   # Auto-refresh every 5s
```

### Check Status
```bash
squeue -u $USER                     # Job queue
curl http://$(cat loadbalancer.log):9000/servers | python3 -m json.tool
```

### Stop Cluster
```bash
scancel <JOB_IDS>                   # Use IDs from start_cluster.sh
scancel -u $USER                    # Cancel all your jobs
```

## File Locations

| File | Purpose |
|------|---------|
| `load_balancer.py` | Load balancer server code |
| `load_balancer.sh` | SBATCH script for load balancer |
| `start_cluster.sh` | Orchestrator to start everything |
| `monitor_cluster.sh` | Status monitoring script |
| `server.sh` | Server SBATCH script (modified) |
| `src/llm_utils.py` | Client code (modified) |
| `servers.json` | Server registry (auto-generated) |
| `loadbalancer.log` | Load balancer hostname (auto-generated) |

## Environment Variables

**Required:**
```bash
USE_LOAD_BALANCER=true
```

**Optional (with defaults):**
```bash
NUM_SERVERS=3
SERVER_BASE_PORT=8000
LOAD_BALANCER_PORT=9000
LOADBALANCER_LOG_FILE=./loadbalancer.log
SERVER_REGISTRY_FILE=./servers.json
```

## URLs

| Endpoint | Description |
|----------|-------------|
| `http://<lb-host>:9000/generate` | Submit inference request |
| `http://<lb-host>:9000/servers` | Get server pool status |
| `http://<lb-host>:9000/` | Health check |

## Typical Workflow

```bash
# Morning: Start cluster
./scripts/start_cluster.sh 5
# Output: Job IDs: LB=123456 Servers=123457,123458,123459,123460,123461

# Monitor startup
watch -n 5 './scripts/monitor_cluster.sh'

# When all servers healthy, submit work
for i in {1..20}; do sbatch scripts/run.sh; done

# Monitor progress
./scripts/monitor_cluster.sh --continuous

# Evening: Clean up
scancel 123456 123457 123458 123459 123460 123461
```

## Debugging

### Load balancer not starting
```bash
cat slurm-<LB_JOB_ID>.out
```

### Servers not registering
```bash
cat servers.json | python3 -m json.tool
tail -f slurm-<SERVER_JOB_ID>.out
```

### Requests failing
```bash
curl http://$(cat loadbalancer.log):9000/servers
# Check if servers are healthy
```

## Performance

- Model loading: ~2-3 minutes per server
- Load balancer overhead: ~1-5ms per request
- Recommended: 3-5 servers for typical workloads
- Maximum tested: 10 servers

## More Information

- Full documentation: [LOAD_BALANCER_README.md](LOAD_BALANCER_README.md)
- Deployment guide: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Environment setup: `env.example`
