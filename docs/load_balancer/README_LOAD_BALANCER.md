# Multi-Server Load Balancer - README Addition

## Add this section to README.md

Insert the following section after the Setup section in README.md:

---

### Multi-Server Load Balancer (NEW!)

We now support running multiple LLM inference servers with automatic load balancing! This allows you to:
- Scale horizontally across multiple GPUs
- Distribute requests to the least busy server
- Improve throughput for large workloads
- Automatic health checking and failover

**Quick Start:**
```bash
# 1. Enable load balancer in .env
echo "USE_LOAD_BALANCER=true" >> .env

# 2. Start cluster with 5 servers
./scripts/start_cluster.sh 5

# 3. Monitor status
./scripts/monitor_cluster.sh

# 4. Submit jobs (they automatically use the load balancer)
sbatch scripts/run.sh
```

**Documentation:**
- [Quick Start Guide](QUICK_START.md) - Get started in 2 minutes
- [Load Balancer README](LOAD_BALANCER_README.md) - Complete documentation
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Step-by-step deployment
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details

**Backward Compatibility:**
The system is fully backward compatible. Set `USE_LOAD_BALANCER=false` (default) to use the original single-server mode.

---
