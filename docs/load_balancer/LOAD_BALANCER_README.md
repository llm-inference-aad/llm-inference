# Multi-Server Load Balancer for LLM Inference

This load balancer system allows you to run multiple LLM inference servers in parallel and automatically distributes requests across them based on server load.

## Architecture

The system consists of three main components:

1. **Load Balancer** (`load_balancer.py`): Central FastAPI server that routes requests to the least busy inference server
2. **Inference Servers** (`server.py`): Multiple GPU-backed servers running the LLM model
3. **Server Registry** (`servers.json`): Shared file where servers register their hostname and port

### How It Works

```
Client Requests
      ↓
Load Balancer (port 9000)
      ↓
Routes to least loaded server
      ↓
┌─────────────┬─────────────┬─────────────┐
│  Server 1   │  Server 2   │  Server 3   │
│  (port 8000)│  (port 8001)│  (port 8002)│
└─────────────┴─────────────┴─────────────┘
```

## Quick Start

### 1. Configure Environment Variables

Add these variables to your `.env` file (or copy from `env.example`):

```bash
# Enable load balancer mode
USE_LOAD_BALANCER=true

# Load balancer configuration
LOAD_BALANCER_PORT=9000
LOADBALANCER_LOG_FILE=/path/to/your/loadbalancer.log
SERVER_REGISTRY_FILE=/path/to/your/servers.json

# Server cluster configuration
NUM_SERVERS=3
SERVER_BASE_PORT=8000
```

### 2. Start the Cluster

Launch the load balancer and multiple inference servers:

```bash
# Start cluster with 3 servers (default)
./scripts/start_cluster.sh

# Or specify number of servers
./scripts/start_cluster.sh 5
```

This will:
- Submit a load balancer job
- Wait for load balancer to start
- Submit N inference server jobs with auto-incremented ports
- Display all job IDs for monitoring

### 3. Monitor Cluster Status

Check the status of your cluster:

```bash
# One-time status check
./scripts/monitor_cluster.sh

# Continuous monitoring (refreshes every 5 seconds)
./scripts/monitor_cluster.sh --continuous
```

The monitor shows:
- Load balancer status
- Number of healthy servers
- Active requests per server
- Request statistics (total, successful, failed)
- Average response times
- Load scores for each server

### 4. Submit Client Jobs

Your existing client jobs will automatically use the load balancer if `USE_LOAD_BALANCER=true`:

```bash
sbatch scripts/run.sh
sbatch scripts/run.sh
sbatch scripts/run.sh
```

Requests will be automatically distributed across all healthy servers.

## Load Balancing Algorithm

The system uses a **least-connections** strategy:

1. Filter out unhealthy servers (failed health checks)
2. Calculate load score for each healthy server:
   - Primary factor: Number of active requests (×100)
   - Secondary factor: Average response time (×0.1)
3. Route request to server with lowest load score
4. Track request metrics and update server status

### Health Checking

- Health checks run every 30 seconds
- Servers are marked unhealthy after 3 consecutive failures
- Unhealthy servers are excluded from routing
- Recovered servers are automatically re-enabled

### Retry Logic

- Failed requests are automatically retried on other servers
- Maximum 2 retries before returning error
- Helps handle transient failures gracefully

## Manual Management

### Check Server Pool

```bash
# Get detailed server information
curl http://$(cat loadbalancer.log):9000/servers | python3 -m json.tool
```

### View Server Registry

```bash
# See all registered servers
cat servers.json | python3 -m json.tool
```

### Check Job Status

```bash
# List your running jobs
squeue -u $USER

# View load balancer log
tail -f slurm-<LOADBALANCER_JOB_ID>.out

# View specific server log
tail -f slurm-<SERVER_JOB_ID>.out
```

### Stop the Cluster

```bash
# Cancel all jobs (use job IDs from start_cluster.sh output)
scancel <LOADBALANCER_JOB_ID> <SERVER_JOB_IDS...>

# Or cancel all your jobs
scancel -u $USER
```

## Backward Compatibility

The system is fully backward compatible with single-server mode:

```bash
# Single server mode (existing behavior)
USE_LOAD_BALANCER=false sbatch scripts/server.sh
USE_LOAD_BALANCER=false sbatch scripts/run.sh
```

When `USE_LOAD_BALANCER=false` (default), clients connect directly to the server specified in `hostname.log`.

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LOAD_BALANCER` | `false` | Enable/disable load balancer mode |
| `LOAD_BALANCER_PORT` | `9000` | Port for load balancer service |
| `LOADBALANCER_LOG_FILE` | `./loadbalancer.log` | File where load balancer writes its hostname |
| `SERVER_REGISTRY_FILE` | `./servers.json` | Shared registry file for server discovery |
| `NUM_SERVERS` | `3` | Number of servers to start with `start_cluster.sh` |
| `SERVER_BASE_PORT` | `8000` | Starting port for servers (auto-incremented) |

### Load Balancer Tuning

Edit `load_balancer.py` to adjust these parameters:

```python
REGISTRY_RELOAD_INTERVAL = 10  # How often to reload server registry (seconds)
HEALTH_CHECK_INTERVAL = 30     # How often to check server health (seconds)
HEALTH_CHECK_TIMEOUT = 5       # Timeout for health check requests (seconds)
MAX_HEALTH_CHECK_FAILURES = 3  # Failures before marking unhealthy
MAX_RETRIES = 2                # Request retry attempts
```

## Architecture Details

### Server Registration

1. Each server writes to `servers.json` on startup using file locking (flock)
2. Registry contains: hostname, port, registration timestamp
3. Load balancer reloads registry every 10 seconds
4. Servers can re-register on restart (updates timestamp)

### Request Flow

```
1. Client → submit_local_server() in src/llm_utils.py
2. Reads loadbalancer.log to find load balancer
3. Sends POST to http://loadbalancer:9000/generate
4. Load balancer selects least loaded server
5. Forwards request to http://server:8000/generate
6. Returns response to client
```

### Metrics Tracking

Each server tracks:
- Active requests (current)
- Total requests (lifetime)
- Successful/failed requests
- Last response time
- Average response time (exponential moving average)

## Troubleshooting

### Load balancer not starting

```bash
# Check if job is pending
squeue -u $USER

# View error messages
cat slurm-<JOB_ID>.out
```

### Servers not registering

```bash
# Check registry file
cat servers.json

# Verify SERVER_REGISTRY_FILE is set in server jobs
# Should be exported by start_cluster.sh
```

### No healthy servers

```bash
# Check server logs
tail -f slurm-<SERVER_JOB_ID>.out

# Manually test server health
curl http://<SERVER_HOSTNAME>:8000/
```

### Requests failing

```bash
# Check load balancer logs for routing errors
cat slurm-<LOADBALANCER_JOB_ID>.out | grep ERROR

# Test load balancer directly
curl -X POST http://$(cat loadbalancer.log):9000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "max_new_tokens": 10}'
```

## Performance Considerations

### Scaling

- Each server requires 1 GPU
- Load balancer runs on CPU only (minimal resources)
- Recommended: 3-5 servers for typical workloads
- Maximum tested: 10 servers

### Network

- All communication is HTTP (no SSL overhead)
- Load balancer adds ~1-5ms latency per request
- Servers should be on same network for best performance

### Resource Usage

```
Load Balancer: 4 CPU cores, 8GB RAM
Each Server:   1 GPU, 16 CPU cores, 160GB RAM
```

## API Reference

### Load Balancer Endpoints

#### POST /generate
Forward inference request to least loaded server.

**Request:**
```json
{
  "prompt": "Your prompt here",
  "max_new_tokens": 100,
  "top_p": 0.8,
  "temperature": 0.7,
  "job_id": "optional_job_identifier"
}
```

**Response:**
```json
{
  "generated_text": "Generated output",
  "response_time_sec": 2.5,
  "batch_size": 1,
  "queue_wait_time_sec": 0.1,
  "e2e_latency_sec": 2.6,
  "run_hash": "abc123",
  "evaluationScore": 0.95
}
```

#### GET /servers
Get status of all servers in the pool.

**Response:**
```json
{
  "total_servers": 3,
  "healthy_servers": 3,
  "total_active_requests": 5,
  "servers": [
    {
      "hostname": "node1.cluster.edu",
      "port": 8000,
      "active_requests": 2,
      "total_requests": 150,
      "successful_requests": 148,
      "failed_requests": 2,
      "avg_response_time": 3.2,
      "is_healthy": true,
      "load_score": 200.32
    }
  ]
}
```

#### GET /
Health check endpoint.

**Response:**
```json
{
  "message": "LLM Load Balancer is running!",
  "total_servers": 3,
  "healthy_servers": 3
}
```

## Examples

### Starting Different Cluster Sizes

```bash
# Small cluster (2 servers)
./scripts/start_cluster.sh 2

# Medium cluster (5 servers)
./scripts/start_cluster.sh 5

# Large cluster (10 servers)
./scripts/start_cluster.sh 10
```

### Mixed Workload

```bash
# Start cluster
./scripts/start_cluster.sh 5

# Submit multiple client jobs
for i in {1..20}; do
    sbatch scripts/run.sh
    sleep 1
done

# Monitor in real-time
./scripts/monitor_cluster.sh --continuous
```

### Testing Load Distribution

```bash
# Start cluster
./scripts/start_cluster.sh 3

# Send test requests
for i in {1..10}; do
    curl -X POST http://$(cat loadbalancer.log):9000/generate \
      -H "Content-Type: application/json" \
      -d "{\"prompt\": \"Test $i\", \"max_new_tokens\": 50}" &
done

# Check distribution
curl http://$(cat loadbalancer.log):9000/servers | python3 -m json.tool
```

## Contributing

When adding features to the load balancer:

1. Update `load_balancer.py` for routing logic
2. Update `server.sh` for server registration
3. Update `src/llm_utils.py` for client behavior
4. Test with various cluster sizes
5. Update this documentation

## License

Same as the main LLM Inference project.
