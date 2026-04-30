"""
Load Balancer for LLM Inference Servers

This load balancer distributes inference requests across multiple servers
using a least-connections algorithm. It automatically discovers servers
from a shared registry file and performs health checks.
"""

import asyncio
import aiohttp
import time
import json
import os
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

app = FastAPI(title="LLM Load Balancer", version="1.0")

# Configuration
REGISTRY_FILE = os.getenv("SERVER_REGISTRY_FILE", "./servers.json")
REGISTRY_RELOAD_INTERVAL = 10  # seconds
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 5  # seconds
MAX_HEALTH_CHECK_FAILURES = 3  # consecutive failures before marking unhealthy
MAX_RETRIES = 2  # retry failed requests on other servers
PREFIX_HASH_CHARS = int(os.getenv("PREFIX_HASH_CHARS", "2048"))
RUN_ID = os.getenv("RUN_ID", "server-only")
METRICS_BASE_PATH = Path(os.getenv("METRICS_PATH", "./metrics"))
if RUN_ID == "server-only":
    GATEWAY_METRICS_DIR = METRICS_BASE_PATH / "data"
else:
    GATEWAY_METRICS_DIR = Path("./runs") / RUN_ID / "metrics"
GATEWAY_METRICS_DIR.mkdir(parents=True, exist_ok=True)
GATEWAY_METRICS_FILE = GATEWAY_METRICS_DIR / f"gateway-{int(time.time())}.json"


class ServerInfo:
    """Information about a single inference server"""

    def __init__(self, hostname: str, port: int, registered_at: str):
        self.hostname = hostname
        self.port = port
        self.url = f"http://{hostname}:{port}"
        self.registered_at = registered_at
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hit_routes = 0
        self.cache_miss_routes = 0
        self.last_response_time = 0.0
        self.avg_response_time = 0.0
        self.is_healthy = True
        self.last_health_check = None
        self.consecutive_failures = 0
        self._lock = threading.Lock()

    def increment_active(self):
        """Thread-safe increment of active requests"""
        with self._lock:
            self.active_requests += 1
            self.total_requests += 1

    def decrement_active(self, response_time: float, success: bool):
        """Thread-safe decrement of active requests and update metrics"""
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.last_response_time = response_time

            if success:
                self.successful_requests += 1
                # Update rolling average response time
                if self.avg_response_time == 0:
                    self.avg_response_time = response_time
                else:
                    # Exponential moving average
                    self.avg_response_time = 0.8 * self.avg_response_time + 0.2 * response_time
            else:
                self.failed_requests += 1

    def get_load_score(self) -> float:
        """Calculate load score for server selection (lower is better)"""
        # Primary factor: number of active requests
        score = self.active_requests * 100

        # Secondary factor: average response time (if available)
        if self.avg_response_time > 0:
            score += self.avg_response_time * 0.1

        return score

    def to_dict(self) -> Dict:
        """Convert server info to dictionary for JSON serialization"""
        return {
            "hostname": self.hostname,
            "port": self.port,
            "url": self.url,
            "registered_at": self.registered_at,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cache_hit_routes": self.cache_hit_routes,
            "cache_miss_routes": self.cache_miss_routes,
            "last_response_time": round(self.last_response_time, 4),
            "avg_response_time": round(self.avg_response_time, 4),
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "consecutive_failures": self.consecutive_failures,
            "load_score": round(self.get_load_score(), 2)
        }


class LoadBalancer:
    """Manages server pool and routes requests"""

    def __init__(self):
        self.servers: Dict[str, ServerInfo] = {}  # key: "hostname:port"
        self.server_lock = threading.Lock()
        self.registry_reload_task = None
        self.health_check_task = None
        self.last_registry_mtime = 0
        self.prefix_owners: Dict[str, str] = {}
        self.metrics_lock = threading.Lock()
        self.metrics = {
            "started_at": datetime.now().isoformat(),
            "prefix_hash_chars": PREFIX_HASH_CHARS,
            "total_requests": 0,
            "cache_hit_routes": 0,
            "cache_miss_routes": 0,
            "failed_requests": 0,
            "requests": []
        }
        self._write_metrics()

    def _get_server_key(self, hostname: str, port: int) -> str:
        """Generate unique key for server"""
        return f"{hostname}:{port}"

    def load_servers_from_registry(self):
        """Load servers from registry file"""
        try:
            registry_path = Path(REGISTRY_FILE)

            if not registry_path.exists():
                print(f"Registry file not found: {REGISTRY_FILE}")
                return

            # Check if file has been modified
            mtime = registry_path.stat().st_mtime
            if mtime == self.last_registry_mtime:
                return  # No changes

            self.last_registry_mtime = mtime

            with open(REGISTRY_FILE, 'r') as f:
                data = json.load(f)

            servers_data = data.get("servers", [])

            with self.server_lock:
                # Track current server keys
                new_server_keys = set()

                for server_data in servers_data:
                    hostname = server_data.get("hostname")
                    port = server_data.get("port")
                    registered_at = server_data.get("registered_at", "unknown")

                    if not hostname or not port:
                        continue

                    server_key = self._get_server_key(hostname, port)
                    new_server_keys.add(server_key)

                    # Add new server or update existing
                    if server_key not in self.servers:
                        self.servers[server_key] = ServerInfo(hostname, port, registered_at)
                        print(f"[Registry] Added new server: {hostname}:{port}")
                    else:
                        # Update registration time if changed
                        self.servers[server_key].registered_at = registered_at

                # Remove servers that are no longer in registry
                removed_keys = set(self.servers.keys()) - new_server_keys
                for key in removed_keys:
                    server = self.servers[key]
                    print(f"[Registry] Removed server: {server.hostname}:{server.port}")
                    del self.servers[key]

                # Drop prefix owners for workers that disappeared from the registry.
                self.prefix_owners = {
                    prefix_hash: owner_key
                    for prefix_hash, owner_key in self.prefix_owners.items()
                    if owner_key in self.servers
                }

        except json.JSONDecodeError as e:
            print(f"Error parsing registry file: {e}")
        except Exception as e:
            print(f"Error loading servers from registry: {e}")

    async def reload_registry_loop(self):
        """Periodically reload server registry"""
        while True:
            try:
                self.load_servers_from_registry()
                await asyncio.sleep(REGISTRY_RELOAD_INTERVAL)
            except Exception as e:
                print(f"Error in registry reload loop: {e}")
                await asyncio.sleep(REGISTRY_RELOAD_INTERVAL)

    async def health_check_server(self, server: ServerInfo) -> bool:
        """Check if a server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server.url}/",
                    timeout=aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Health check failed for {server.hostname}:{server.port} - {e}")
            return False

    async def health_check_loop(self):
        """Periodically check server health"""
        while True:
            try:
                with self.server_lock:
                    servers_to_check = list(self.servers.values())

                for server in servers_to_check:
                    is_healthy = await self.health_check_server(server)
                    server.last_health_check = datetime.now()

                    if is_healthy:
                        server.consecutive_failures = 0
                        if not server.is_healthy:
                            print(f"[Health] Server recovered: {server.hostname}:{server.port}")
                        server.is_healthy = True
                    else:
                        server.consecutive_failures += 1
                        if server.consecutive_failures >= MAX_HEALTH_CHECK_FAILURES:
                            if server.is_healthy:
                                print(f"[Health] Server marked unhealthy: {server.hostname}:{server.port}")
                            server.is_healthy = False

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                print(f"Error in health check loop: {e}")
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    def get_least_loaded_server(self) -> Optional[ServerInfo]:
        """Get the server with lowest load score"""
        with self.server_lock:
            if not self.servers:
                return None

            # Filter healthy servers
            healthy_servers = [s for s in self.servers.values() if s.is_healthy]

            if not healthy_servers:
                # If no healthy servers, try any server as fallback
                print("[Warning] No healthy servers available, using any available server")
                if self.servers:
                    return min(self.servers.values(), key=lambda s: s.get_load_score())
                return None

            # Return server with lowest load score
            return min(healthy_servers, key=lambda s: s.get_load_score())

    @staticmethod
    def extract_route_prefix(prompt: str) -> tuple[str, str]:
        """Extract the stable part of a prompt for cache-affinity routing.

        LLMGE mutation prompts vary mainly in the embedded code block. Routing
        on the raw first N chars is too selective because different code blocks
        create different hashes. For these prompts, use the instruction/template
        text before the code block marker so similar mutation operators route to
        the same worker. The full prompt is still sent to vLLM unchanged.
        """
        for marker in ("\nThe current code block:", "\n```python"):
            marker_index = prompt.find(marker)
            if marker_index > 0:
                route_prefix = prompt[:marker_index].strip()
                if route_prefix:
                    return route_prefix[:PREFIX_HASH_CHARS], "llmge_instruction_prefix"

        return prompt[:PREFIX_HASH_CHARS], "raw_prefix"

    @staticmethod
    def compute_prefix_hash(prompt: str) -> tuple[str, str, int]:
        """Compute a stable route hash used for cache-affinity routing."""
        route_prefix, strategy = LoadBalancer.extract_route_prefix(prompt)
        return hashlib.sha256(route_prefix.encode("utf-8")).hexdigest(), strategy, len(route_prefix)

    def get_cache_aware_server(self, prefix_hash: str) -> tuple[Optional[ServerInfo], bool, Optional[str]]:
        """Prefer the worker that most recently served this prefix, if healthy."""
        with self.server_lock:
            owner_key = self.prefix_owners.get(prefix_hash)
            if owner_key:
                owner = self.servers.get(owner_key)
                if owner and owner.is_healthy:
                    owner.cache_hit_routes += 1
                    return owner, True, owner_key

        server = self.get_least_loaded_server()
        if server:
            server.cache_miss_routes += 1
        return server, False, owner_key

    def record_prefix_owner(self, prefix_hash: str, server: ServerInfo) -> None:
        """Remember which worker has recently processed a prefix."""
        with self.server_lock:
            self.prefix_owners[prefix_hash] = self._get_server_key(server.hostname, server.port)

    def record_request_metrics(self, request_metrics: Dict) -> None:
        """Persist gateway routing metrics for later benchmark analysis."""
        with self.metrics_lock:
            self.metrics["total_requests"] += 1
            if request_metrics.get("cache_hit"):
                self.metrics["cache_hit_routes"] += 1
            else:
                self.metrics["cache_miss_routes"] += 1
            if not request_metrics.get("success"):
                self.metrics["failed_requests"] += 1
            self.metrics["requests"].append(request_metrics)
            self._write_metrics()

    def _write_metrics(self) -> None:
        try:
            with open(GATEWAY_METRICS_FILE, "w") as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"[Metrics] Failed to write gateway metrics: {e}")

    def get_all_servers(self) -> List[Dict]:
        """Get information about all servers"""
        with self.server_lock:
            return [server.to_dict() for server in self.servers.values()]

    def get_cache_stats(self) -> Dict:
        """Get cache-affinity routing stats."""
        with self.metrics_lock:
            total = max(1, self.metrics["total_requests"])
            return {
                "metrics_file": str(GATEWAY_METRICS_FILE),
                "prefix_hash_chars": PREFIX_HASH_CHARS,
                "tracked_prefixes": len(self.prefix_owners),
                "total_requests": self.metrics["total_requests"],
                "cache_hit_routes": self.metrics["cache_hit_routes"],
                "cache_miss_routes": self.metrics["cache_miss_routes"],
                "cache_hit_rate": self.metrics["cache_hit_routes"] / total,
                "failed_requests": self.metrics["failed_requests"],
            }


# Global load balancer instance
load_balancer = LoadBalancer()


class LLMRequest(BaseModel):
    """LLM inference request model"""
    prompt: str
    max_new_tokens: int = 100000
    top_p: float = 0.8
    temperature: float = 0.7
    job_id: str = "default"
    gene_id: str = None  # Identifier for the individual this request belongs to


@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Route inference request to a worker with cache affinity when possible.

    Implements retry logic if server fails.
    """
    last_error = None
    attempts = 0
    prefix_hash, prefix_strategy, route_prefix_chars = load_balancer.compute_prefix_hash(request.prompt)

    while attempts <= MAX_RETRIES:
        try:
            # Prefer the worker that has already processed this prefix so vLLM
            # can reuse its local prefix/KV cache. Fall back to least-loaded.
            server, cache_hit, prefix_owner = load_balancer.get_cache_aware_server(prefix_hash)

            if server is None:
                raise HTTPException(
                    status_code=503,
                    detail="No servers available. Please ensure servers are running and registered."
                )

            # Increment active requests
            server.increment_active()
            start_time = time.time()

            try:
                # Forward request to selected server
                async with aiohttp.ClientSession() as session:
                    payload = request.dict()
                    payload.update({
                        "prefix_hash": prefix_hash,
                        "cache_hit": cache_hit,
                        "cache_owner": prefix_owner,
                    })

                    print(
                        f"[Route] Forwarding request (job_id={request.job_id}, gene_id={request.gene_id}, "
                        f"cache_hit={cache_hit}) to {server.hostname}:{server.port}"
                    )

                    async with session.post(
                        f"{server.url}/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=None)  # No timeout for long inference
                    ) as response:
                        response_time = time.time() - start_time

                        if response.status == 200:
                            result = await response.json()
                            server.decrement_active(response_time, success=True)
                            load_balancer.record_prefix_owner(prefix_hash, server)
                            routing_metrics = {
                                "timestamp": datetime.now().isoformat(),
                                "job_id": request.job_id,
                                "gene_id": request.gene_id,
                                "prefix_hash": prefix_hash,
                                "prefix_strategy": prefix_strategy,
                                "route_prefix_chars": route_prefix_chars,
                                "cache_hit": cache_hit,
                                "prefix_owner_before": prefix_owner,
                                "selected_server": f"{server.hostname}:{server.port}",
                                "response_time_sec": round(response_time, 4),
                                "attempt": attempts + 1,
                                "success": True,
                                "worker_ttft_sec": result.get("ttft_sec"),
                                "worker_latency_sec": result.get("e2e_latency_sec", result.get("_latency_sec")),
                            }
                            load_balancer.record_request_metrics(routing_metrics)
                            result["gateway_cache_hit"] = cache_hit
                            result["gateway_prefix_hash"] = prefix_hash
                            result["gateway_selected_server"] = f"{server.hostname}:{server.port}"
                            result["gateway_response_time_sec"] = round(response_time, 4)
                            print(
                                f"[Route] Request completed in {response_time:.2f}s on "
                                f"{server.hostname}:{server.port} (cache_hit={cache_hit})"
                            )
                            return result
                        else:
                            error_text = await response.text()
                            server.decrement_active(response_time, success=False)
                            last_error = f"Server {server.hostname}:{server.port} returned {response.status}: {error_text}"
                            print(f"[Error] {last_error}")
                            load_balancer.record_request_metrics({
                                "timestamp": datetime.now().isoformat(),
                                "job_id": request.job_id,
                                "gene_id": request.gene_id,
                                "prefix_hash": prefix_hash,
                                "prefix_strategy": prefix_strategy,
                                "route_prefix_chars": route_prefix_chars,
                                "cache_hit": cache_hit,
                                "prefix_owner_before": prefix_owner,
                                "selected_server": f"{server.hostname}:{server.port}",
                                "response_time_sec": round(response_time, 4),
                                "attempt": attempts + 1,
                                "success": False,
                                "error": last_error,
                            })

                            # Mark server as potentially unhealthy
                            server.consecutive_failures += 1

                            attempts += 1
                            if attempts <= MAX_RETRIES:
                                print(f"[Retry] Attempt {attempts}/{MAX_RETRIES}")
                                continue
                            else:
                                raise HTTPException(status_code=response.status, detail=error_text)

            except aiohttp.ClientError as e:
                response_time = time.time() - start_time
                server.decrement_active(response_time, success=False)
                server.consecutive_failures += 1
                last_error = f"Connection error to {server.hostname}:{server.port}: {str(e)}"
                print(f"[Error] {last_error}")
                load_balancer.record_request_metrics({
                    "timestamp": datetime.now().isoformat(),
                    "job_id": request.job_id,
                    "gene_id": request.gene_id,
                    "prefix_hash": prefix_hash,
                    "prefix_strategy": prefix_strategy,
                    "route_prefix_chars": route_prefix_chars,
                    "cache_hit": cache_hit,
                    "prefix_owner_before": prefix_owner,
                    "selected_server": f"{server.hostname}:{server.port}",
                    "response_time_sec": round(response_time, 4),
                    "attempt": attempts + 1,
                    "success": False,
                    "error": last_error,
                })

                attempts += 1
                if attempts <= MAX_RETRIES:
                    print(f"[Retry] Attempt {attempts}/{MAX_RETRIES}")
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    raise HTTPException(status_code=503, detail=last_error)

            finally:
                # Ensure active requests is decremented even on exception
                pass

        except HTTPException:
            raise
        except Exception as e:
            last_error = str(e)
            print(f"[Error] Unexpected error: {last_error}")
            attempts += 1
            if attempts <= MAX_RETRIES:
                print(f"[Retry] Attempt {attempts}/{MAX_RETRIES}")
                continue
            else:
                raise HTTPException(status_code=500, detail=f"Load balancer error: {last_error}")

    # Should not reach here, but just in case
    raise HTTPException(status_code=500, detail=f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


@app.get("/servers")
async def list_servers():
    """Get status of all servers in the pool"""
    servers = load_balancer.get_all_servers()

    return {
        "total_servers": len(servers),
        "healthy_servers": sum(1 for s in servers if s["is_healthy"]),
        "total_active_requests": sum(s["active_requests"] for s in servers),
        "servers": servers
    }


@app.get("/cache-stats")
async def cache_stats():
    """Get prefix-cache-aware routing statistics."""
    return load_balancer.get_cache_stats()


@app.get("/metrics")
async def gateway_metrics():
    """Get gateway routing metrics and cache hit rate."""
    return {
        "cache_stats": load_balancer.get_cache_stats(),
        "servers": load_balancer.get_all_servers(),
    }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "LLM Load Balancer is running!",
        "total_servers": len(load_balancer.servers),
        "healthy_servers": sum(1 for s in load_balancer.servers.values() if s.is_healthy),
        "cache_stats": load_balancer.get_cache_stats(),
    }


@app.on_event("startup")
async def startup_event():
    """Initialize load balancer on startup"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] ===== LOAD BALANCER STARTUP =====")
    print(f"[{timestamp}] Registry file: {REGISTRY_FILE}")
    print(f"[{timestamp}] Registry reload interval: {REGISTRY_RELOAD_INTERVAL}s")
    print(f"[{timestamp}] Health check interval: {HEALTH_CHECK_INTERVAL}s")
    print(f"[{timestamp}] Prefix hash chars: {PREFIX_HASH_CHARS}")
    print(f"[{timestamp}] Gateway metrics file: {GATEWAY_METRICS_FILE}")

    # Initial load of servers
    load_balancer.load_servers_from_registry()

    # Start background tasks
    load_balancer.registry_reload_task = asyncio.create_task(load_balancer.reload_registry_loop())
    load_balancer.health_check_task = asyncio.create_task(load_balancer.health_check_loop())

    print(f"[{timestamp}] Load balancer started successfully")
    print(f"[{timestamp}] Available endpoints: /generate (POST), /servers (GET), /cache-stats (GET), /metrics (GET), / (GET)")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("Shutting down load balancer...")

    if load_balancer.registry_reload_task:
        load_balancer.registry_reload_task.cancel()

    if load_balancer.health_check_task:
        load_balancer.health_check_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
