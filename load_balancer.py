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
    
    def get_all_servers(self) -> List[Dict]:
        """Get information about all servers"""
        with self.server_lock:
            return [server.to_dict() for server in self.servers.values()]


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
    Route inference request to least loaded server.
    
    Implements retry logic if server fails.
    """
    last_error = None
    attempts = 0
    
    while attempts <= MAX_RETRIES:
        try:
            # Select least loaded server
            server = load_balancer.get_least_loaded_server()
            
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
                    
                    print(f"[Route] Forwarding request (job_id={request.job_id}, gene_id={request.gene_id}) to {server.hostname}:{server.port}")
                    
                    async with session.post(
                        f"{server.url}/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=None)  # No timeout for long inference
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            result = await response.json()
                            server.decrement_active(response_time, success=True)
                            print(f"[Route] Request completed in {response_time:.2f}s on {server.hostname}:{server.port}")
                            return result
                        else:
                            error_text = await response.text()
                            server.decrement_active(response_time, success=False)
                            last_error = f"Server {server.hostname}:{server.port} returned {response.status}: {error_text}"
                            print(f"[Error] {last_error}")
                            
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


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "LLM Load Balancer is running!",
        "total_servers": len(load_balancer.servers),
        "healthy_servers": sum(1 for s in load_balancer.servers.values() if s.is_healthy)
    }


@app.on_event("startup")
async def startup_event():
    """Initialize load balancer on startup"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] ===== LOAD BALANCER STARTUP =====")
    print(f"[{timestamp}] Registry file: {REGISTRY_FILE}")
    print(f"[{timestamp}] Registry reload interval: {REGISTRY_RELOAD_INTERVAL}s")
    print(f"[{timestamp}] Health check interval: {HEALTH_CHECK_INTERVAL}s")
    
    # Initial load of servers
    load_balancer.load_servers_from_registry()
    
    # Start background tasks
    load_balancer.registry_reload_task = asyncio.create_task(load_balancer.reload_registry_loop())
    load_balancer.health_check_task = asyncio.create_task(load_balancer.health_check_loop())
    
    print(f"[{timestamp}] Load balancer started successfully")
    print(f"[{timestamp}] Available endpoints: /generate (POST), /servers (GET), / (GET)")


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


