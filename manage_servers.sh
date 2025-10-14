#!/bin/bash

# Server Management Utilities
# Usage: ./manage_servers.sh [COMMAND]
# Commands:
#   status     - Show current server status
#   cleanup    - Remove dead/unhealthy servers from registry
#   ports      - Show port usage
#   add N      - Add N servers (same as add_server.sh)

set -e

COMMAND=${1:-status}
NUM_SERVERS=${2:-1}

# Load environment variables
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$(pwd)}"
export SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE:-${LLM_INFERENCE_ROOT_DIR}/servers.json}"
export LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${LLM_INFERENCE_ROOT_DIR}/loadbalancer.log}"

case "$COMMAND" in
    "status")
        echo "===== Server Cluster Status ====="
        echo ""
        
        if [[ -f "$SERVER_REGISTRY_FILE" ]]; then
            echo "Registry file: $SERVER_REGISTRY_FILE"
            echo ""
            
            uv run python << EOF
import json
import sys
from datetime import datetime

try:
    with open('$SERVER_REGISTRY_FILE', 'r') as f:
        registry = json.load(f)
    
    servers = registry.get('servers', [])
    print(f"Total servers in registry: {len(servers)}")
    print()
    
    if servers:
        print("Server Details:")
        print("-" * 60)
        for i, server in enumerate(servers, 1):
            hostname = server.get('hostname', 'unknown')
            port = server.get('port', 'unknown')
            registered_at = server.get('registered_at', 'unknown')
            
            print(f"  Server {i}: {hostname}:{port}")
            print(f"    Registered: {registered_at}")
            print()
    else:
        print("No servers registered.")
        
except Exception as e:
    print(f"Error reading registry: {e}")
    sys.exit(1)
EOF
        else
            echo "Registry file not found: $SERVER_REGISTRY_FILE"
        fi
        
        echo ""
        echo "Load balancer status:"
        if [[ -f "$LOADBALANCER_LOG_FILE" ]]; then
            LB_HOST=$(cat "$LOADBALANCER_LOG_FILE")
            echo "  Load balancer: $LB_HOST:9000"
        else
            echo "  Load balancer: Not running"
        fi
        ;;
        
    "cleanup")
        echo "===== Cleaning Up Dead Servers ====="
        echo ""
        
        if [[ ! -f "$SERVER_REGISTRY_FILE" ]]; then
            echo "Registry file not found: $SERVER_REGISTRY_FILE"
            exit 1
        fi
        
        echo "Checking server health..."
        
        python3 << EOF
import json
import requests
import sys
from datetime import datetime, timedelta

try:
    with open('$SERVER_REGISTRY_FILE', 'r') as f:
        registry = json.load(f)
    
    servers = registry.get('servers', [])
    healthy_servers = []
    removed_count = 0
    
    for server in servers:
        hostname = server.get('hostname', '')
        port = server.get('port', 8000)
        url = f"http://{hostname}:{port}/"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                healthy_servers.append(server)
                print(f"✅ {hostname}:{port} - Healthy")
            else:
                print(f"❌ {hostname}:{port} - HTTP {response.status_code}")
                removed_count += 1
        except Exception as e:
            print(f"❌ {hostname}:{port} - Unreachable ({str(e)[:50]}...)")
            removed_count += 1
    
    # Update registry with only healthy servers
    registry['servers'] = healthy_servers
    
    with open('$SERVER_REGISTRY_FILE', 'w') as f:
        json.dump(registry, f, indent=2)
    
    print()
    print(f"Cleanup complete:")
    print(f"  Healthy servers: {len(healthy_servers)}")
    print(f"  Removed servers: {removed_count}")
    
except Exception as e:
    print(f"Error during cleanup: {e}")
    sys.exit(1)
EOF
        ;;
        
    "ports")
        echo "===== Port Usage ====="
        echo ""
        
        if [[ -f "$SERVER_REGISTRY_FILE" ]]; then
            uv run python << EOF
import json
import sys

try:
    with open('$SERVER_REGISTRY_FILE', 'r') as f:
        registry = json.load(f)
    
    servers = registry.get('servers', [])
    
    if servers:
        ports = [s.get('port', 8000) for s in servers]
        ports.sort()
        
        print("Used ports:")
        for port in ports:
            print(f"  {port}")
        
        print()
        print(f"Next available port: {max(ports) + 1}")
    else:
        print("No servers registered.")
        print("Next available port: 8000")
        
except Exception as e:
    print(f"Error reading registry: {e}")
    sys.exit(1)
EOF
        else
            echo "Registry file not found: $SERVER_REGISTRY_FILE"
            echo "Next available port: 8000"
        fi
        ;;
        
    "add")
        echo "Adding $NUM_SERVERS server(s)..."
        ./add_server.sh $NUM_SERVERS
        ;;
        
    *)
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  status     - Show current server status"
        echo "  cleanup    - Remove dead/unhealthy servers from registry"
        echo "  ports      - Show port usage"
        echo "  add N      - Add N servers"
        echo ""
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 cleanup"
        echo "  $0 ports"
        echo "  $0 add 2"
        ;;
esac


