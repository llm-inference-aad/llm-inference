#!/bin/bash

# Monitor LLM Inference Cluster Status
# Usage: ./monitor_cluster.sh [OPTIONS]
#   -c, --continuous    Continuous monitoring (refresh every 5 seconds)
#   -h, --help          Show this help message

CONTINUOUS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--continuous)
            CONTINUOUS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  -c, --continuous    Continuous monitoring (refresh every 5 seconds)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load environment variables
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export LLM_INFERENCE_ROOT_DIR="${LLM_INFERENCE_ROOT_DIR:-$(pwd)}"
export LOADBALANCER_LOG_FILE="${LOADBALANCER_LOG_FILE:-${LLM_INFERENCE_ROOT_DIR}/loadbalancer.log}"
export SERVER_REGISTRY_FILE="${SERVER_REGISTRY_FILE:-${LLM_INFERENCE_ROOT_DIR}/servers.json}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-9000}"

# Function to display cluster status
show_status() {
    clear
    echo "======================================"
    echo "  LLM Inference Cluster Monitor"
    echo "======================================"
    echo ""
    date
    echo ""
    
    # Check if load balancer is running
    if [[ ! -f "$LOADBALANCER_LOG_FILE" ]]; then
        echo "❌ Load balancer not found"
        echo "   File missing: $LOADBALANCER_LOG_FILE"
        echo ""
        echo "Start the cluster with: ./start_cluster.sh [NUM_SERVERS]"
        return
    fi
    
    LOADBALANCER_HOSTNAME=$(cat "$LOADBALANCER_LOG_FILE")
    LOADBALANCER_URL="http://$LOADBALANCER_HOSTNAME:$LOAD_BALANCER_PORT"
    
    echo "Load Balancer: $LOADBALANCER_URL"
    echo ""
    
    # Check if load balancer is responding
    if ! curl -s -f -m 2 "$LOADBALANCER_URL/" > /dev/null 2>&1; then
        echo "❌ Load balancer not responding"
        echo "   Check if the job is still running: squeue -u \$USER"
        return
    fi
    
    echo "✅ Load balancer is running"
    echo ""
    
    # Get server pool status
    echo "Server Pool Status:"
    echo "-----------------------------------"
    
    SERVER_STATUS=$(curl -s -f -m 5 "$LOADBALANCER_URL/servers" 2>/dev/null)
    
    if [[ -z "$SERVER_STATUS" ]]; then
        echo "❌ Could not retrieve server status"
        return
    fi
    
    # Parse and display server information using Python
    python3 << EOF
import json
import sys

try:
    data = json.loads('''$SERVER_STATUS''')
    
    total_servers = data.get('total_servers', 0)
    healthy_servers = data.get('healthy_servers', 0)
    total_active = data.get('total_active_requests', 0)
    
    print(f"Total Servers:    {total_servers}")
    print(f"Healthy Servers:  {healthy_servers}")
    print(f"Active Requests:  {total_active}")
    print()
    
    if total_servers == 0:
        print("No servers registered yet. Servers may still be starting up.")
    else:
        print("Individual Server Details:")
        print()
        servers = data.get('servers', [])
        
        for i, server in enumerate(servers, 1):
            status_icon = "✅" if server['is_healthy'] else "❌"
            hostname = server['hostname']
            port = server['port']
            active = server['active_requests']
            total = server['total_requests']
            success = server['successful_requests']
            failed = server['failed_requests']
            avg_time = server['avg_response_time']
            load_score = server['load_score']
            
            print(f"  Server {i}: {status_icon} {hostname}:{port}")
            print(f"    Active Requests:     {active}")
            print(f"    Total Requests:      {total}")
            print(f"    Successful:          {success}")
            print(f"    Failed:              {failed}")
            print(f"    Avg Response Time:   {avg_time:.2f}s")
            print(f"    Load Score:          {load_score:.2f}")
            print()

except json.JSONDecodeError as e:
    print(f"Error parsing server status: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
    
    echo ""
    echo "-----------------------------------"
    echo ""
    echo "Commands:"
    echo "  View registry:        cat $SERVER_REGISTRY_FILE | python3 -m json.tool"
    echo "  Test load balancer:   curl $LOADBALANCER_URL/"
    echo "  Check job queue:      squeue -u \$USER"
    echo ""
    
    if [[ "$CONTINUOUS" == true ]]; then
        echo "Refreshing in 5 seconds... (Press Ctrl+C to stop)"
    fi
}

# Main loop
if [[ "$CONTINUOUS" == true ]]; then
    while true; do
        show_status
        sleep 5
    done
else
    show_status
fi

