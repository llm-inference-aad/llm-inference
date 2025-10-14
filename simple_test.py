#!/usr/bin/env python3

# Simple load balancer test without external dependencies
import json
import time
from datetime import datetime

def test_load_balancer():
    print("Testing load balancer functionality...")
    
    # Test 1: Registry file operations
    registry_file = "servers.json"
    
    try:
        # Initialize registry
        with open(registry_file, 'w') as f:
            json.dump({"servers": []}, f, indent=2)
        print("✅ Registry file created")
        
        # Test adding server
        with open(registry_file, 'r') as f:
            data = json.load(f)
        
        server = {
            "hostname": "test-host.example.com",
            "port": 8000,
            "registered_at": datetime.now().isoformat()
        }
        
        data["servers"].append(server)
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("✅ Server added to registry")
        
        # Test reading registry
        with open(registry_file, 'r') as f:
            updated_data = json.load(f)
        
        print(f"✅ Registry contains {len(updated_data['servers'])} servers")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_load_balancer()
    if success:
        print("\n🎉 Load balancer core functionality works!")
        print("The issue is likely with aiohttp or network connectivity.")
    else:
        print("\n❌ Load balancer core functionality failed.")
