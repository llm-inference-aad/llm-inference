#!/usr/bin/env python3
"""
Test script to verify the local server integration works correctly.
"""

import sys
sys.path.append("src")

from llm_utils import submit_local_server
from cfg.constants import LLM_MODEL, INFERENCE_SUBMISSION

def test_local_server():
    """Test the local server connection and response."""
    print(f"Testing local server integration...")
    print(f"LLM_MODEL: {LLM_MODEL}")
    print(f"INFERENCE_SUBMISSION: {INFERENCE_SUBMISSION}")
    
    # Simple test prompt
    test_prompt = "Write a simple Python function that adds two numbers:"
    
    try:
        print(f"\nSending test prompt: {test_prompt}")
        response = submit_local_server(test_prompt, max_new_tokens=200, temperature=0.7)
        print(f"\nServer response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print("✅ Local server test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Local server test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_local_server()
    sys.exit(0 if success else 1)


