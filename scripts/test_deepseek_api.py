#!/usr/bin/env python3
"""
Minimal test script for DeepSeek API integration.

Usage:
    python scripts/test_deepseek_api.py
    python scripts/test_deepseek_api.py --model deepseek-coder
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_utils import submit_deepseek_api, clean_code_from_llm
from src.cfg.constants import DEEPSEEK_API_KEY


def test_api(model_id="deepseek-chat", temperature=0.7, top_p=0.8):
    """Test DeepSeek API with a simple code generation prompt"""
    
    if not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set")
        print("Add to .env: DEEPSEEK_API_KEY=your_key_here")
        sys.exit(1)
    
    print(f"Testing {model_id} (temp={temperature}, top_p={top_p})")
    print("-" * 60)
    
    # Mutation prompt 
    code_block = """
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
"""
    
    prompt = f"""Q: What complex modifications can be explored to potentially enhance the performance of this existing code snippet?

The current code block:
```python
{code_block.strip()}
```"""
    
    try:
        start = time.time()
        result = submit_deepseek_api(
            prompt,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=1024,
            gene_id="test"
        )
        elapsed = time.time() - start
        
        code = clean_code_from_llm(result)
        
        print(f"\n[SUCCESS] Latency: {elapsed:.2f}s")
        print(f"Response length: {len(result)} chars")
        print(f"\nExtracted code:\n{code[:200]}..." if len(code) > 200 else f"\nExtracted code:\n{code}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DeepSeek API")
    parser.add_argument("--model", default="deepseek-coder",
                       choices=["deepseek-chat", "deepseek-coder", "deepseek-reasoner"])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.5)
    
    args = parser.parse_args()
    
    success = test_api(args.model, args.temperature, args.top_p)
    
    if success:
        print("\n[READY] DeepSeek API is working!")
        print("Set LLM_MODEL='deepseek' in src/cfg/constants.py to use it.")
    else:
        sys.exit(1)

