#!/usr/bin/env python
"""
Download and verify Llama-3.2-1B-Instruct model weights.
Can resume if interrupted. Tests model loading.
"""
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_REPO = "meta-llama/Llama-3.2-1B-Instruct"
CACHE_DIR = Path("/home/hice1/jgil37/scratch/llm_models")
MODEL_PATH = CACHE_DIR / "meta-llama" / "Llama-3.2-1B-Instruct"

def main():
    print(f"Model Repository: {MODEL_REPO}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"Target Model Path: {MODEL_PATH}")
    print()
    
    # Create directory
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # Check if weights already exist
    weights_files = list(MODEL_PATH.glob("*.safetensors")) + list(MODEL_PATH.glob("*.bin"))
    if weights_files:
        print(f"[INFO] Found {len(weights_files)} weight files:")
        for f in weights_files:
            size_gb = f.stat().st_size / (1024**3)
            print(f"  - {f.name}: {size_gb:.2f} GB")
        print()
    else:
        print("[INFO] No weight files found. Starting download...")
        print()
    
    try:
        # Download tokenizer
        print("[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True
        )
        print("[OK] Tokenizer downloaded")
        print()
        
        # Download model weights
        print("[2/2] Downloading model weights (this may take 5-15 minutes)...")
        print("      Please be patient, this is a one-time download...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            cache_dir=str(CACHE_DIR),
            torch_dtype=torch.float32,
            device_map="cpu",
            load_in_4bit=False,
            trust_remote_code=True,
            resume_download=True,
            force_download=False
        )
        print("[OK] Model weights downloaded")
        print()
        
        # Verify downloads
        print("[VERIFY] Checking downloaded files:")
        weights_files = list(MODEL_PATH.glob("*.safetensors")) + list(MODEL_PATH.glob("*.bin"))
        config_files = list(MODEL_PATH.glob("*.json")) + list(MODEL_PATH.glob("*.md"))
        
        print(f"  Weight files: {len(weights_files)}")
        total_size_gb = sum(f.stat().st_size for f in weights_files) / (1024**3)
        print(f"  Total weight size: {total_size_gb:.2f} GB")
        print(f"  Config/metadata files: {len(config_files)}")
        print()
        
        # Quick inference test
        print("[TEST] Running quick inference test...")
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input: '{test_input}'")
        print(f"  Output: '{result}'")
        print()
        
        print("[SUCCESS] Model download and verification complete!")
        print(f"Model is ready at: {MODEL_PATH}")
        print()
        print("Next steps:")
        print("  1. Run: sbatch submit_1b_test.sh")
        print("  2. Or run: bash test_five_configs.sh (with mock server)")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Download/verification failed:")
        print(f"  {type(e).__name__}: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  - Check internet connection")
        print("  - Verify disk space: df -h /home/hice1/jgil37/scratch")
        print("  - Check model cache: du -sh /home/hice1/jgil37/scratch/llm_models")
        print("  - Ensure HuggingFace token is set (if model is gated)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
