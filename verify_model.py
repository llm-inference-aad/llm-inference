#!/usr/bin/env python3
"""
Simple script to verify that the deepseek-mutation-finetune model works correctly.
Tests model loading and basic inference.
"""

import os
import sys
import time
import torch
import transformers
from pathlib import Path

# Model path
MODEL_PATH = "/home/hice1/rmanimaran8/deepseek-mutation-finetune"

def print_step(message):
    """Print a formatted step message"""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}")

def verify_model_directory():
    """Check if model directory exists and contains necessary files"""
    print_step("Checking Model Directory")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model directory does not exist: {MODEL_PATH}")
        return False
    
    print(f"✅ Model directory exists: {MODEL_PATH}")
    
    # Check for common model files
    required_files = ["config.json"]
    optional_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    
    missing_required = []
    for file in required_files:
        file_path = os.path.join(MODEL_PATH, file)
        if os.path.exists(file_path):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_required.append(file)
    
    if missing_required:
        print(f"\n⚠️  Warning: Missing required files: {missing_required}")
        print("   Model may still work if it's a Hugging Face model in a different format")
    
    # Check for model weights
    weight_files = list(Path(MODEL_PATH).glob("*.safetensors")) + \
                   list(Path(MODEL_PATH).glob("*.bin")) + \
                   list(Path(MODEL_PATH).glob("model*.pt"))
    
    if weight_files:
        print(f"✅ Found model weight files: {len(weight_files)} files")
    else:
        print("⚠️  Warning: No model weight files found (.safetensors, .bin, or .pt)")
        print("   If this is a symbolic link or symlink to HF cache, this is okay")
    
    return True

def load_model():
    """Load the model and tokenizer"""
    print_step("Loading Model and Tokenizer")
    
    try:
        start_time = time.time()
        
        print(f"Loading from: {MODEL_PATH}")
        print("This may take a few minutes...")
        
        # Load tokenizer first (usually faster)
        print("\n[1/2] Loading tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer loaded in {time.time() - start_time:.2f} seconds")
        
        # Load model
        print("\n[2/2] Loading model...")
        model_start = time.time()
        
        # Check available device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="sdpa" if device == "cuda" else None
        ).eval()
        
        if device == "cpu":
            model = model.to(device)
        
        model_load_time = time.time() - model_start
        total_time = time.time() - start_time
        
        print(f"✅ Model loaded in {model_load_time:.2f} seconds")
        print(f"✅ Total loading time: {total_time:.2f} seconds")
        
        # Print model info
        print(f"\nModel Info:")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Dtype: {next(model.parameters()).dtype}")
        
        return model, tokenizer, True
        
    except Exception as e:
        print(f"❌ ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, False

def test_inference(model, tokenizer, test_prompts=None):
    """Test model inference with sample prompts"""
    print_step("Testing Inference")
    
    if test_prompts is None:
        # Default test prompts
        test_prompts = [
            "Write a Python function to calculate the factorial of a number:",
            "What is 2+2?",
            "Hello, how are you?",
        ]
    
    device = next(model.parameters()).device
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            generation_time = time.time() - start_time
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new tokens
            input_length = inputs['input_ids'].shape[1]
            new_text = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
            
            print(f"Generated text:\n{new_text}")
            print(f"\n✅ Generated in {generation_time:.2f} seconds")
            print(f"   Tokens: {outputs.shape[1] - input_length} new tokens")
            
        except Exception as e:
            print(f"❌ ERROR during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def main():
    """Main verification function"""
    print("\n" + "="*60)
    print("  DEEPSEEK-MUTATION-FINETUNE MODEL VERIFICATION")
    print("="*60)
    
    # Step 1: Verify directory
    if not verify_model_directory():
        print("\n❌ Model directory verification failed!")
        sys.exit(1)
    
    # Step 2: Load model
    model, tokenizer, success = load_model()
    if not success:
        print("\n❌ Model loading failed!")
        sys.exit(1)
    
    # Step 3: Test inference
    success = test_inference(model, tokenizer)
    if not success:
        print("\n❌ Inference testing failed!")
        sys.exit(1)
    
    # Success!
    print_step("Verification Complete!")
    print("✅ All tests passed!")
    print("✅ Model is working correctly!")
    print("\nYou can now use this model for inference.")

if __name__ == "__main__":
    main()

