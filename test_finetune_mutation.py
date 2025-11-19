#!/usr/bin/env python3
"""
Test script for finetune_mutation.py
Tests data loading, formatting, and dry-run mode.
"""

import sys
import json
from pathlib import Path

# Test 1: Check if dataset.json exists and is valid
def test_dataset_exists():
    print("Test 1: Checking if dataset.json exists...")
    dataset_path = Path("dataset.json")
    if not dataset_path.exists():
        print("❌ FAIL: dataset.json not found")
        return False
    print("✅ PASS: dataset.json exists")
    return True

def test_dataset_valid():
    print("\nTest 2: Checking if dataset.json is valid JSON...")
    try:
        with open("dataset.json", 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("❌ FAIL: dataset.json is not a list")
            return False
        if len(data) == 0:
            print("⚠️  WARN: dataset.json is empty")
        else:
            print(f"✅ PASS: dataset.json contains {len(data)} entries")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ FAIL: Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error reading dataset: {e}")
        return False

def test_dataset_structure():
    print("\nTest 3: Checking dataset structure...")
    try:
        with open("dataset.json", 'r') as f:
            data = json.load(f)
        
        required_fields = ['prompt', 'fitness', 'generated_text']
        missing_fields = []
        
        for i, entry in enumerate(data):
            for field in required_fields:
                if field not in entry:
                    missing_fields.append(f"Entry {i} missing '{field}'")
        
        if missing_fields:
            print(f"❌ FAIL: Missing fields in dataset:")
            for msg in missing_fields[:5]:  # Show first 5
                print(f"   - {msg}")
            return False
        
        print("✅ PASS: All entries have required fields")
        return True
    except Exception as e:
        print(f"❌ FAIL: Error checking structure: {e}")
        return False

def test_data_loading_function():
    print("\nTest 4: Testing load_and_format_data function...")
    try:
        # Import the function
        sys.path.insert(0, '.')
        from finetune_mutation import load_and_format_data
        
        dataset = load_and_format_data("dataset.json")
        
        if len(dataset) == 0:
            print("⚠️  WARN: Dataset is empty after formatting")
        else:
            print(f"✅ PASS: Loaded {len(dataset)} examples")
            
            # Check format
            sample = dataset[0]
            if 'text' not in sample:
                print("❌ FAIL: Formatted dataset missing 'text' field")
                return False
            
            text = sample['text']
            if '### Instruction:' not in text:
                print("❌ FAIL: Formatted text missing '### Instruction:'")
                return False
            if '### Target Fitness:' not in text:
                print("❌ FAIL: Formatted text missing '### Target Fitness:'")
                return False
            if '### Response:' not in text:
                print("❌ FAIL: Formatted text missing '### Response:'")
                return False
            
            print("✅ PASS: Data formatting looks correct")
            print(f"   Sample text preview: {text[:100]}...")
        
        return True
    except ImportError as e:
        print(f"❌ FAIL: Could not import function: {e}")
        print("   Make sure you're in the correct directory")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error in load_and_format_data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dry_run():
    print("\nTest 5: Testing dry-run mode...")
    try:
        # Test by importing and calling main directly with dry-run flag
        import sys
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Save original argv
        original_argv = sys.argv[:]
        try:
            sys.argv = ['finetune_mutation.py', '--dry-run']
            
            # Capture output
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                from finetune_mutation import main
                main()
            
            output = f.getvalue()
            
            if "Dry run complete" not in output:
                print("⚠️  WARN: Dry-run completed but didn't print expected message")
                print("Output:", output[:200])
                return False
            else:
                print("✅ PASS: Dry-run completed successfully")
                # Verify it loaded the dataset
                if "Loaded" in output and "examples" in output:
                    print("   ✓ Dataset loaded correctly")
            
            return True
        finally:
            sys.argv = original_argv
        
    except Exception as e:
        print(f"❌ FAIL: Error running dry-run: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    print("\nTest 6: Testing imports...")
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        print("✅ PASS: All imports successful")
        return True
    except ImportError as e:
        print(f"❌ FAIL: Import error: {e}")
        print("   Run: pip install peft bitsandbytes trl scipy datasets transformers torch")
        return False

def main():
    print("=" * 60)
    print("Testing finetune_mutation.py")
    print("=" * 60)
    
    tests = [
        test_dataset_exists,
        test_dataset_valid,
        test_dataset_structure,
        test_imports,
        test_data_loading_function,
        test_dry_run,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ FAIL: Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(tests)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

