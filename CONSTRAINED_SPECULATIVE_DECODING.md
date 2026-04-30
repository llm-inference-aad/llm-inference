# Constrained Decoding & Speculative Decoding Implementation

This document describes the implementation of **Constrained Decoding** and **Speculative Decoding** in your vLLM framework.

## Overview

### Constrained Decoding
Constrains the language model's output to match specific patterns or schemas:
- **JSON**: Output must conform to a specified JSON schema
- **Grammar**: Output must follow an LBNF (Backus-Naur Form) grammar
- **Regex**: Output must match a regular expression pattern

### Speculative Decoding
Accelerates token generation by using a draft model or heuristics to predict multiple tokens ahead:
- **Suffix**: Uses tree-based suffix speculation (dynamic depth)
- **N-gram**: Uses n-gram prompt lookup speculation
- **Draft Model**: Uses a smaller draft model for speculation

## Implementation Changes

### 1. server_vllm.py

#### New Imports
- Added `re` (regex validation)
- Added `Optional, Dict, Tuple` from typing

#### Environment Variables for Constrained Decoding
```python
CONSTRAINED_DECODING_ENABLED          # Enable/disable constraint support
DEFAULT_CONSTRAINT_TYPE               # Default constraint type: json|grammar|regex
DEFAULT_JSON_SCHEMA                   # Default JSON schema for all requests
CONSTRAINT_VALIDATION_TIMEOUT         # Timeout for constraint validation (seconds)
CONSTRAINT_LOGGING_ENABLED            # Log constraint-related decisions
ENABLE_JSON_CONSTRAINTS               # Backward-compatible flag
```

#### Environment Variables for Speculative Decoding
```python
ENABLE_SPECULATIVE_DECODING           # Enable/disable speculative decoding
VLLM_SPECULATIVE_MODEL                # Path to draft model (for draft_model method)
VLLM_NUM_SPECULATIVE_TOKENS           # Number of tokens to speculate (default: 5)
VLLM_SPECULATIVE_METHOD               # Method: suffix|ngram|draft_model
VLLM_ADAPTIVE_SPECULATION             # Use adaptive suffix speculation
VLLM_SUFFIX_DECODING_MAX_TREE_DEPTH   # Max tree depth for suffix speculation
VLLM_SUFFIX_DECODING_MAX_SPEC_FACTOR  # Max speculation factor
VLLM_SUFFIX_DECODING_MIN_TOKEN_PROB   # Min token probability for speculation
VLLM_SPECULATIVE_CONFIG               # Raw JSON config for speculative_config
SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP  # Cap temperature for constrained+speculative
SPECULATIVE_CONSTRAINED_TOP_P_CAP        # Cap top_p for constrained+speculative
ENABLE_PROFILED_BATCH_GROUPING           # Group batches by constraint profile
```

#### Updated LLMRequest Schema
```python
class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    top_p: float = 0.8
    temperature: float = 0.7
    job_id: str | None = "default"
    gene_id: str | None = None
    # New fields for constrained decoding
    constraint_type: str | None = None  # "json", "grammar", "regex"
    constraint: str | None = None       # Grammar or regex pattern
    json_schema: dict | None = None     # JSON schema for JSON constraints
```

#### Helper Functions
1. `_get_constraint_signature()`: Generate constraint profile signatures for batch grouping
2. `_validate_constraint()`: Validate constraint fields before generation
3. `_apply_speculative_config()`: Apply speculative decoding configuration to SamplingParams

#### Updated /generate Endpoint
- Validates constraint fields
- Applies constraints to vLLM SamplingParams
- Configures speculative decoding if enabled
- Handles combined constrained+speculative scenarios
- Tracks constraint_type in metrics

#### Metrics Updates
- Added `constraint_type` field to track constraint usage
- Added `speculative_accepted` field for future speculation metrics
- Enhanced session metadata with constraint/speculation config

### 2. server.sh

#### Constraint Enforcement Logic
Added section that:
- Checks if `CONSTRAINT_TYPE` is set
- Checks if `ENABLE_SPECULATIVE_DECODING` is true
- **Forces `VLLM_BACKEND=true`** if either constraint or speculation is enabled
- Logs the decision to stdout

This ensures constraints and speculation can only run with vLLM backend.

## Usage Examples

### 1. JSON Constraint
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Extract person info from: Alice, age 30, works at TechCorp",
        "max_new_tokens": 256,
        "constraint_type": "json",
        "json_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "company": {"type": "string"}
            },
            "required": ["name", "age"]
        }
    }
)
```

### 2. Grammar Constraint
```python
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "City with zip code: San Francisco 94102",
        "max_new_tokens": 256,
        "constraint_type": "grammar",
        "constraint": """
            start ::= city_name ws zip_code
            city_name ::= [a-zA-Z ]+
            ws ::= " "
            zip_code ::= [0-9]{5}
        """
    }
)
```

### 3. Regex Constraint
```python
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Email format:",
        "max_new_tokens": 256,
        "constraint_type": "regex",
        "constraint": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    }
)
```

### 4. Enable Speculative Decoding in .env
```bash
ENABLE_SPECULATIVE_DECODING=true
VLLM_ADAPTIVE_SPECULATION=true
VLLM_SPECULATIVE_METHOD=suffix
VLLM_NUM_SPECULATIVE_TOKENS=5
```

### 5. Combine Constraints + Speculation
```bash
# .env configuration
CONSTRAINT_TYPE=json
DEFAULT_JSON_SCHEMA='{"type":"object","properties":{"answer":{"type":"string"}}}'
ENABLE_SPECULATIVE_DECODING=true
VLLM_SPECULATIVE_METHOD=suffix

# Temperature/top_p will be automatically capped for better acceptance
SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP=0.25
SPECULATIVE_CONSTRAINED_TOP_P_CAP=0.9
```

## Configuration Files

### .env.vllm
Example configuration file with all constraint and speculation options documented.

### Load Balancer Integration
The load_balancer.py must include constraint fields when forwarding requests:
```python
constraint_fields = {
    "constraint_type": request.get("constraint_type"),
    "constraint": request.get("constraint"),
    "json_schema": request.get("json_schema"),
}
```

## Key Features

### 1. Backward Compatibility
- Legacy `ENABLE_JSON_CONSTRAINTS` still supported
- Falls back to default constraints if not specified in request
- Optional per-request overrides

### 2. Batch Grouping (Future)
- `ENABLE_PROFILED_BATCH_GROUPING`: Groups requests by constraint signature
- Prevents mixing constrained/unconstrained requests in same batch
- Improves speculative acceptance rates

### 3. Temperature/Top-P Capping
For constrained + speculative combinations:
- Temperatures capped to `SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP` (default: 0.25)
- Top_p capped to `SPECULATIVE_CONSTRAINED_TOP_P_CAP` (default: 0.9)
- Improves acceptance without affecting quality

### 4. Constraint Logging
- Enable with `CONSTRAINT_LOGGING_ENABLED=true`
- Logs all constraint validation decisions
- Helpful for debugging constraint issues

### 5. Timeout Protection
- `CONSTRAINT_VALIDATION_TIMEOUT`: Prevents hanging on complex constraints
- Default: 30 seconds

## Metrics & Monitoring

Each request stores:
```json
{
    "constraint_type": "json|grammar|regex|null",
    "speculative_accepted": null  // Populated when speculative decoding active
}
```

## Troubleshooting

### Constraint Not Applied
1. Check `CONSTRAINED_DECODING_ENABLED=true` in environment
2. Verify `CONSTRAINT_TYPE` is set or request has `constraint_type`
3. Enable `CONSTRAINT_LOGGING_ENABLED=true` to see validation logs
4. Verify vLLM backend is active (not HuggingFace)

### Speculative Decoding Not Working
1. Ensure `ENABLE_SPECULATIVE_DECODING=true`
2. For draft_model method: set `VLLM_SPECULATIVE_MODEL` to draft model path
3. Check vLLM version supports speculative decoding
4. Monitor acceptance rates in metrics

### Combined Constraints + Speculation Issues
1. Temperature/top_p will be auto-capped - this is expected
2. Lower `VLLM_NUM_SPECULATIVE_TOKENS` if acceptance is low
3. Use suffix method for dynamic depth adjustment
4. Enable `CONSTRAINT_LOGGING_ENABLED` for detailed logs

## Testing

Run the provided examples in [examples/constrained_decoding_demo.py](examples/constrained_decoding_demo.py):
```bash
python examples/constrained_decoding_demo.py
```

## References

- vLLM Constrained Decoding: https://docs.vllm.ai/en/latest/features/constraints.html
- vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/features/speculative_decoding.html
- Load balancer support: [load_balancer.py](load_balancer.py)
