"""
Mock FastAPI server for constrained decoding smoke test.
Validates constraint logic without loading a full model.
"""
import json
import hashlib
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock Constrained Decoding Server", version="1.0")

RUN_HASH = hashlib.md5(f"{uuid.uuid4()}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
METRICS_DIR = Path("runs/server-only/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

class LLMRequest(BaseModel):
    input: str
    constraint_type: Optional[str] = None
    constraint: Optional[str] = None
    json_schema: Optional[dict] = None
    decode_args: Optional[dict] = None
    # Speculative decoding params
    enable_speculative: Optional[bool] = False
    speculative_method: Optional[str] = None  # "suffix", "ngram", "draft_model"
    num_speculative_tokens: Optional[int] = 5
    draft_model: Optional[str] = None

def _validate_constraint(constraint_type: Optional[str], constraint: Optional[str], json_schema: Optional[dict]) -> Tuple[bool, str]:
    """Validate constraint fields. Returns (is_valid, error_message)."""
    if not constraint_type:
        return True, ""
    
    constraint_type = constraint_type.lower()
    if constraint_type not in ("json", "grammar", "regex"):
        return False, f"Unknown constraint_type: {constraint_type}. Must be 'json', 'grammar', or 'regex'."
    
    if constraint_type == "json":
        if not json_schema:
            return False, "json_schema required for constraint_type='json'"
        if not isinstance(json_schema, dict):
            return False, "json_schema must be a dictionary"
    elif constraint_type in ("grammar", "regex"):
        if not constraint:
            return False, f"constraint required for constraint_type='{constraint_type}'"
        if constraint_type == "regex":
            try:
                re.compile(constraint)
            except re.error as e:
                return False, f"Invalid regex pattern: {e}"
    
    return True, ""

def _get_constraint_signature(constraint_type: Optional[str], constraint: Optional[str], json_schema: Optional[dict]) -> str:
    """Generate a signature for constraint profile."""
    if not constraint_type:
        return "unconstrained"
    if constraint_type == "json":
        schema_str = json.dumps(json_schema, sort_keys=True) if json_schema else ""
        return f"json:{hashlib.md5(schema_str.encode()).hexdigest()[:8]}"
    elif constraint_type == "grammar":
        return f"grammar:{hashlib.md5((constraint or '').encode()).hexdigest()[:8]}"
    elif constraint_type == "regex":
        return f"regex:{hashlib.md5((constraint or '').encode()).hexdigest()[:8]}"
    return "unconstrained"

def _generate_constrained_json(schema: dict) -> str:
    """Generate a simple JSON response matching the schema."""
    result = {}
    for key, prop in schema.get("properties", {}).items():
        if prop.get("type") == "number":
            result[key] = 0.95
        elif prop.get("type") == "integer":
            result[key] = 42
        elif prop.get("type") == "string":
            result[key] = "response"
        elif prop.get("type") == "boolean":
            result[key] = True
    return json.dumps(result)

@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Mock generate endpoint with constraint and speculative decoding support.
    Returns a canned response matching the constraint (if provided).
    """
    import time
    start_time = time.time()
    
    try:
        # Validate constraints
        constraint_type = request.constraint_type
        constraint = request.constraint
        json_schema = request.json_schema
        
        is_valid, error_msg = _validate_constraint(constraint_type, constraint, json_schema)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        constraint_sig = _get_constraint_signature(constraint_type, constraint, json_schema)
        
        # Generate mock constrained response
        if constraint_type == "json":
            generated_text = _generate_constrained_json(json_schema or {})
        else:
            generated_text = "Mock generated text for constrained response."
        
        # Mock speculative decoding acceptance rate
        speculative_accepted = None
        if request.enable_speculative:
            # Simulate acceptance rate (80% in mock, varies in reality)
            speculative_accepted = 0.80
            speculative_method = request.speculative_method or "suffix"
            num_spec_tokens = request.num_speculative_tokens or 5
        else:
            speculative_method = None
            num_spec_tokens = 0
        
        e2e_time = time.time() - start_time
        
        return {
            "generated_text": generated_text,
            "response_time_sec": round(e2e_time, 4),
            "_latency_sec": round(e2e_time, 4),
            "e2e_latency_sec": round(e2e_time, 4),
            "run_hash": RUN_HASH,
            "constraint_type": constraint_type,
            "constraint_signature": constraint_sig,
            "evaluationScore": 0.95,
            "speculative_decoding_enabled": request.enable_speculative,
            "speculative_method": speculative_method,
            "speculative_accepted": speculative_accepted,
            "num_speculative_tokens": num_spec_tokens,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Mock Constrained Decoding Server (for smoke tests)",
        "version": "1.0",
        "run_hash": RUN_HASH,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
