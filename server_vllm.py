"""
vLLM-backed inference server with PagedAttention KV cache optimization,
Constrained Decoding, and Speculative Decoding support.

Replaces HuggingFace Transformers pipeline with vLLM engine for:
- PagedAttention: Near-zero KV cache waste (95%+ utilization vs ~30% before)
- Continuous batching: Requests start/finish independently (no wait-then-process)
- Prefix caching: System prompt KV cache computed once, reused across all requests
- Tensor parallelism: Proper multi-GPU sharding (vs naive layer splitting)
- Constrained Decoding: JSON, Grammar, and Regex constraints via vLLM
- Speculative Decoding: Suffix, Draft, or N-gram speculation

API contract includes constrained decoding and speculative decoding support.
"""

import time
import os
import json
import hashlib
import uuid
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

# Force vLLM to use spawn to prevent CUDA initialization crashes
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.cfg.constants import *
from evaluator import OutputEvaluator

try:
    from src.rag.runtime import get_runtime
except Exception:
    def get_runtime():
        return None

# ---------------------------------------------------------------------------
# App & Config
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM API (vLLM+Constraints)", version="3.0")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct",
)
MODEL_ID = os.getenv("MODEL_ID", Path(MODEL_PATH).name)
MODEL_REVISION = os.getenv("MODEL_REVISION", "untracked")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
MODEL_QUANTIZATION = os.getenv("MODEL_QUANTIZATION", "bf16")

# vLLM-specific settings
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "2"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "bfloat16")
_vllm_quant_raw = os.getenv("VLLM_QUANTIZATION", "").strip()
if _vllm_quant_raw.lower() in ("", "none", "false", "0", "no"):
    VLLM_QUANTIZATION = None
else:
    VLLM_QUANTIZATION = _vllm_quant_raw
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", "true").lower() in (
    "true", "1", "yes",
)
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "false").lower() in ("true", "1", "yes",)

# ---------------------------------------------------------------------------
# Constrained Decoding Configuration
# ---------------------------------------------------------------------------
CONSTRAINED_DECODING_ENABLED = os.getenv("CONSTRAINED_DECODING_ENABLED", "true").lower() in (
    "true", "1", "yes",
)
DEFAULT_CONSTRAINT_TYPE = os.getenv("DEFAULT_CONSTRAINT_TYPE", "").lower().strip()
# Support backward-compat ENABLE_JSON_CONSTRAINTS
if not DEFAULT_CONSTRAINT_TYPE and os.getenv("ENABLE_JSON_CONSTRAINTS", "").lower() in ("true", "1", "yes"):
    DEFAULT_CONSTRAINT_TYPE = "json"
DEFAULT_JSON_SCHEMA = os.getenv("DEFAULT_JSON_SCHEMA", "")
if DEFAULT_JSON_SCHEMA:
    try:
        DEFAULT_JSON_SCHEMA = json.loads(DEFAULT_JSON_SCHEMA)
    except json.JSONDecodeError:
        print(f"[vLLM WARN] Invalid DEFAULT_JSON_SCHEMA JSON, ignoring", flush=True)
        DEFAULT_JSON_SCHEMA = None

CONSTRAINT_VALIDATION_TIMEOUT = int(os.getenv("CONSTRAINT_VALIDATION_TIMEOUT", "30"))
CONSTRAINT_LOGGING_ENABLED = os.getenv("CONSTRAINT_LOGGING_ENABLED", "false").lower() in (
    "true", "1", "yes",
)

# ---------------------------------------------------------------------------
# Speculative Decoding Configuration (vLLM only)
# ---------------------------------------------------------------------------
ENABLE_SPECULATIVE_DECODING = os.getenv("ENABLE_SPECULATIVE_DECODING", "false").lower() in (
    "true", "1", "yes",
)
VLLM_SPECULATIVE_MODEL = os.getenv("VLLM_SPECULATIVE_MODEL", "")
VLLM_NUM_SPECULATIVE_TOKENS = int(os.getenv("VLLM_NUM_SPECULATIVE_TOKENS", "5"))
_vllm_spec_method_raw = os.getenv("VLLM_SPECULATIVE_METHOD", "").lower().strip()
VLLM_ADAPTIVE_SPECULATION = os.getenv("VLLM_ADAPTIVE_SPECULATION", "false").lower() in (
    "true", "1", "yes",
)
# If ADAPTIVE=true but no method specified, vLLM defaults to suffix speculation
if VLLM_ADAPTIVE_SPECULATION and not _vllm_spec_method_raw:
    VLLM_SPECULATIVE_METHOD = "suffix"
else:
    VLLM_SPECULATIVE_METHOD = _vllm_spec_method_raw or ""

VLLM_SUFFIX_DECODING_MAX_TREE_DEPTH = int(os.getenv("VLLM_SUFFIX_DECODING_MAX_TREE_DEPTH", "24"))
VLLM_SUFFIX_DECODING_MAX_SPEC_FACTOR = float(os.getenv("VLLM_SUFFIX_DECODING_MAX_SPEC_FACTOR", "1.0"))
VLLM_SUFFIX_DECODING_MIN_TOKEN_PROB = float(os.getenv("VLLM_SUFFIX_DECODING_MIN_TOKEN_PROB", "0.1"))

# Raw JSON config merged into speculative_config (lowest priority)
_vllm_spec_config_raw = os.getenv("VLLM_SPECULATIVE_CONFIG", "")
VLLM_SPECULATIVE_CONFIG = {}
if _vllm_spec_config_raw:
    try:
        VLLM_SPECULATIVE_CONFIG = json.loads(_vllm_spec_config_raw)
    except json.JSONDecodeError:
        print(f"[vLLM WARN] Invalid VLLM_SPECULATIVE_CONFIG JSON, ignoring", flush=True)

# Temperature/top_p capping for constrained+speculative (to improve acceptance)
SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP = float(
    os.getenv("SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP", "0.25")
)
SPECULATIVE_CONSTRAINED_TOP_P_CAP = float(
    os.getenv("SPECULATIVE_CONSTRAINED_TOP_P_CAP", "0.9")
)
ENABLE_PROFILED_BATCH_GROUPING = os.getenv("ENABLE_PROFILED_BATCH_GROUPING", "true").lower() in (
    "true", "1", "yes",
)

# Adaptive speculation tuning (keeps the default stable, but can adapt per prompt/constraint profile)
ENABLE_ADAPTIVE_SPECULATION = os.getenv("VLLM_ADAPTIVE_SPECULATION", "true").lower() in (
    "true", "1", "yes",
)
SPECULATIVE_TOKEN_MIN = int(os.getenv("VLLM_SPECULATIVE_TOKEN_MIN", "2"))
SPECULATIVE_TOKEN_MAX = int(os.getenv("VLLM_SPECULATIVE_TOKEN_MAX", str(max(8, VLLM_NUM_SPECULATIVE_TOKENS * 2))))
SPECULATIVE_ACCEPTANCE_UPPER = float(os.getenv("VLLM_SPECULATIVE_ACCEPTANCE_UPPER", "0.80"))
SPECULATIVE_ACCEPTANCE_LOWER = float(os.getenv("VLLM_SPECULATIVE_ACCEPTANCE_LOWER", "0.45"))
SPECULATIVE_ACCEPTANCE_EWMA_ALPHA = float(os.getenv("VLLM_SPECULATIVE_ACCEPTANCE_ALPHA", "0.30"))
SPECULATIVE_ACCEPTANCE_STATE: Dict[str, float] = {}
SPECULATIVE_ACCEPTANCE_LOCK = threading.Lock()

# System prompt — cached via vLLM's automatic prefix caching
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Output a single fenced code block with runnable Python code and nothing else.\n"
    "Do not include explanations, comments outside the code block, or extra code fences.\n"
    "Begin with ```python and end with ```. If you cannot comply, output exactly FAIL.\n\n",
)

# ---------------------------------------------------------------------------
# Metrics (identical contract to server_baseline.py)
# ---------------------------------------------------------------------------
RUN_HASH = hashlib.md5(
    f"{uuid.uuid4()}_{datetime.now().isoformat()}".encode()
).hexdigest()[:16]

RUN_ID = os.getenv("RUN_ID", "server-only")

METRICS_DIR = Path(os.getenv("RUN_METRICS_DIR", RUN_METRICS_DIR))
METRICS_FILE = METRICS_DIR / f"latency-{RUN_HASH}.json"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

metrics_metadata = {
    "run_id": RUN_ID,
    "run_hash": RUN_HASH,
    "session_start": datetime.now().isoformat(),
    "model_path": MODEL_PATH,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "model_version": MODEL_VERSION,
    "model_quantization": MODEL_QUANTIZATION,
    "engine": "vllm",
    "batch_size": "continuous",
    "batch_wait_time": 0,
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "max_model_len": MAX_MODEL_LEN,
    "vllm_dtype": VLLM_DTYPE,
    "vllm_quantization": VLLM_QUANTIZATION,
    "prefix_caching": ENABLE_PREFIX_CACHING,
    "constrained_decoding_enabled": CONSTRAINED_DECODING_ENABLED,
    "default_constraint_type": DEFAULT_CONSTRAINT_TYPE,
    "speculative_decoding_enabled": ENABLE_SPECULATIVE_DECODING,
    "vllm_speculative_method": VLLM_SPECULATIVE_METHOD,
    "vllm_num_speculative_tokens": VLLM_NUM_SPECULATIVE_TOKENS,
    "requests": [],
}

with open(METRICS_FILE, "w") as f:
    json.dump(metrics_metadata, f, indent=2)


def save_latency_metrics(
    request_data,
    e2e_time,
    generation_time,
    queue_wait_time=None,
    evaluation_score=None,
    prompt_tokens=None,
    completion_tokens=None,
    total_tokens=None,
    constraint_type=None,
    speculative_accepted=None,
):
    """Save end-to-end latency metrics to JSON file."""
    try:
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

        request_metrics = {
            "timestamp": datetime.now().isoformat(),
            "job_id": request_data.get("job_id", "default"),
            "gene_id": request_data.get("gene_id"),
            "prompt_length": len(request_data.get("prompt", "")),
            "max_new_tokens": request_data.get("max_new_tokens", MAX_MODEL_LEN),
            "temperature": request_data.get("temperature", 0.7),
            "top_p": request_data.get("top_p", 0.8),
            "_latency_sec": round(e2e_time, 4),
            "batch_processing_time_sec": round(generation_time, 4),
            "generation_time_sec": round(generation_time, 4),
            "batch_size": 1,
            "queue_wait_time_sec": round(queue_wait_time, 4) if queue_wait_time is not None else None,
            "evaluation_score": evaluation_score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "constraint_type": constraint_type,
            "speculative_accepted": speculative_accepted,
        }

        metrics["requests"].append(request_metrics)

        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"Error saving latency metrics: {e}", flush=True)


# ---------------------------------------------------------------------------
# Request schema (with constrained decoding & speculative decoding support)
# ---------------------------------------------------------------------------
class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    top_p: float = 0.8
    temperature: float = 0.7
    job_id: str | None = "default"
    gene_id: str | None = None
    # Constrained decoding fields (optional, per-request)
    constraint_type: str | None = None  # "json", "grammar", "regex"
    constraint: str | None = None  # Grammar or regex pattern
    json_schema: dict | None = None  # JSON schema for constraint_type="json"


# ---------------------------------------------------------------------------
# Constraint Validation & Helper Functions
# ---------------------------------------------------------------------------
def _get_constraint_signature(constraint_type: Optional[str], constraint: Optional[str], json_schema: Optional[dict]) -> str:
    """Generate a signature for batch grouping based on constraint profile."""
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


def _constraint_profile_key(constraint_type: Optional[str], constraint: Optional[str], json_schema: Optional[dict]) -> str:
    """Stable key for adapting speculation by request profile."""
    if not constraint_type:
        return "unconstrained"
    return _get_constraint_signature(constraint_type, constraint, json_schema)


def _adaptive_speculative_token_budget(
    prompt_len: int,
    has_constraint: bool,
    constraint_type: Optional[str],
    acceptance_ema: Optional[float],
) -> int:
    """Choose a speculative token budget using prompt length, constraint strictness, and recent acceptance."""
    budget = VLLM_NUM_SPECULATIVE_TOKENS

    # Prompt length heuristic: short prompts are more stable, long prompts need less speculation.
    if prompt_len < 96:
        budget += 2
    elif prompt_len < 256:
        budget += 1
    elif prompt_len > 1024:
        budget -= 2
    elif prompt_len > 512:
        budget -= 1

    # Strict constraints reduce speculative breadth.
    if has_constraint:
        if constraint_type == "json":
            budget -= 2
        elif constraint_type == "regex":
            budget -= 1
        elif constraint_type == "grammar":
            budget -= 1

    # Recent acceptance feedback nudges the budget up/down.
    if acceptance_ema is not None:
        if acceptance_ema >= SPECULATIVE_ACCEPTANCE_UPPER + 0.05:
            budget += 2
        elif acceptance_ema >= SPECULATIVE_ACCEPTANCE_UPPER:
            budget += 1
        elif acceptance_ema <= SPECULATIVE_ACCEPTANCE_LOWER - 0.10:
            budget -= 2
        elif acceptance_ema <= SPECULATIVE_ACCEPTANCE_LOWER:
            budget -= 1

    return max(SPECULATIVE_TOKEN_MIN, min(SPECULATIVE_TOKEN_MAX, budget))


def _build_speculative_config(
    prompt_len: int,
    has_constraint: bool,
    constraint_type: Optional[str],
) -> Tuple[dict, int]:
    """Build a request-specific speculative configuration."""
    speculative_config = dict(VLLM_SPECULATIVE_CONFIG)

    if not ENABLE_SPECULATIVE_DECODING:
        return speculative_config, 0

    if VLLM_SPECULATIVE_METHOD:
        speculative_config["method"] = VLLM_SPECULATIVE_METHOD

    profile_key = f"{constraint_type or 'unconstrained'}:{VLLM_SPECULATIVE_METHOD or 'off'}"
    with SPECULATIVE_ACCEPTANCE_LOCK:
        acceptance_ema = SPECULATIVE_ACCEPTANCE_STATE.get(profile_key)

    num_tokens = VLLM_NUM_SPECULATIVE_TOKENS
    if ENABLE_ADAPTIVE_SPECULATION:
        num_tokens = _adaptive_speculative_token_budget(prompt_len, has_constraint, constraint_type, acceptance_ema)

    speculative_config["num_speculative_tokens"] = num_tokens

    if VLLM_SPECULATIVE_METHOD == "suffix":
        # Tighten suffix tree depth as prompt grows; keep it bounded for constrained prompts.
        speculative_config.setdefault("max_tree_depth", min(VLLM_SUFFIX_DECODING_MAX_TREE_DEPTH, max(8, num_tokens * 4)))
        speculative_config.setdefault("max_spec_factor", min(VLLM_SUFFIX_DECODING_MAX_SPEC_FACTOR, max(0.4, num_tokens / 10.0)))
        speculative_config.setdefault(
            "min_token_prob",
            max(VLLM_SUFFIX_DECODING_MIN_TOKEN_PROB, 0.15 if has_constraint else VLLM_SUFFIX_DECODING_MIN_TOKEN_PROB),
        )
    elif VLLM_SPECULATIVE_METHOD == "ngram":
        speculative_config.setdefault("num_speculative_tokens", num_tokens)
    elif VLLM_SPECULATIVE_METHOD == "draft_model" and VLLM_SPECULATIVE_MODEL:
        speculative_config.setdefault("draft_model", VLLM_SPECULATIVE_MODEL)
        speculative_config.setdefault("num_speculative_tokens", num_tokens)

    return speculative_config, num_tokens


def _extract_speculative_accepted(output) -> Optional[float]:
    """Best-effort extraction of speculative acceptance from vLLM output objects."""
    candidate_paths = [
        ("metrics", "speculative_accepted"),
        ("metrics", "acceptance_rate"),
        ("metrics", "speculative_acceptance_rate"),
        ("speculative_accepted",),
        ("speculative_acceptance_rate",),
        ("acceptance_rate",),
    ]

    def _follow(obj, attrs):
        cur = obj
        for attr in attrs:
            if cur is None:
                return None
            if isinstance(cur, dict):
                cur = cur.get(attr)
            else:
                cur = getattr(cur, attr, None)
        return cur

    for attrs in candidate_paths:
        value = _follow(output, attrs)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _update_speculative_acceptance(profile_key: str, accepted: Optional[float]) -> None:
    """Update the rolling acceptance EMA for a request profile."""
    if accepted is None:
        return
    accepted = max(0.0, min(1.0, float(accepted)))
    with SPECULATIVE_ACCEPTANCE_LOCK:
        prev = SPECULATIVE_ACCEPTANCE_STATE.get(profile_key, accepted)
        SPECULATIVE_ACCEPTANCE_STATE[profile_key] = (
            SPECULATIVE_ACCEPTANCE_EWMA_ALPHA * accepted
            + (1.0 - SPECULATIVE_ACCEPTANCE_EWMA_ALPHA) * prev
        )


def _apply_speculative_config(sampling_params, has_constraint: bool, speculative_config: Optional[dict] = None) -> None:
    """Apply speculative decoding configuration to vLLM SamplingParams if enabled.

    When a request-specific speculative_config is provided, it takes precedence.
    """
    if not ENABLE_SPECULATIVE_DECODING:
        return
    
    # For constrained+speculative, cap temperature/top_p for better acceptance
    if has_constraint:
        sampling_params.temperature = min(sampling_params.temperature, SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP)
        sampling_params.top_p = min(sampling_params.top_p, SPECULATIVE_CONSTRAINED_TOP_P_CAP)
        if CONSTRAINT_LOGGING_ENABLED:
            print(f"[vLLM] Constrained+Speculative: capped temp={sampling_params.temperature}, top_p={sampling_params.top_p}", flush=True)
    
    # Build speculative_config if the caller did not provide one.
    if speculative_config is None:
        speculative_config = dict(VLLM_SPECULATIVE_CONFIG)  # Start with user config

        if VLLM_SPECULATIVE_METHOD:
            speculative_config["method"] = VLLM_SPECULATIVE_METHOD

        if VLLM_SPECULATIVE_METHOD == "suffix":
            speculative_config.setdefault("num_speculative_tokens", VLLM_NUM_SPECULATIVE_TOKENS)
            speculative_config.setdefault("max_tree_depth", VLLM_SUFFIX_DECODING_MAX_TREE_DEPTH)
            speculative_config.setdefault("max_spec_factor", VLLM_SUFFIX_DECODING_MAX_SPEC_FACTOR)
            speculative_config.setdefault("min_token_prob", VLLM_SUFFIX_DECODING_MIN_TOKEN_PROB)
        elif VLLM_SPECULATIVE_METHOD == "ngram":
            speculative_config.setdefault("num_speculative_tokens", VLLM_NUM_SPECULATIVE_TOKENS)
        elif VLLM_SPECULATIVE_METHOD == "draft_model" and VLLM_SPECULATIVE_MODEL:
            speculative_config.setdefault("draft_model", VLLM_SPECULATIVE_MODEL)
            speculative_config.setdefault("num_speculative_tokens", VLLM_NUM_SPECULATIVE_TOKENS)
    
    if speculative_config and hasattr(sampling_params, "speculative_config"):
        sampling_params.speculative_config = speculative_config


# ---------------------------------------------------------------------------
# vLLM Engine (singleton)
# ---------------------------------------------------------------------------
_llm = None


def get_llm():
    """Lazy-init the vLLM engine (loaded once at startup)."""
    global _llm
    if _llm is None:
        from vllm import LLM

        print(f"[vLLM] Loading model: {MODEL_PATH}", flush=True)
        print(f"[vLLM] model_id={MODEL_ID}", flush=True)
        print(f"[vLLM] model_revision={MODEL_REVISION}", flush=True)
        print(f"[vLLM] model_version={MODEL_VERSION}", flush=True)
        print(f"[vLLM] model_quantization={MODEL_QUANTIZATION}", flush=True)
        print(f"[vLLM] tensor_parallel_size={TENSOR_PARALLEL_SIZE}", flush=True)
        print(f"[vLLM] gpu_memory_utilization={GPU_MEMORY_UTILIZATION}", flush=True)
        print(f"[vLLM] max_model_len={MAX_MODEL_LEN}", flush=True)
        print(f"[vLLM] dtype={VLLM_DTYPE}", flush=True)
        print(f"[vLLM] quantization={VLLM_QUANTIZATION}", flush=True)
        print(f"[vLLM] enable_prefix_caching={ENABLE_PREFIX_CACHING}", flush=True)
        print(f"[vLLM] enforce_eager={ENFORCE_EAGER}", flush=True)
        if CONSTRAINED_DECODING_ENABLED:
            print(f"[vLLM] constrained_decoding_enabled, default_constraint_type={DEFAULT_CONSTRAINT_TYPE}", flush=True)
        if ENABLE_SPECULATIVE_DECODING:
            print(f"[vLLM] speculative_decoding enabled, method={VLLM_SPECULATIVE_METHOD}", flush=True)

        start = time.time()
        llm_kwargs = dict(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype=VLLM_DTYPE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            enforce_eager=ENFORCE_EAGER,
            trust_remote_code=True,
        )
        # Note: device/platform selection is handled by vLLM internals.
        # If platform detection fails, set appropriate vLLM env vars
        # (e.g., VLLM_* settings) before starting the server instead of
        # passing a 'device' kwarg which may not be supported by this vLLM version.
        if VLLM_QUANTIZATION:
            llm_kwargs["quantization"] = VLLM_QUANTIZATION
        _llm = LLM(**llm_kwargs)
        elapsed = time.time() - start
        print(f"[vLLM] Model loaded in {elapsed:.1f}s", flush=True)

    return _llm


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Generate text with optional constrained decoding and speculative decoding.
    
    Supports:
    - constraint_type: "json", "grammar", "regex" (optional)
    - constraint: Grammar or regex pattern (for grammar/regex constraints)
    - json_schema: Schema dict (for JSON constraints)
    """
    from vllm import SamplingParams

    _start_time = time.time()

    try:
        # Resolve constraint_type (request override or default)
        constraint_type = (request.constraint_type or DEFAULT_CONSTRAINT_TYPE).lower() if request.constraint_type or DEFAULT_CONSTRAINT_TYPE else None
        constraint = request.constraint
        json_schema = request.json_schema or DEFAULT_JSON_SCHEMA
        
        # Validate constraints
        is_valid, error_msg = _validate_constraint(constraint_type, constraint, json_schema)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        has_constraint = bool(constraint_type)
        if has_constraint and CONSTRAINT_LOGGING_ENABLED:
            print(f"[vLLM] Constraint validated: type={constraint_type}", flush=True)
        

        # Prepend system prompt (prefix-cached by vLLM)
        full_prompt = SYSTEM_PROMPT + request.prompt

        # Optionally augment the prompt with RAG context when enabled.
        # RAG is intentionally read-only here: it retrieves from the centralized
        # vector DB and prepends historical mutations / research context.
        augmented_prompt = full_prompt
        try:
            if globals().get("RAG_ENABLED", False):
                runtime = get_runtime()
                if runtime is not None:
                    augmented_template, mutations = runtime.enhance_template(
                        template=full_prompt,
                        mutation_type=None,
                        query_code=request.prompt,
                        gene_id=request.gene_id,
                    )
                    augmented_prompt = augmented_template
                    print(
                        f"[vLLM] RAG context injected: retrieved {len(mutations)} items for gene {request.gene_id}",
                        flush=True,
                    )
        except Exception as e:
            print(f"[vLLM] Warning: RAG augmentation failed: {e}", flush=True)

        # Clamp max_tokens so prompt + output fits within MAX_MODEL_LEN
        llm = get_llm()
        prompt_token_ids = llm.get_tokenizer().encode(augmented_prompt)
        prompt_len = len(prompt_token_ids)
        safe_max_tokens = min(request.max_new_tokens, MAX_MODEL_LEN - prompt_len)
        if safe_max_tokens < 1:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long ({prompt_len} tokens) for MAX_MODEL_LEN={MAX_MODEL_LEN}. "
                       f"No room for output tokens.",
            )
        if safe_max_tokens < request.max_new_tokens:
            print(
                f"[vLLM] Clamped max_tokens {request.max_new_tokens}→{safe_max_tokens} "
                f"(prompt={prompt_len}, limit={MAX_MODEL_LEN})",
                flush=True,
            )

        # Build vLLM sampling parameters
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=safe_max_tokens,
            repetition_penalty=1.1,
        )

        profile_key = _constraint_profile_key(constraint_type, constraint, json_schema)
        
        # Add constraint parameters if present
        if has_constraint:
            if constraint_type == "json":
                params.guided_choice = None  # vLLM uses guided_json or similar
                # vLLM v0.4.0+ uses: params.constraint_dict = {"type": "json", "schema": json_schema}
                # For compatibility, we'll set these as attributes
                if hasattr(params, 'constraint_dict'):
                    params.constraint_dict = {"type": "json", "schema": json_schema}
                else:
                    # Fallback: try setting directly
                    params.json_schema = json_schema
            elif constraint_type == "grammar":
                if hasattr(params, 'constraint_dict'):
                    params.constraint_dict = {"type": "grammar", "grammar": constraint}
                else:
                    params.grammar = constraint
            elif constraint_type == "regex":
                if hasattr(params, 'constraint_dict'):
                    params.constraint_dict = {"type": "regex", "pattern": constraint}
                else:
                    params.regex = constraint
        
        # Apply speculative decoding configuration
        speculative_config, chosen_spec_tokens = _build_speculative_config(prompt_len, has_constraint, constraint_type)
        _apply_speculative_config(params, has_constraint, speculative_config)

        print(
            f"[vLLM] Request received [Job: {request.job_id}, Gene: {request.gene_id}, Constraint: {constraint_type}, SpecTokens: {chosen_spec_tokens}]",
            flush=True,
        )

        # Generate — vLLM handles batching, KV cache, scheduling internally
        gen_start = time.time()
        llm = get_llm()
        outputs = llm.generate([augmented_prompt], params)
        gen_time = time.time() - gen_start

        output = outputs[0]
        generated_text = output.outputs[0].text
        speculative_accepted = _extract_speculative_accepted(output)

        # Feed acceptance back into the adaptive policy.
        _update_speculative_acceptance(profile_key, speculative_accepted)

        # Token counts (from vLLM)
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens

        # Evaluation score
        evaluation_score = OutputEvaluator.calculate_evaluation_score(generated_text)

        e2e_time = time.time() - _start_time

        # Save metrics
        save_latency_metrics(
            {"prompt": full_prompt, "job_id": request.job_id, "gene_id": request.gene_id,
             "max_new_tokens": request.max_new_tokens, "temperature": request.temperature,
             "top_p": request.top_p},
            e2e_time,
            gen_time,
            evaluation_score=evaluation_score,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            constraint_type=constraint_type,
            speculative_accepted=speculative_accepted,
        )

        print(
            f"[vLLM] Completed in {e2e_time:.2f}s "
            f"(gen={gen_time:.2f}s, {prompt_tokens}→{completion_tokens} tokens, "
            f"eval={evaluation_score})",
            flush=True,
        )

        return {
            "generated_text": generated_text,
            "response_time_sec": round(gen_time, 2),
            "_latency_sec": round(e2e_time, 4),
            "e2e_latency_sec": round(e2e_time, 4),
            "run_hash": RUN_HASH,
            "evaluationScore": evaluation_score,
            "batch_size": 1,
            "queue_wait_time_sec": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "constraint_type": constraint_type,
            "speculative_accepted": speculative_accepted,
            "vllm_num_speculative_tokens": chosen_spec_tokens,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "LLM API (vLLM) with Constraints & Speculation is running!",
        "version": "3.0",
        "constrained_decoding": CONSTRAINED_DECODING_ENABLED,
        "speculative_decoding": ENABLE_SPECULATIVE_DECODING,
    }


@app.on_event("startup")
async def startup_event():
    """Load the vLLM engine at server startup."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] ===== vLLM SERVER STARTUP =====", flush=True)
    get_llm()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] ===== vLLM SERVER READY =====", flush=True)
    print(f"[{timestamp}] Endpoints: /generate (POST), / (GET)", flush=True)
    if CONSTRAINED_DECODING_ENABLED:
        print(f"[{timestamp}] Constrained Decoding: ENABLED", flush=True)
    if ENABLE_SPECULATIVE_DECODING:
        print(f"[{timestamp}] Speculative Decoding: ENABLED ({VLLM_SPECULATIVE_METHOD})", flush=True)


print(f"[vLLM] Run Hash: {RUN_HASH}", flush=True)
print(f"[vLLM] Metrics: {METRICS_FILE}", flush=True)
