"""
vLLM-backed inference server with PagedAttention KV cache optimization.

Replaces HuggingFace Transformers pipeline with vLLM engine for:
- PagedAttention: Near-zero KV cache waste (95%+ utilization vs ~30% before)
- Continuous batching: Requests start/finish independently (no wait-then-process)
- Prefix caching: System prompt KV cache computed once, reused across all requests
- Tensor parallelism: Proper multi-GPU sharding (vs naive layer splitting)

API contract is identical to server_baseline.py — drop-in replacement.
"""

import time
import os
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path

# Force vLLM to use spawn to prevent CUDA initialization crashes
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.cfg.constants import *
from evaluator import OutputEvaluator

# ---------------------------------------------------------------------------
# App & Config
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM API (vLLM)", version="2.0")

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
# 0.95 is needed for 70B models on 2x80GB GPUs to leave room for KV cache after loading weights (~70GB/GPU)
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "bfloat16")
_vllm_quant_raw = os.getenv("VLLM_QUANTIZATION", "").strip()
# Treat common "off" values as unset so `.env` can use `false` safely.
if _vllm_quant_raw.lower() in ("", "none", "false", "0", "no"):
    VLLM_QUANTIZATION = None
else:
    VLLM_QUANTIZATION = _vllm_quant_raw
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", "true").lower() in (
    "true", "1", "yes",
)

# Default system prompt for code-gen requests.
# PageIndex and other non-code requests send use_system_prompt=False to skip it.
DEFAULT_CODE_GEN_PROMPT = (
    "Output a single fenced code block with runnable Python code and nothing else.\n"
    "Do not include explanations, comments outside the code block, or extra code fences.\n"
    "Begin with ```python and end with ```. If you cannot comply, output exactly FAIL.\n\n"
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
    "batch_size": "continuous",  # backward-compat: vLLM uses continuous batching
    "batch_wait_time": 0,        # backward-compat: no manual wait in vLLM
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "max_model_len": MAX_MODEL_LEN,
    "vllm_dtype": VLLM_DTYPE,
    "vllm_quantization": VLLM_QUANTIZATION,
    "prefix_caching": ENABLE_PREFIX_CACHING,
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
            "batch_processing_time_sec": round(generation_time, 4),  # backward-compat alias
            "generation_time_sec": round(generation_time, 4),
            "batch_size": 1,  # backward-compat: vLLM batches internally
            "queue_wait_time_sec": round(queue_wait_time, 4) if queue_wait_time is not None else None,
            "evaluation_score": evaluation_score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        metrics["requests"].append(request_metrics)

        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"Error saving latency metrics: {e}", flush=True)


# ---------------------------------------------------------------------------
# Request schema (identical to server_baseline.py)
# ---------------------------------------------------------------------------
class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    top_p: float = 0.8
    temperature: float = 0.7
    job_id: str | None = "default"
    gene_id: str | None = None
    use_system_prompt: bool = True
    system_prompt_override: str | None = None
    do_sample: bool = True
    repetition_penalty: float = 1.1


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

        start = time.time()
        llm_kwargs = dict(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype=VLLM_DTYPE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            trust_remote_code=True,
        )
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
    Drop-in replacement for server_baseline.py /generate endpoint.

    Same request/response contract: accepts LLMRequest, returns dict with
    generated_text, _latency_sec, run_hash, evaluationScore.
    """
    from vllm import SamplingParams

    _start_time = time.time()

    try:
        # Resolve system prompt (mirrors server.py logic)
        if request.use_system_prompt:
            if request.system_prompt_override:
                system_prompt = request.system_prompt_override
            else:
                system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_CODE_GEN_PROMPT)
        else:
            system_prompt = None

        # Build prompt with chat template
        llm = get_llm()
        tokenizer = llm.get_tokenizer()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without a chat template
            full_prompt = (system_prompt + "\n\n" + request.prompt) if system_prompt else request.prompt

        # Clamp max_tokens so prompt + output fits within MAX_MODEL_LEN.
        # vLLM rejects requests where prompt_len + max_tokens > max_model_len.
        prompt_token_ids = tokenizer.encode(full_prompt)
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

        # vLLM sampling parameters
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=safe_max_tokens,
            repetition_penalty=request.repetition_penalty,
        )

        print(
            f"[vLLM] Request received [Job: {request.job_id}, Gene: {request.gene_id}]",
            flush=True,
        )

        # Generate — vLLM handles batching, KV cache, scheduling internally
        gen_start = time.time()
        outputs = llm.generate([full_prompt], params)
        gen_time = time.time() - gen_start

        output = outputs[0]
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        # Token counts (from vLLM — no extra tokenizer call needed)
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens

        # Evaluation score (skip for PageIndex — it generates JSON, not Python code)
        if request.gene_id == "pageindex":
            evaluation_score = None
        else:
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
        )

        print(
            f"[vLLM] Completed in {e2e_time:.2f}s "
            f"(gen={gen_time:.2f}s, {prompt_tokens}→{completion_tokens} tokens, "
            f"eval={evaluation_score})",
            flush=True,
        )

        return {
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "response_time_sec": round(gen_time, 2),
            "_latency_sec": round(e2e_time, 4),
            "e2e_latency_sec": round(e2e_time, 4),
            "run_hash": RUN_HASH,
            "evaluationScore": evaluation_score,
            "batch_size": 1,  # vLLM handles batching internally
            "queue_wait_time_sec": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "LLM API (vLLM) is running!"}


@app.on_event("startup")
async def startup_event():
    """Load the vLLM engine at server startup."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] ===== vLLM SERVER STARTUP =====", flush=True)
    get_llm()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] ===== vLLM SERVER READY =====", flush=True)
    print(f"[{timestamp}] Endpoints: /generate (POST), / (GET)", flush=True)


print(f"[vLLM] Run Hash: {RUN_HASH}", flush=True)
print(f"[vLLM] Metrics: {METRICS_FILE}", flush=True)
