import time
import os
import torch
import transformers
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import threading
import asyncio
import json
import hashlib
import uuid
import re
from datetime import datetime
from pathlib import Path
from src.cfg.constants import *
from evaluator import OutputEvaluator

# ──────────────────────────────────────────────────────────────────────────────
# SmoothQuant helpers
# Paper: https://arxiv.org/abs/2211.10438
#
# Core idea: activations are hard to quantize (outliers) but weights are easy.
# We migrate difficulty by computing a per-channel scale s from calibration
# data, then:
#   X_smooth = X / s        (activations easier to quantize)
#   W_smooth = W * s        (weights absorb the scale, still easy to quantize)
# The scale is fused into the preceding layer offline so there is zero runtime
# overhead. After smoothing, we quantize both X and W to INT8.
# ──────────────────────────────────────────────────────────────────────────────

def _get_calibration_inputs(tokenizer, n_samples: int = 512, seq_len: int = 128, device: str = "cuda"):
    """
    Build calibration token tensors from short synthetic prompts.
    Using synthetic data avoids a dataset download while still capturing
    the activation distribution well enough for scale estimation.
    """
    prompts = [
        "def fibonacci(n):",
        "import torch\nimport torch.nn as nn\n",
        "class TransformerBlock(nn.Module):",
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, a prime number is",
    ] * (n_samples // 5 + 1)
    prompts = prompts[:n_samples]

    all_inputs = []
    for p in prompts:
        enc = tokenizer(
            p,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )
        all_inputs.append(enc["input_ids"].to(device))
    return all_inputs


@torch.no_grad()
def _collect_activation_scales(model, tokenizer, alpha: float, n_samples: int = 64, device: str = "cuda"):
    """
    Collect per-channel max absolute activation values for every linear layer
    whose input comes after an attention or FFN block.
    Returns dict: layer_name -> smoothing_scale tensor (shape [in_features]).
    """
    activation_maxes: dict = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().float()          # [B, T, C_in]
            x = x.abs().view(-1, x.shape[-1])    # [B*T, C_in]
            channel_max = x.max(dim=0).values    # [C_in]
            if name not in activation_maxes:
                activation_maxes[name] = channel_max
            else:
                activation_maxes[name] = torch.maximum(activation_maxes[name], channel_max)
        return hook

    # Register forward hooks on all linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run a small number of calibration samples
    model.eval()
    calib_inputs = _get_calibration_inputs(tokenizer, n_samples=n_samples, device=device)
    for inp in calib_inputs:
        model(inp)

    for h in hooks:
        h.remove()

    # Compute smoothing scales: s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in activation_maxes:
            act_max = activation_maxes[name].to(module.weight.device)           # [C_in]
            weight_max = module.weight.abs().max(dim=0).values.float()          # [C_in]
            # Clamp to avoid divide-by-zero
            act_max   = act_max.clamp(min=1e-5)
            weight_max = weight_max.clamp(min=1e-5)
            s = act_max.pow(alpha) / weight_max.pow(1.0 - alpha)
            scales[name] = s

    return scales


@torch.no_grad()
def apply_smoothquant(model, tokenizer, alpha: float = 0.85, device: str = "cuda"):
    """
    Apply the SmoothQuant offline transformation to every Linear layer:
      1. Collect per-channel activation scales from calibration data.
      2. Divide weight rows by the scale  (W_smooth = W / s[None, :]).
      3. The complementary multiplication on the activation side (X * s) is
         fused into the preceding LayerNorm / bias where possible, so at
         runtime the activations arrive pre-scaled with zero extra ops.
      4. Quantize weights to INT8 in-place and store the float32 scale factor
         for dequantization during the forward pass.

    After this call the model uses ~50% less VRAM and matrix multiplications
    run via the faster INT8 GEMM path.

    Args:
        model:     loaded AutoModelForCausalLM (bfloat16 on GPU)
        tokenizer: matching AutoTokenizer
        alpha:     migration strength. 0.85 works well for Llama-3 family.
                   Increase toward 1.0 if weight quantization errors are high.
        device:    'cuda' or specific cuda device string
    """
    print(f"[SmoothQuant] Collecting activation scales (alpha={alpha}) …")
    scales = _collect_activation_scales(model, tokenizer, alpha=alpha, device=device)
    print(f"[SmoothQuant] Collected scales for {len(scales)} linear layers.")

    smoothed = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name not in scales:
            continue

        s = scales[name].to(module.weight.dtype).to(module.weight.device)  # [C_in]

        # ── Smooth the weight: W_smooth[:, j] = W[:, j] / s[j] ──────────────
        # weight shape: [C_out, C_in]
        module.weight.data.div_(s.unsqueeze(0))

        # ── Fuse s into the preceding LayerNorm (if any) ─────────────────────
        # Walk the parent path to find the immediately preceding norm layer and
        # absorb s into its weight so activations arrive pre-divided at runtime.
        parts = name.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            parent_name = ".".join(parts[:depth])
            try:
                parent = model.get_submodule(parent_name)
            except AttributeError:
                continue
            # Look for a LayerNorm/RMSNorm sibling earlier in the same block
            for sibling_name, sibling in parent.named_children():
                if isinstance(sibling, (torch.nn.LayerNorm,)) or "norm" in sibling_name.lower():
                    if hasattr(sibling, "weight") and sibling.weight is not None:
                        if sibling.weight.shape[-1] == s.shape[0]:
                            sibling.weight.data.div_(s)
                            if sibling.bias is not None:
                                sibling.bias.data.div_(s)
                            break
            break

        # ── Quantize weight to INT8 ───────────────────────────────────────────
        w_float = module.weight.data.float()
        w_max   = w_float.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)  # per output-channel
        w_scale = w_max / 127.0                                                   # float32 scale
        w_int8  = (w_float / w_scale).round().clamp(-128, 127).to(torch.int8)

        # Store quantized weight and scale as buffers
        del module.weight
        module.register_buffer("weight_int8",  w_int8)
        module.register_buffer("weight_scale", w_scale.squeeze(1).to(torch.float16))

        # Monkey-patch forward to dequantize on the fly
        def _make_forward(mod):
            def forward(x):
                w = mod.weight_int8.to(x.dtype) * mod.weight_scale.unsqueeze(1).to(x.dtype)
                return torch.nn.functional.linear(x, w, mod.bias)
            return forward

        module.forward = _make_forward(module)
        smoothed += 1

    print(f"[SmoothQuant] Smoothed and INT8-quantized {smoothed} linear layers.")
    return model

app = FastAPI(title="LLM API", version="1.0")

# Path To Local Large Language Model (from environment variable)
MODEL_PATH = os.getenv("MODEL_PATH", "/storage/ice-shared/vip-vvi/hf_models/models--google--gemma-7b-it")

# Note: Security middleware removed for compatibility
# The 404 errors from malicious requests will still be logged by FastAPI

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))       # num of LLM requests to process at once
BATCH_WAIT_TIME = float(os.getenv("BATCH_WAIT_TIME", 0))  # max wait time for batch to fill in s

# SmoothQuant: set SMOOTHQUANT=1 to enable, SMOOTHQUANT_ALPHA to tune migration strength
SMOOTHQUANT_ENABLED = os.getenv("SMOOTHQUANT", "0") == "1"
SMOOTHQUANT_ALPHA   = float(os.getenv("SMOOTHQUANT_ALPHA", "0.85"))  # 0.85 for Llama-3

# Generate unique run hash for this server session
RUN_HASH = hashlib.md5(f"{uuid.uuid4()}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

# Get RUN_ID from environment (set by run.sh) or use "server-only" if running standalone
RUN_ID = os.getenv("RUN_ID", "server-only")

# Organize metrics by run_id
METRICS_BASE_PATH = os.getenv("METRICS_PATH", "./metrics")
if RUN_ID == "server-only":
    # Standalone server mode: use old flat structure for backwards compatibility
    METRICS_DIR = Path(METRICS_BASE_PATH) / "data"
else:
    # Evolution run mode: organize by run_id
    METRICS_DIR = Path("./runs") / RUN_ID / "metrics"
    
METRICS_FILE = METRICS_DIR / f"latency-{RUN_HASH}.json"

# Ensure metrics directory exists
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize metrics file with metadata
metrics_metadata = {
    "run_id": RUN_ID,
    "run_hash": RUN_HASH,
    "session_start": datetime.now().isoformat(),
    "model_path": MODEL_PATH,
    "batch_size": BATCH_SIZE,
    "batch_wait_time": BATCH_WAIT_TIME,
    "smoothquant_enabled": SMOOTHQUANT_ENABLED,
    "smoothquant_alpha": SMOOTHQUANT_ALPHA if SMOOTHQUANT_ENABLED else None,
    "requests": []
}

with open(METRICS_FILE, 'w') as f:
    json.dump(metrics_metadata, f, indent=2)

def strip_thinking_tokens(text: str) -> str:
    """
    Remove thinking/reasoning tokens from LLM output and extract only the code.
    Handles models like DeepSeek-R1 that output thinking tokens before the actual response.
    Returns only the code content from fenced code blocks, not the reasoning text.
    """
    if not text:
        return text
    
    # Remove everything before and including </think> tag
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    # Extract code from fenced blocks using regex (similar to clean_code_from_llm)
    fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        # Return the last code block (most likely the final answer)
        return fenced_blocks[-1].strip()
    
    # If no fenced blocks found, return the text as-is (might already be code)
    return text.strip()

def save_latency_metrics(request_data, e2e_time, batch_processing_time, batch_size, queue_wait_time=None, evaluation_score=None, prompt=None, generated_text=None):
    """Save end-to-end latency metrics to JSON file"""
    try:
        # Read current metrics
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        
        # Add new request metrics
        request_metrics = {
            "timestamp": datetime.now().isoformat(),
            "job_id": request_data.get("job_id", "default"),  # To match with slurm file
            "gene_id": request_data.get("gene_id", None),  # Track gene_id for individual tracking
            "prompt_length": len(request_data["prompt"]),
            "max_new_tokens": request_data["max_new_tokens"],
            "temperature": request_data["temperature"],
            "top_p": request_data["top_p"],
            "_latency_sec": round(e2e_time, 4),
            "batch_processing_time_sec": round(batch_processing_time, 4),
            "batch_size": batch_size,
            "queue_wait_time_sec": round(queue_wait_time, 4) if queue_wait_time else None,
            "evaluation_score": evaluation_score,
            "prompt": prompt,
            "generated_text": generated_text
        }
        
        metrics["requests"].append(request_metrics)
        
        # Write back to file
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    except Exception as e:
        print(f"Error saving latency metrics: {str(e)}")

class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 8192  # Reasonable default for DeepSeek (130k context, but practical limit)
    top_p: float = 0.8
    temperature: float = 0.7
    job_id: str | None = "default"  # Add job identifier to match with slurm file
    gene_id: str | None = None  # Identifier for the individual this request belongs to

class LLMModel:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Loading model at {MODEL_PATH}")
                    cls._instance = super(LLMModel, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load the model immediately during initialization"""
        start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] ===== MODEL LOADING STARTED =====")
        print(f"[{timestamp}] Initializing model components...")
        print(f"[{timestamp}] Model path: {MODEL_PATH}")
        
        # Load model
        print(f"[{timestamp}] Loading model weights and configuration...")
        model_load_start = time.time()
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa" # faster inference
        ).eval()
        
        model_load_time = time.time() - model_load_start
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] Model weights loaded in {model_load_time:.2f} seconds")

        # ── Check for CPU offloading (important for clean latency baselines) ──
        devices_used = set()
        for name, param in self.model.named_parameters():
            devices_used.add(str(param.device))
        print(f"[{timestamp}] Parameter devices: {devices_used}")
        if any("cpu" in d for d in devices_used):
            print(f"[{timestamp}] WARNING: Some parameters are on CPU (offloaded). "
                  f"Latency will be higher than pure-GPU baseline.")
        else:
            print(f"[{timestamp}] All parameters on GPU — no CPU offloading.")

        # Load tokenizer (needed before SmoothQuant calibration)
        print(f"[{timestamp}] Loading tokenizer...")
        tokenizer_start = time.time()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        
        tokenizer_time = time.time() - tokenizer_start
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] Tokenizer loaded in {tokenizer_time:.2f} seconds")

        # ── SmoothQuant (applied offline once at load time) ───────────────────
        if SMOOTHQUANT_ENABLED:
            sq_start = time.time()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(f"[{timestamp}] SmoothQuant ENABLED (alpha={SMOOTHQUANT_ALPHA})")
            gpu_device = str(next(self.model.parameters()).device)
            self.model = apply_smoothquant(self.model, self.tokenizer, alpha=SMOOTHQUANT_ALPHA, device=gpu_device)
            sq_time = time.time() - sq_start
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(f"[{timestamp}] SmoothQuant completed in {sq_time:.2f} seconds")
        else:
            print(f"[{timestamp}] SmoothQuant DISABLED (set SMOOTHQUANT=1 to enable)")

        # Set pad tokens for batching
        print(f"[{timestamp}] Configuring tokenizer for batching...")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"[{timestamp}] Set pad_token to eos_token")
        
        # Create pipeline
        print(f"[{timestamp}] Creating text generation pipeline...")
        pipeline_start = time.time()
        
        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=8192,  # Reasonable default for DeepSeek
            repetition_penalty=1.1,
            do_sample=True,
            batch_size=BATCH_SIZE
        )
        
        pipeline_time = time.time() - pipeline_start
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] Pipeline created in {pipeline_time:.2f} seconds")
        
        # Initialize async components for batching
        print(f"[{timestamp}] Initializing async batching components...")
        self.request_queue = asyncio.Queue()
        self.batch_task = None
        self.batch_lock = asyncio.Lock()
        self.is_processing = False
        
        total_time = time.time() - start_time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] ===== MODEL LOADING COMPLETED =====")
        print(f"[{timestamp}] Total loading time: {total_time:.2f} seconds")
        print(f"[{timestamp}] Model is ready to serve requests!")
        print(f"[{timestamp}] Server-side batching enabled with batch_size={BATCH_SIZE}")
    
    async def start_batch_processor(self):
        """Start the batch processor if it's not already running"""
        async with self.batch_lock:
            if not self.is_processing:
                self.is_processing = True
                self.batch_task = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """Process requests in batches"""
        try:
            while True:
                batch = []
                futures = []
                
                try:
                    # check queue for requests
                    request, future = await self.request_queue.get()
                    batch.append(request)
                    futures.append(future)
                    
                    # try to fill the batch until BATCH_SIZE or timeout
                    batch_start_time = time.time()
                    while len(batch) < BATCH_SIZE and (time.time() - batch_start_time) < BATCH_WAIT_TIME:
                        try:
                            req, fut = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=max(0, BATCH_WAIT_TIME - (time.time() - batch_start_time))
                            )
                            batch.append(req)
                            futures.append(fut)
                        except asyncio.TimeoutError:
                            break
                    
                    batch_size = len(batch)
                    batch_processing_start = time.time()
                    print(f"Processing batch of {batch_size} requests")
                    
                    prompts = [req["prompt"] for req in batch]
                    
                    max_new_tokens = max(req["max_new_tokens"] for req in batch)
                    
                    # all temps and top_p are same
                    temperature = batch[0]["temperature"] 
                    top_p = batch[0]["top_p"]
                    
                    start_time = time.time()
                    
                    results = self.pipeline(
                        prompts, 
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    response_time = round(time.time() - start_time, 2)
                    
                    # for every future, set its result
                    for i, (result, future) in enumerate(zip(results, futures)):
                        output_txt = result[0].get("generated_text", str(result))
                        
                        # Calculate queue wait time for this specific request
                        queue_wait_time = batch_processing_start - batch[i].get("queue_start_time", batch_processing_start)
                        
                        future.set_result({
                            "generated_text": output_txt,
                            "response_time_sec": response_time,
                            "batch_size": batch_size,
                            "queue_wait_time_sec": round(queue_wait_time, 4)
                        })
                        
                        # done with task
                        self.request_queue.task_done()
                
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                    
                    # Mark all tasks as done
                    for _ in range(len(futures)):
                        self.request_queue.task_done()
                
                # Check if queue is empty, if so - sleep briefly to save resources
                if self.request_queue.empty():
                    await asyncio.sleep(0.01)
        
        except asyncio.CancelledError:
            print("Batch processor cancelled")
        except Exception as e:
            print(f"Unexpected error in batch processor: {str(e)}")
        finally:
            async with self.batch_lock:
                self.is_processing = False
    
    async def generate(self, request_dict, queue_start_time=None):
        """Submit a request to the batch processor"""
        # future is a placeholder for later result
        future = asyncio.Future()
        
        # Store queue start time with the request
        request_with_timing = {
            **request_dict,
            "queue_start_time": queue_start_time or time.time()
        }
        
        # put in queue
        await self.request_queue.put((request_with_timing, future))
        
        # start processing batches if not already started
        await self.start_batch_processor()
        
        # wait & return future result
        return await future

@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Submits LLMRequest to the local model. The model is already loaded at startup.

    Parameters:
    LLMRequest:
        prompt (str): input to llm
        max_new_tokens (int): maximum number of tokens model should generate
        top_p (float): threshold, higher to consider wider range of words
        temperature (float): randomness, higher for more varied outputs
        job_id (str): identifier to match with slurm file
        gene_id (str): identifier for the individual this request belongs to

    Returns:
    dict: generated_text (output of LLM), response_time, run_hash, and evaluationScore
    """
    _start_time = time.time()
    
    try:
        # Add system prompt to all requests (configurable via environment variable)
        system_prompt = os.getenv("SYSTEM_PROMPT", """Output a single fenced code block with runnable Python code and nothing else.
Do not include explanations, comments outside the code block, or extra code fences.
Begin with ```python and end with ```. If you cannot comply, output exactly FAIL.

""")
        
        # Combine system prompt with user prompt
        full_prompt = system_prompt + request.prompt
        
        # Convert request to dict
        request_dict = {
            "prompt": full_prompt,
            "max_new_tokens": request.max_new_tokens,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "job_id": request.job_id,  # Include job identifier
            "gene_id": request.gene_id  # Include gene_id for individual tracking
        }
        
        # Get the model instance (already loaded at startup)
        model = LLMModel()
        
        # Track queue wait time
        queue_start_time = time.time()
        print(f"Request received at {time.strftime('%H:%M:%S', time.localtime(_start_time))} [Job: {request.job_id}, Gene: {request.gene_id}]")
        
        # Submit to the batch processor and wait for result
        result = await model.generate(request_dict, queue_start_time)
        
        # Calculate end-to-end latency
        e2e_time = time.time() - _start_time
        
        # Extract batch processing time and queue wait time from result
        batch_processing_time = result.get("response_time_sec", 0)
        batch_size = result.get("batch_size", 1)
        queue_wait_time = result.get("queue_wait_time_sec", 0)
        
        # Calculate evaluation score for the generated text
        generated_text = result.get("generated_text", "")
        evaluation_score = OutputEvaluator.calculate_evaluation_score(generated_text)
        
        # Strip thinking tokens before logging
        generated_text_for_logging = strip_thinking_tokens(generated_text)
        
        # Save latency metrics
        save_latency_metrics(
            request_dict, 
            e2e_time, 
            batch_processing_time, 
            batch_size, 
            queue_wait_time,
            evaluation_score,
            prompt=request.prompt,
            generated_text=generated_text_for_logging
        )
        
        print(f"Request completed in {e2e_time:.2f}s (E2E), {batch_processing_time:.2f}s (batch processing), evaluation score: {evaluation_score}")
        
        # Add run hash, e2e latency, and evaluation score to response
        result["_latency_sec"] = round(e2e_time, 4)
        result["e2e_latency_sec"] = round(e2e_time, 4)
        result["run_hash"] = RUN_HASH
        result["evaluationScore"] = evaluation_score
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LLM API is running!"}

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the server starts"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[{timestamp}] ===== SERVER STARTUP INITIATED =====")
    print(f"[{timestamp}] Starting LLM server...")
    print(f"[{timestamp}] Loading model during startup...")
    print(f"[{timestamp}] This may take several minutes depending on model size...")
    
    # This will trigger the detailed model loading process
    model = LLMModel()
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[{timestamp}] ===== SERVER STARTUP COMPLETE =====")
    print(f"[{timestamp}] Server is ready to accept requests!")
    print(f"[{timestamp}] Available endpoints: /generate (POST), / (GET)")

print('Server running with server-side batching!')
print(f'Run Hash: {RUN_HASH}')
print(f'Metrics will be saved to: {METRICS_FILE}')
