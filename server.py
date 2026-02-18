import time
import os
import asyncio
import json
import hashlib
import uuid
import re
import psutil
import GPUtil
import multiprocessing
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA multi-GPU
# This MUST be done before any CUDA operations (before importing vllm)
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading

from src.cfg.constants import *
from evaluator import OutputEvaluator

# vLLM backend (replaces HuggingFace transformers for continuous batching)
# Optional: install with uv sync --extra vllm
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = SamplingParams = None  # type: ignore
    VLLM_AVAILABLE = False

app = FastAPI(title="LLM API", version="1.0")

# Path To Local Large Language Model (from environment variable)
MODEL_PATH = os.getenv("MODEL_PATH", "/storage/ice-shared/vip-vvi/hf_models/models--google--gemma-7b-it")

# vLLM configuration
USE_VLLM = os.getenv("USE_VLLM", "true").lower() in ("true", "1", "yes")
GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "0"))  # 0 = auto from model config
TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))  # 2+ for 70B on multi-GPU

# Legacy batching params (used when USE_VLLM=false with HF fallback - not used with vLLM)
# vLLM uses continuous batching internally; these are kept for metrics/logging only
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BATCH_WAIT_TIME = float(os.getenv("BATCH_WAIT_TIME", "0.5"))

# Thread pool for running vLLM's synchronous generate() without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=1)

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
    "vllm_enabled": USE_VLLM,
    "backend": "vllm" if USE_VLLM else "huggingface",
    "backend_description": "vLLM (continuous batching)" if USE_VLLM else "HuggingFace (standard)",
    "batch_size": BATCH_SIZE,
    "batch_wait_time": BATCH_WAIT_TIME,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION if USE_VLLM else None,
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE if USE_VLLM else None,
    "requests": [],
    "memory_snapshots": []
}

with open(METRICS_FILE, 'w') as f:
    json.dump(metrics_metadata, f, indent=2)

# Global metrics tracking
_metrics_lock = threading.Lock()
_total_requests = 0
_session_start_time = time.time()


def get_memory_stats():
    """Get current CPU and GPU memory usage statistics."""
    try:
        # CPU memory
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / (1024 * 1024)
        system_memory = psutil.virtual_memory()
        system_memory_used_gb = system_memory.used / (1024 ** 3)
        system_memory_total_gb = system_memory.total / (1024 ** 3)
        system_memory_percent = system_memory.percent
        
        # GPU memory
        gpu_stats = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    "gpu_id": gpu.id,
                    "gpu_name": gpu.name,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_util_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2) if gpu.memoryTotal > 0 else 0,
                    "gpu_util_percent": gpu.load * 100
                })
        except Exception as e:
            gpu_stats = [{"error": str(e)}]
        
        return {
            "process_memory_mb": round(cpu_memory_mb, 2),
            "system_memory_used_gb": round(system_memory_used_gb, 2),
            "system_memory_total_gb": round(system_memory_total_gb, 2),
            "system_memory_percent": system_memory_percent,
            "gpus": gpu_stats
        }
    except Exception as e:
        return {"error": str(e)}


def save_memory_snapshot():
    """Save a memory snapshot to metrics file."""
    try:
        memory_stats = get_memory_stats()
        
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            **memory_stats
        }
        
        metrics["memory_snapshots"].append(snapshot)
        
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    except Exception as e:
        print(f"Error saving memory snapshot: {str(e)}")


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
    global _total_requests
    
    try:
        # Get memory stats at time of request
        memory_stats = get_memory_stats()
        
        # Read current metrics
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)

        # Add new request metrics
        request_metrics = {
            "timestamp": datetime.now().isoformat(),
            "job_id": request_data.get("job_id", "default"),
            "gene_id": request_data.get("gene_id", None),
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
            "generated_text": generated_text,
            "memory_at_request": memory_stats
        }

        metrics["requests"].append(request_metrics)
        
        # Increment global counter
        with _metrics_lock:
            _total_requests += 1

        # Write back to file
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"Error saving latency metrics: {str(e)}")


def calculate_metrics_summary():
    """Calculate summary statistics from all requests."""
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        
        requests = metrics.get("requests", [])
        
        if not requests:
            return {
                "total_requests": 0,
                "avg_latency_sec": 0,
                "avg_service_latency_sec": 0,
                "avg_batch_processing_sec": 0,
                "avg_queue_wait_sec": 0,
                "avg_evaluation_score": 0,
                "throughput_requests_per_sec": 0,
                "throughput_requests_per_min": 0
            }
        
        # Calculate averages
        latencies = [r["_latency_sec"] for r in requests if "_latency_sec" in r]
        batch_times = [r["batch_processing_time_sec"] for r in requests if "batch_processing_time_sec" in r]
        queue_times = [r["queue_wait_time_sec"] for r in requests if r.get("queue_wait_time_sec") is not None]
        eval_scores = [r["evaluation_score"] for r in requests if r.get("evaluation_score") is not None]

        # Service latency excludes queue wait time; prefer batch time if available
        service_latencies = []
        for r in requests:
            if r.get("batch_processing_time_sec") is not None:
                service_latencies.append(r["batch_processing_time_sec"])
            elif (r.get("_latency_sec") is not None) and (r.get("queue_wait_time_sec") is not None):
                e2e = float(r["_latency_sec"])  # type: ignore
                qwt = float(r["queue_wait_time_sec"])  # type: ignore
                service_latencies.append(max(e2e - qwt, 0.0))
            elif r.get("_latency_sec") is not None:
                service_latencies.append(r["_latency_sec"])
        
        # Calculate throughput
        session_duration = time.time() - _session_start_time
        throughput_per_sec = len(requests) / session_duration if session_duration > 0 else 0
        throughput_per_min = throughput_per_sec * 60
        
        return {
            "total_requests": len(requests),
            "avg_latency_sec": round(sum(latencies) / len(latencies), 4) if latencies else 0,
            "min_latency_sec": round(min(latencies), 4) if latencies else 0,
            "max_latency_sec": round(max(latencies), 4) if latencies else 0,
            "avg_service_latency_sec": round(sum(service_latencies) / len(service_latencies), 4) if service_latencies else 0,
            "min_service_latency_sec": round(min(service_latencies), 4) if service_latencies else 0,
            "max_service_latency_sec": round(max(service_latencies), 4) if service_latencies else 0,
            "avg_batch_processing_sec": round(sum(batch_times) / len(batch_times), 4) if batch_times else 0,
            "avg_queue_wait_sec": round(sum(queue_times) / len(queue_times), 4) if queue_times else 0,
            "avg_evaluation_score": round(sum(eval_scores) / len(eval_scores), 4) if eval_scores else 0,
            "throughput_requests_per_sec": round(throughput_per_sec, 4),
            "throughput_requests_per_min": round(throughput_per_min, 2),
            "session_duration_sec": round(session_duration, 2)
        }
    except Exception as e:
        print(f"Error calculating metrics summary: {str(e)}")
        return {}


def print_metrics_summary():
    """Print comprehensive metrics summary to stdout (appears in slurm.out)."""
    print("\n" + "="*80)
    print(f"{'METRICS SUMMARY':^80}")
    print("="*80)
    
    vllm_status = "ENABLED ✅" if USE_VLLM else "DISABLED ❌"
    backend = "vLLM (continuous batching)" if USE_VLLM else "HuggingFace (standard)"
    print(f"")
    print(f"  *** vLLM STATUS: {vllm_status} ***")
    print(f"")
    print(f"Backend: {backend}")
    print(f"Run ID: {RUN_ID}")
    print(f"Run Hash: {RUN_HASH}")
    print(f"Model: {MODEL_PATH}")
    
    # Get current memory stats
    memory = get_memory_stats()
    print("\n" + "-"*80)
    print(f"{'MEMORY USAGE':^80}")
    print("-"*80)
    print(f"Process Memory: {memory.get('process_memory_mb', 0):.2f} MB")
    print(f"System Memory: {memory.get('system_memory_used_gb', 0):.2f} GB / {memory.get('system_memory_total_gb', 0):.2f} GB ({memory.get('system_memory_percent', 0):.1f}%)")
    
    for gpu in memory.get('gpus', []):
        if 'error' not in gpu:
            print(f"GPU {gpu['gpu_id']} ({gpu['gpu_name']}): {gpu['memory_used_mb']:.0f} MB / {gpu['memory_total_mb']:.0f} MB ({gpu['memory_util_percent']:.1f}%)")
    
    # Get request metrics
    summary = calculate_metrics_summary()
    
    print("\n" + "-"*80)
    print(f"{'PERFORMANCE METRICS':^80}")
    print("-"*80)
    print(f"Total Requests: {summary.get('total_requests', 0)}")
    print(f"Session Duration: {summary.get('session_duration_sec', 0):.2f} seconds")
    print(f"")
    print(f"Latency (E2E):")
    print(f"  Average: {summary.get('avg_latency_sec', 0):.4f} sec")
    print(f"  Min:     {summary.get('min_latency_sec', 0):.4f} sec")
    print(f"  Max:     {summary.get('max_latency_sec', 0):.4f} sec")
    print(f"")
    print(f"Latency (Service, excl. queue):")
    print(f"  Average: {summary.get('avg_service_latency_sec', 0):.4f} sec")
    print(f"  Min:     {summary.get('min_service_latency_sec', 0):.4f} sec")
    print(f"  Max:     {summary.get('max_service_latency_sec', 0):.4f} sec")
    print(f"")
    print(f"Batch Processing Time: {summary.get('avg_batch_processing_sec', 0):.4f} sec (avg)")
    print(f"Queue Wait Time:       {summary.get('avg_queue_wait_sec', 0):.4f} sec (avg)")
    print(f"")
    print(f"Throughput:")
    print(f"  {summary.get('throughput_requests_per_sec', 0):.4f} requests/sec")
    print(f"  {summary.get('throughput_requests_per_min', 0):.2f} requests/min")
    print(f"")
    print(f"Model Performance (Evolution Fitness):")
    print(f"  Average Evaluation Score: {summary.get('avg_evaluation_score', 0):.4f}")
    
    print("="*80 + "\n")


async def periodic_metrics_reporter():
    """Background task to print metrics every 5 minutes."""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        print_metrics_summary()
        save_memory_snapshot()


class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 8192
    top_p: float = 0.8
    temperature: float = 0.7
    job_id: str | None = "default"
    gene_id: str | None = None


class VLLMModel:
    """vLLM-backed model with continuous batching."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Loading vLLM model at {MODEL_PATH}")
                    cls._instance = super(VLLMModel, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize vLLM engine with continuous batching."""
        start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] ===== vLLM MODEL LOADING STARTED =====")
        print(f"[{timestamp}] Model path: {MODEL_PATH}")
        print(f"[{timestamp}] GPU memory utilization: {GPU_MEMORY_UTILIZATION}")

        model_load_start = time.time()

        llm_kwargs = dict(
            model=MODEL_PATH,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        )
        if MAX_MODEL_LEN > 0:
            llm_kwargs["max_model_len"] = MAX_MODEL_LEN

        self.llm = LLM(**llm_kwargs)

        model_load_time = time.time() - model_load_start
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] vLLM model loaded in {model_load_time:.2f} seconds")

        # Async batching components
        self.request_queue = asyncio.Queue()
        self.batch_task = None
        self.batch_lock = asyncio.Lock()
        self.is_processing = False

        total_time = time.time() - start_time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] ===== vLLM LOADING COMPLETED =====")
        print(f"[{timestamp}] Total loading time: {total_time:.2f} seconds")
        print(f"[{timestamp}] Continuous batching enabled")

    async def start_batch_processor(self):
        """Start the batch processor if not already running."""
        async with self.batch_lock:
            if not self.is_processing:
                self.is_processing = True
                self.batch_task = asyncio.create_task(self._batch_processor())

    def _generate_sync(self, prompts: list, sampling_params: SamplingParams) -> list:
        """Synchronous vLLM generate (runs in thread pool)."""
        outputs = self.llm.generate(prompts, sampling_params)
        return [out.outputs[0].text for out in outputs]

    async def _batch_processor(self):
        """Process requests in batches using vLLM's continuous batching."""
        try:
            while True:
                batch = []
                futures = []

                try:
                    request, future = await self.request_queue.get()
                    batch.append(request)
                    futures.append(future)

                    batch_start_time = time.time()
                    while len(batch) < BATCH_SIZE and (time.time() - batch_start_time) < BATCH_WAIT_TIME:
                        try:
                            req, fut = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=max(0.01, BATCH_WAIT_TIME - (time.time() - batch_start_time))
                            )
                            batch.append(req)
                            futures.append(fut)
                        except asyncio.TimeoutError:
                            break

                    batch_size = len(batch)
                    batch_processing_start = time.time()
                    print(f"Processing batch of {batch_size} requests with vLLM")

                    prompts = [req["prompt"] for req in batch]
                    max_new_tokens = max(req["max_new_tokens"] for req in batch)
                    temperature = batch[0]["temperature"]
                    top_p = batch[0]["top_p"]

                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=1.1,
                    )

                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        _executor,
                        self._generate_sync,
                        prompts,
                        sampling_params,
                    )

                    response_time = round(time.time() - batch_processing_start, 2)

                    for i, (output_txt, future) in enumerate(zip(results, futures)):
                        queue_wait_time = batch_processing_start - batch[i].get("queue_start_time", batch_processing_start)
                        future.set_result({
                            "generated_text": output_txt,
                            "response_time_sec": response_time,
                            "batch_size": batch_size,
                            "queue_wait_time_sec": round(queue_wait_time, 4)
                        })
                        self.request_queue.task_done()

                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                    for _ in range(len(futures)):
                        self.request_queue.task_done()

                if self.request_queue.empty():
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            print("Batch processor cancelled")
        except Exception as e:
            print(f"Unexpected error in batch processor: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            async with self.batch_lock:
                self.is_processing = False

    async def generate(self, request_dict, queue_start_time=None):
        """Submit a request to the batch processor."""
        future = asyncio.Future()
        request_with_timing = {
            **request_dict,
            "queue_start_time": queue_start_time or time.time()
        }
        await self.request_queue.put((request_with_timing, future))
        await self.start_batch_processor()
        return await future


# HuggingFace fallback (when USE_VLLM=false)
class HFModel:
    """Legacy HuggingFace model - use USE_VLLM=true for vLLM backend."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    import torch
                    import transformers
                    print(f"Loading HuggingFace model at {MODEL_PATH}")
                    cls._instance = super(HFModel, cls).__new__(cls)
                    cls._instance._init_hf()
        return cls._instance

    def _init_hf(self):
        import torch
        import transformers
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=8192,
            repetition_penalty=1.1,
            do_sample=True,
            batch_size=BATCH_SIZE,
        )
        self.request_queue = asyncio.Queue()
        self.batch_task = None
        self.batch_lock = asyncio.Lock()
        self.is_processing = False

    async def start_batch_processor(self):
        async with self.batch_lock:
            if not self.is_processing:
                self.is_processing = True
                self.batch_task = asyncio.create_task(self._batch_processor_hf())

    async def _batch_processor_hf(self):
        try:
            while True:
                batch, futures = [], []
                try:
                    req, fut = await self.request_queue.get()
                    batch.append(req)
                    futures.append(fut)
                    batch_start = time.time()
                    while len(batch) < BATCH_SIZE and (time.time() - batch_start) < BATCH_WAIT_TIME:
                        try:
                            r, f = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=max(0.01, BATCH_WAIT_TIME - (time.time() - batch_start))
                            )
                            batch.append(r)
                            futures.append(f)
                        except asyncio.TimeoutError:
                            break
                    batch_size = len(batch)
                    prompts = [r["prompt"] for r in batch]
                    max_new_tokens = max(r["max_new_tokens"] for r in batch)
                    t0 = time.time()
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        _executor,
                        lambda: self.pipeline(
                            prompts,
                            max_new_tokens=max_new_tokens,
                            temperature=batch[0]["temperature"],
                            top_p=batch[0]["top_p"],
                        ),
                    )
                    rt = round(time.time() - t0, 2)
                    batch_start_time = time.time() - rt
                    for i, (res, fut) in enumerate(zip(results, futures)):
                        txt = res[0].get("generated_text", str(res))
                        qwt = batch_start_time - batch[i].get("queue_start_time", batch_start_time)
                        fut.set_result({
                            "generated_text": txt,
                            "response_time_sec": rt,
                            "batch_size": batch_size,
                            "queue_wait_time_sec": round(qwt, 4),
                        })
                        self.request_queue.task_done()
                except Exception as e:
                    print(f"HF batch error: {e}")
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(e)
                    for _ in futures:
                        self.request_queue.task_done()
                if self.request_queue.empty():
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            async with self.batch_lock:
                self.is_processing = False

    async def generate(self, request_dict, queue_start_time=None):
        future = asyncio.Future()
        await self.request_queue.put(({**request_dict, "queue_start_time": queue_start_time or time.time()}, future))
        await self.start_batch_processor()
        return await future


def get_model():
    """Return the active model backend."""
    if USE_VLLM:
        if not VLLM_AVAILABLE:
            print(
                "[WARN] USE_VLLM=true but vllm not installed. "
                "Install with: uv sync --extra vllm. Falling back to HuggingFace."
            )
            return HFModel()
        return VLLMModel()
    return HFModel()


@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Submits LLMRequest to the local model (vLLM or HuggingFace backend).
    """
    _start_time = time.time()

    try:
        system_prompt = os.getenv("SYSTEM_PROMPT", """Output a single fenced code block with runnable Python code and nothing else.
Do not include explanations, comments outside the code block, or extra code fences.
Begin with ```python and end with ```. If you cannot comply, output exactly FAIL.

""")

        full_prompt = system_prompt + request.prompt
        request_dict = {
            "prompt": full_prompt,
            "max_new_tokens": request.max_new_tokens,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "job_id": request.job_id,
            "gene_id": request.gene_id,
        }

        model = get_model()
        queue_start_time = time.time()
        print(f"Request received at {time.strftime('%H:%M:%S', time.localtime(_start_time))} [Job: {request.job_id}, Gene: {request.gene_id}]")

        result = await model.generate(request_dict, queue_start_time)
        e2e_time = time.time() - _start_time

        batch_processing_time = result.get("response_time_sec", 0)
        batch_size = result.get("batch_size", 1)
        queue_wait_time = result.get("queue_wait_time_sec", 0)
        generated_text = result.get("generated_text", "")
        evaluation_score = OutputEvaluator.calculate_evaluation_score(generated_text)
        generated_text_for_logging = strip_thinking_tokens(generated_text)

        save_latency_metrics(
            request_dict,
            e2e_time,
            batch_processing_time,
            batch_size,
            queue_wait_time,
            evaluation_score,
            prompt=request.prompt,
            generated_text=generated_text_for_logging,
        )

        print(f"Request completed in {e2e_time:.2f}s (E2E), {batch_processing_time:.2f}s (batch), score: {evaluation_score}")

        result["_latency_sec"] = round(e2e_time, 4)
        result["e2e_latency_sec"] = round(e2e_time, 4)
        result["run_hash"] = RUN_HASH
        result["evaluationScore"] = evaluation_score

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "LLM API is running!", "backend": "vllm" if USE_VLLM else "huggingface"}


@app.on_event("startup")
async def startup_event():
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"\n{'='*80}")
    print(f"[{timestamp}] ===== SERVER STARTUP INITIATED =====")
    print(f"{'='*80}")
    print(f"")
    print(f"  *** vLLM STATUS: {'ENABLED ✅' if USE_VLLM else 'DISABLED ❌'} ***")
    print(f"  Backend: {'vLLM (continuous batching)' if USE_VLLM else 'HuggingFace (standard)'}")
    print(f"  Model: {MODEL_PATH}")
    print(f"")
    
    # Load model
    model = get_model()
    
    # Save initial memory snapshot
    save_memory_snapshot()
    
    # Print initial metrics
    print_metrics_summary()
    
    # Start periodic metrics reporter
    asyncio.create_task(periodic_metrics_reporter())
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[{timestamp}] ===== SERVER STARTUP COMPLETE =====")
    print(f"[{timestamp}] Endpoints: /generate (POST), / (GET)")
    print(f"[{timestamp}] Periodic metrics will be printed every 5 minutes")


@app.on_event("shutdown")
async def shutdown_event():
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"\n{'='*80}")
    print(f"[{timestamp}] ===== SERVER SHUTDOWN INITIATED =====")
    print(f"{'='*80}")
    
    # Save final memory snapshot
    save_memory_snapshot()
    
    # Print final metrics summary
    print("\n" + "="*80)
    print(f"{'FINAL METRICS SUMMARY':^80}")
    print("="*80)
    vllm_status = "ENABLED ✅" if USE_VLLM else "DISABLED ❌"
    print(f"")
    print(f"  *** vLLM STATUS: {vllm_status} ***")
    print(f"")
    print_metrics_summary()
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"{'='*80}")
    print(f"[{timestamp}] ===== SERVER SHUTDOWN COMPLETE =====")
    print(f"[{timestamp}] Metrics saved to: {METRICS_FILE}")
    print(f"{'='*80}\n")


print(f"\n{'='*80}")
print(f"{'SERVER CONFIGURATION':^80}")
print(f"{'='*80}")
print(f"  vLLM Status: {'ENABLED ✅' if USE_VLLM else 'DISABLED ❌'}")
print(f"  Backend: {'vLLM (continuous batching)' if USE_VLLM else 'HuggingFace (standard)'}")
print(f"  Run Hash: {RUN_HASH}")
print(f"  Metrics File: {METRICS_FILE}")
print(f"{'='*80}\n")
