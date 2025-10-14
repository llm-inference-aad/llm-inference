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
from datetime import datetime
from pathlib import Path
from src.cfg.constants import *
from evaluator import OutputEvaluator

app = FastAPI(title="LLM API", version="1.0")

# Path To Local Large Language Model (from environment variable)
MODEL_PATH = os.getenv("MODEL_PATH", "/storage/ice-shared/vip-vvi/hf_models/models--google--gemma-7b-it")

# Note: Security middleware removed for compatibility
# The 404 errors from malicious requests will still be logged by FastAPI

BATCH_SIZE = 1  # num of LLM requests to process at once
BATCH_WAIT_TIME = 0  # max wait time for batch to fill in s

# Generate unique run hash for this server session
RUN_HASH = hashlib.md5(f"{uuid.uuid4()}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
METRICS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "metrics" / "data"
METRICS_FILE = METRICS_DIR / f"e2e-latency-{RUN_HASH}.json"

# Ensure metrics directory exists
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize metrics file with metadata
metrics_metadata = {
    "run_hash": RUN_HASH,
    "session_start": datetime.now().isoformat(),
    "model_path": MODEL_PATH,
    "batch_size": BATCH_SIZE,
    "batch_wait_time": BATCH_WAIT_TIME,
    "requests": []
}

with open(METRICS_FILE, 'w') as f:
    json.dump(metrics_metadata, f, indent=2)

def save_latency_metrics(request_data, e2e_time, batch_processing_time, batch_size, queue_wait_time=None, evaluation_score=None):
    """Save end-to-end latency metrics to JSON file"""
    try:
        # Read current metrics
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        
        # Add new request metrics
        request_metrics = {
            "timestamp": datetime.now().isoformat(),
            "job_id": request_data.get("job_id", "default"),  # To match with slurm file
            "prompt_length": len(request_data["prompt"]),
            "max_new_tokens": request_data["max_new_tokens"],
            "temperature": request_data["temperature"],
            "top_p": request_data["top_p"],
            "e2e_latency_sec": round(e2e_time, 4),
            "batch_processing_time_sec": round(batch_processing_time, 4),
            "batch_size": batch_size,
            "queue_wait_time_sec": round(queue_wait_time, 4) if queue_wait_time else None,
            "evaluation_score": evaluation_score
        }
        
        metrics["requests"].append(request_metrics)
        
        # Write back to file
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    except Exception as e:
        print(f"Error saving latency metrics: {str(e)}")

class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100000
    top_p: float = 0.8
    temperature: float = 0.7
    job_id: str = "default"  # Add job identifier to match with slurm file

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

        # Load tokenizer
        print(f"[{timestamp}] Loading tokenizer...")
        tokenizer_start = time.time()
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        
        tokenizer_time = time.time() - tokenizer_start
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] Tokenizer loaded in {tokenizer_time:.2f} seconds")
        
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
            max_new_tokens=100000,
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

    Returns:
    dict: generated_text (output of LLM), response_time, run_hash, and evaluationScore
    """
    e2e_start_time = time.time()
    
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
            "job_id": request.job_id  # Include job identifier
        }
        
        # Get the model instance (already loaded at startup)
        model = LLMModel()
        
        # Track queue wait time
        queue_start_time = time.time()
        print(f"Request received at {time.strftime('%H:%M:%S', time.localtime(e2e_start_time))} [Job: {request.job_id}]")
        
        # Submit to the batch processor and wait for result
        result = await model.generate(request_dict, queue_start_time)
        
        # Calculate end-to-end latency
        e2e_time = time.time() - e2e_start_time
        
        # Extract batch processing time and queue wait time from result
        batch_processing_time = result.get("response_time_sec", 0)
        batch_size = result.get("batch_size", 1)
        queue_wait_time = result.get("queue_wait_time_sec", 0)
        
        # Calculate evaluation score for the generated text
        generated_text = result.get("generated_text", "")
        evaluation_score = OutputEvaluator.calculate_evaluation_score(generated_text)
        
        # Save latency metrics
        save_latency_metrics(
            request_dict, 
            e2e_time, 
            batch_processing_time, 
            batch_size, 
            queue_wait_time,
            evaluation_score
        )
        
        print(f"Request completed in {e2e_time:.2f}s (E2E), {batch_processing_time:.2f}s (batch processing), evaluation score: {evaluation_score}")
        
        # Add run hash, e2e latency, and evaluation score to response
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