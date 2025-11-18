import sys
sys.path.append("src")

import re
import os
import glob
import time
import numpy as np
import transformers
from torch import bfloat16
from cfg.constants import *
from utils.print_utils import box_print
from utils.rag_metrics import record_metric
from rag.runtime import get_runtime

from typing import Optional
#import fire
# from llama import Llama
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer
from google import genai
from google.genai import types


from huggingface_hub.utils import HfHubHTTPError
import os, time, random


def retrieve_base_code(idx):
    """Retrieves base code for quality control."""
    base_network = SEED_NETWORK
    return split_file(base_network)[1:][idx].strip()


def clean_code_from_llm(code_from_llm):
    """Cleans the code received from LLM."""
    code_from_llm = code_from_llm.strip()
    if not code_from_llm:
        return ""

    # Surya: Extract code from fenced blocks using regex instead to avoid capturing markdown prose
    fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", code_from_llm, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        return fenced_blocks[-1].strip()

    # If no triple backticks, try to find Python code patterns
    lines = code_from_llm.split('\n')
    python_lines = []
    in_code = False

    keywords = ('def ', 'class ', 'import ', 'from ', '@', 'for ', 'while ', 'if ', 'try', 'with ', 'return ', 'pass', 'raise ')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_code:
                python_lines.append(line)
            continue
        if any(stripped.startswith(keyword) for keyword in keywords):
            in_code = True
        if in_code:
            python_lines.append(line)

    if python_lines:
        return '\n'.join(python_lines).strip()

    # No recognizable code found
    return ""


def _validate_python_snippet(snippet: str) -> tuple[bool, str]:
    """Compile-check Python code before saving to catch syntax errors early."""
    if not snippet or not snippet.strip():
        return False, "empty snippet"
    try:
        compile(snippet, "<llm_snippet>", "exec")
    except (SyntaxError, IndentationError, ValueError) as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    except Exception as exc:  # Catch other rare issues such as encoding errors
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, ""


def _format_retry_prompt(base_prompt: str, attempt: int) -> str:
    """Add stricter formatting instructions on retry attempts to coerce valid code."""
    if attempt == 0:
        return base_prompt
    enforcement = (
        "\n\nSTRICT INSTRUCTIONS: Return only the fully updated Python code inside a single ```python``` fenced block. "
        "Do not include commentary, analysis, or markdown outside that block."
    )
    return f"{base_prompt}{enforcement}"


def validate_module_source(source_code: str, module_path: str, module_name: Optional[str] = None) -> None:
    """Execute module source to catch runtime errors (NameError, etc.) before evaluation."""
    unique_name = module_name or f"_llmge_validation_{hash(module_path)}"
    module_globals = {"__name__": unique_name, "__file__": module_path}
    exec(compile(source_code, module_path, "exec"), module_globals, {})


def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, temperature, inference_submission=False, gene_id=None):
    """Generate augmented code with retry loop: validates syntax before accepting LLM output."""
    box_print("PROMPT TO LLM", print_bbox_len=60, new_line_end=False)
    print(txt2llm)

    if inference_submission is False:
        if LLM_MODEL == 'local_server':
            llm_code_generator = submit_local_server
        else:
            llm_code_generator = submit_mixtral
        qc_func = llm_code_qc
    else:
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_hf
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        elif LLM_MODEL == 'gemini':
            llm_code_generator = submit_gemini_api
        qc_func = llm_code_qc_hf

    last_error = ""
    # Surya: Better retry loop with a configurable max retry constant: re-prompt LLM if generated code fails validation tests
    for attempt in range(LLM_GENERATION_MAX_RETRIES):
        prompt = _format_retry_prompt(txt2llm, attempt)
        if apply_quality_control:
            base_code = retrieve_base_code(augment_idx)
            raw_response, generate_text = llm_code_generator(
                prompt, return_gen=True, top_p=top_p, temperature=temperature, gene_id=gene_id
            )
            candidate_code = qc_func(raw_response, base_code, generate_text)
        else:
            raw_response = llm_code_generator(prompt, top_p=top_p, temperature=temperature, gene_id=gene_id)
            box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
            print(raw_response)
            candidate_code = clean_code_from_llm(raw_response)

        candidate_code = clean_code_from_llm(candidate_code)

        is_valid, validation_error = _validate_python_snippet(candidate_code)
        if is_valid:
            box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
            print(candidate_code)
            return candidate_code

        last_error = validation_error or "unable to extract python code"
        box_print("INVALID LLM OUTPUT", print_bbox_len=60, new_line_end=False)
        print(f"Attempt {attempt + 1} failed validation: {last_error}")

    raise RuntimeError(
        f"LLM failed to provide valid Python after {LLM_GENERATION_MAX_RETRIES} attempts. Last error: {last_error}"
    )

def extract_note(txt):
    """Extracts note from the part if present."""
    if "# -- NOTE --" in txt:
        note_txt = txt.split('# -- NOTE --')
        return '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    return ''

# Function to load and split the file
def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts


def _augment_template_with_rag(template_text: str, mutation_label: str | None, query_code: str | None = None) -> str:
    """
    Inject RAG context into a template when the runtime is enabled.
    """
    runtime = get_runtime()
    if runtime is None:
        return template_text
    start = time.perf_counter()
    augmented_template, mutations = runtime.enhance_template(
        template=template_text,
        mutation_type=mutation_label,
        query_code=query_code,
    )
    duration_ms = (time.perf_counter() - start) * 1000
    record_metric(
        "rag_prompt_enhancement",
        {
            "mutation_type": mutation_label,
            "retrieval_ms": duration_ms,
            "retrieved_mutations": len(mutations),
            "prompt_tokens": len(augmented_template.split()),
        },
    )
    return augmented_template


def _prepend_rag_context_to_prompt(prompt_text: str, mutation_label: str | None) -> str:
    """
    Build an instruction prefix that references top-performing mutations.
    """
    runtime = get_runtime()
    if runtime is None or not mutation_label:
        return prompt_text
    start = time.perf_counter()
    mutations = runtime.collect_context(mutation_type=mutation_label)
    duration_ms = (time.perf_counter() - start) * 1000
    if not mutations:
        return prompt_text
    context_block = runtime.format_context(mutations)
    rag_prefix = (
        "Here are some successful mutations from prior generations. "
        "Consider how their approaches might inspire your own creative solution, but feel free to explore novel directions.\n"
        f"{context_block}\n\n"
    )
    record_metric(
        "rag_prompt_rephrase_context",
        {
            "mutation_type": mutation_label,
            "retrieval_ms": duration_ms,
            "retrieved_mutations": len(mutations),
            "prompt_tokens": len(prompt_text.split()),
        },
    )
    return f"{rag_prefix}{prompt_text}"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def llm_code_qc(code_from_llm, base_code, generate_text):
    # TODO: make parameter
    template_path = os.path.join(ROOT_DIR, 'templates/llm_quality_control.txt')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    res = generate_text(prompt2llm) # clean txt
    code_from_llm = res[0]["generated_text"]
    code_from_llm = '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()
    return code_from_llm


def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    # TODO: make parameter
    fname = np.random.choice(['llm_quality_control_p.txt', 'llm_quality_control_p.txt'])
    template_path = os.path.join(ROOT_DIR, f'templates/{fname}')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    code_from_llm = submit_mixtral_hf(prompt2llm, max_new_tokens=4096, top_p=0.1, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm




def submit_mixtral_hf(
    txt2mixtral,
    max_new_tokens=4096,
    top_p=0.15,
    temperature=0.1,
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    return_gen=False,
    gene_id=None,
):
    # Respect an env override (no hard cap)
    max_new_tokens = int(os.getenv("MIXTRAL_MAX_NEW_TOKENS", max_new_tokens))

    # Use the correct env var for HF token; pass it explicitly
    # (HF_API_KEY isn't used by the hub; HUGGING_FACE_HUB_TOKEN / HF_TOKEN are.)
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    client = InferenceClient(model=model_id, token=token, timeout=60)

    # Leave caching ON unless you *need* to defeat it; disabling cache burns credits faster
    # client.headers["x-use-cache"] = "0"

    messages = [{"role": "user", "content": "Provide code in Python\n" + txt2mixtral}]

    # Exponential backoff for transient limits; fast-fail for 402
    attempts = 0
    while True:
        try:
            resp = client.chat.completions.create(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                seed=101,
            )
            text = resp.choices[0].message.content
            return (text, None) if return_gen else text

        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = str(e)
            # Log once for observability
            print(f"[HF ERROR] status={status} model={model_id} chat.completions; {msg}", flush=True)

            # Monthly credit gate -> do not retry this path
            if status == 402 or "Payment Required" in msg:
                # Optional fallback: switch to another path/provider you control
                # return submit_llama3_hf(txt2mixtral, max_new_tokens=max_new_tokens,
                #                         top_p=top_p, temperature=temperature, return_gen=return_gen)
                raise RuntimeError("HF Inference Providers credits exhausted (402).") from e

            # Classic rate limit or transient provider error -> back off and retry
            if status in (429, 500, 502, 503, 504):
                delay = min(2 ** attempts, 30) + random.uniform(0, 0.5)
                time.sleep(delay)
                attempts += 1
                if attempts <= 5:
                    continue

            # Anything else -> bubble up
            raise
    
def submit_llama3_hf(txt2llama, max_new_tokens=4096, top_p=0.15, temperature=0.1, 
                      model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", return_gen=False, gene_id=None):
    """
    This function submits a model prompt to Llama3 through the HuggingFace Inference API

    Parameters
    ----------
    txt2llama : str
        Prompt that will be sent to Llama3
    max_new_tokens : int, optional
        A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
        Which Llama3 variant to utilize for inference, by default "meta-llama/Meta-Llama-3.1-70B-Instruct"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """    
    max_new_tokens = np.random.randint(2048, 4096)
    # Use environment variable for HF API key
    # Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2llama,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_gemini_api(txt2gemini, gene_id=None, **kwargs):
    """
    This function submits a model prompt to Gemini through its API

    Parameters
    ----------
    txt2gemini : str
        Prompt that will be sent to Gemini

    Returns
    -------
    str
        Model's output from inference
    """    
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[txt2gemini],
        
    )
    return response.text



def submit_mixtral(txt2mixtral, max_new_tokens=4096, top_p=0.15, temperature=0.1, 
                   model_id="gpt2", return_gen=False, gene_id=None):
    max_new_tokens = np.random.randint(2048, 4096)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2mixtral)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
    
def mutate_prompts(n=5):
    templates = np.random.choice(glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt'), n)
    for i, template in enumerate(templates):
        path, filename = os.path.split(template)
        with open(template, 'r') as file:
            prompt_text = file.read()
        prompt_text = prompt_text.split("```")[0].strip()
        mutation_label = os.path.splitext(filename)[0]
        prompt_base = (
            "Rephrase the following prompt template text. "
            "Return ONLY the rephrased prompt text, do NOT include any code examples or code blocks. "
            "The output should be a prompt template that can be used to instruct an LLM to modify code. "
            "Preserve the placeholder {} where code should be inserted.\n\n"
            "Original prompt template:\n```\n{}\n```\n\n"
            "Rephrased prompt template (text only, no code):"
        ).format(prompt_text)
        prompt = _prepend_rag_context_to_prompt(prompt_base, mutation_label)
        temp = np.random.uniform(0.01, 0.4)
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_hf
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        elif LLM_MODEL == 'local_server':
            llm_code_generator = submit_local_server
        else:
            llm_code_generator = submit_mixtral_hf  # fallback
        output = llm_code_generator(prompt, temperature=temp, gene_id="mutate_prompts").strip()
        # Remove any code blocks that LLM might have generated
        if "```" in output:
            # Extract text before first code block
            output = output.split("```")[0].strip()
        # Ensure output ends with the code placeholder
        if "{}" not in output:
            output = output + "\n```python\n{}\n```"
        elif not output.rstrip().endswith("```"):
            # If {} exists but doesn't end with code block, add it
            output = output.rstrip() + "\n```python\n{}\n```"
        with open(os.path.join(path, "mutant{}.txt".format(i)), 'w') as file:
            file.write(output)


def submit_local_server(txt2llm, max_new_tokens=8192, top_p=0.8, temperature=0.7, gene_id=None, **kwargs):
    """
    Submit a request to the local FastAPI server or load balancer running on PACE-ICE cluster.
    This helper now includes retry logic and an optional remote fallback to keep evolution moving
    even when the primary gateway is down.
    """
    try:
        # Check if load balancer mode is enabled
        use_load_balancer = os.getenv("USE_LOAD_BALANCER", "false").lower() in ['true', '1', 'yes']
        
        if use_load_balancer:
            # Load balancer mode: connect to load balancer
            loadbalancer_file = os.getenv("LOADBALANCER_LOG_FILE", f"{ROOT_DIR}/loadbalancer.log")
            
            if not os.path.exists(loadbalancer_file):
                raise Exception("Load balancer hostname file not found. Make sure the load balancer is running.")
            
            with open(loadbalancer_file, 'r') as f:
                server_hostname = f.read().strip()
            
            # Use load balancer port
            server_port = os.getenv("LOAD_BALANCER_PORT", "9000")
            api_url = f"http://{server_hostname}:{server_port}/generate"
            print(f"[INFO] Using load balancer at {api_url}")
        else:
            # Single server mode: connect directly to server (backward compatible)
            hostname_file = os.getenv("HOSTNAME_LOG_FILE", f"{ROOT_DIR}/hostname.log")

            if not os.path.exists(hostname_file):
                raise Exception("Server hostname file not found. Make sure the server is running.")
            
            with open(hostname_file, 'r') as f:
                server_hostname = f.read().strip()
            
            # Construct the API URL
            server_port = os.getenv("SERVER_PORT", "8000")
            api_url = f"http://{server_hostname}:{server_port}/generate"
            print(f"[INFO] Using single server at {api_url}")
        
        # Get job identification from environment (use Slurm job ID directly)
        # Try multiple sources to find the Slurm job ID
        job_id = os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOBID") or os.getenv("JOB_ID")
        
        # Debug: Print what we found
        if job_id:
            print(f"[DEBUG] Found job_id from environment: {job_id}")
        
        if not job_id:
            # Try to read from slurm environment file if it exists
            try:
                slurm_env_file = f"/proc/{os.getpid()}/environ"
                if os.path.exists(slurm_env_file):
                    with open(slurm_env_file, 'rb') as f:
                        env_data = f.read().decode('utf-8', errors='ignore')
                        for item in env_data.split('\x00'):
                            if item.startswith('SLURM_JOB_ID='):
                                job_id = item.split('=', 1)[1]
                                print(f"[DEBUG] Found job_id from /proc/environ: {job_id}")
                                break
            except Exception as e:
                print(f"[DEBUG] Could not read /proc/environ: {e}")
        
        # Final fallback
        if not job_id:
            job_id = "local"
            print(f"[DEBUG] Using fallback job_id: {job_id}")
        
        # Prepare the request payload
        payload = {
            "prompt": txt2llm,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "job_id": job_id,  # Add job identifier to match with slurm file
            "gene_id": gene_id  # Add gene_id to track individual
        }
        
        timeout_seconds = float(os.getenv("LOCAL_SERVER_TIMEOUT", 300))
        max_retries = int(os.getenv("LOCAL_SERVER_MAX_RETRIES", 3))
        enable_remote_fallback = os.getenv("ENABLE_LLM_REMOTE_FALLBACK", "false").lower() in {"1", "true", "yes"}
        fallback_target = os.getenv("LLM_REMOTE_FALLBACK_TARGET", "mixtral_hf")
        last_exception: Exception | None = None
        
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(api_url, json=payload, timeout=timeout_seconds)
                if response.status_code == 200:
                    result = response.json()
                    return result.get("generated_text", "")
                else:
                    raise Exception(f"Server returned status code {response.status_code}: {response.text}")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exception = exc
                print(f"[WARN] Local server attempt {attempt}/{max_retries} failed: {exc}")
            except Exception as exc:
                last_exception = exc
                print(f"[WARN] Local server attempt {attempt}/{max_retries} failed: {exc}")
                # Non-transient error; exit retry loop
                break
            
            if attempt < max_retries:
                backoff = min(5 * attempt, 30)
                print(f"[INFO] Retrying local server in {backoff} seconds...")
                time.sleep(backoff)
        
        if last_exception and enable_remote_fallback:
            print(f"[WARN] Falling back to remote LLM due to local server failure: {last_exception}")
            safe_tokens = min(max_new_tokens, 4096)
            try:
                if fallback_target == "mixtral_hf":
                    return submit_mixtral_hf(
                        txt2llm,
                        max_new_tokens=safe_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        return_gen=False,
                        gene_id=gene_id,
                    )
                elif fallback_target == "mixtral":
                    return submit_mixtral(
                        txt2llm,
                        max_new_tokens=safe_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        return_gen=False,
                        gene_id=gene_id,
                    )
                else:
                    raise Exception(f"Unknown fallback target '{fallback_target}'")
            except Exception as fallback_exc:
                raise Exception(
                    f"Local server failed after {max_retries} attempts and remote fallback "
                    f"also failed: {fallback_exc}"
                ) from fallback_exc
        
        if last_exception:
            raise Exception(f"Error calling local server after {max_retries} attempts: {last_exception}")
        raise Exception("Unhandled error calling local server.")
            
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to local server. Make sure the server is running.")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The server may be overloaded.")
    except Exception as e:
        raise Exception(f"Error calling local server: {str(e)}")
