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
    
    # Try to extract code from triple backticks first
    if "```" in code_from_llm:
        parts = code_from_llm.split("```")
        if len(parts) > 1:
            # Get the first code block
            code_block = parts[1]
            lines = code_block.split('\n')
            # Skip the first line if it's a language identifier (e.g., "python")
            if len(lines) > 1 and lines[0].strip() in ['python', 'py', '']:
                return '\n'.join(lines[1:]).strip()
            else:
                return code_block.strip()
    
    # If no triple backticks, try to find Python code patterns
    lines = code_from_llm.split('\n')
    python_lines = []
    in_code = False
    
    for line in lines:
        # Look for Python function/class definitions
        if any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ']):
            in_code = True
        if in_code:
            python_lines.append(line)
    
    if python_lines:
        return '\n'.join(python_lines).strip()
    
    # Fallback: return the original text
    return code_from_llm


def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, temperature, inference_submission=False, gene_id=None):
    """Generates augmented code using Mixtral."""
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
    
    if apply_quality_control:
        base_code = retrieve_base_code(augment_idx)
        code_from_llm, generate_text = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature, gene_id=gene_id)
        code_from_llm = qc_func(code_from_llm, base_code, generate_text)
    else:
        code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature, gene_id=gene_id)
        box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
        print(code_from_llm)
        code_from_llm = clean_code_from_llm(code_from_llm)
    box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    return code_from_llm

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
    
    code_from_llm = submit_mixtral_hf(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm




def submit_mixtral_hf(
    txt2mixtral,
    max_new_tokens=1024,
    top_p=0.15,
    temperature=0.1,
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    return_gen=False,
    gene_id=None,
):
    # Respect an env override and cap hard
    max_new_tokens = min(int(os.getenv("MIXTRAL_MAX_NEW_TOKENS", max_new_tokens)), 2048)

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
    
def submit_llama3_hf(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
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
    max_new_tokens = np.random.randint(900, 1300)
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



def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="gpt2", return_gen=False, gene_id=None):
    max_new_tokens = np.random.randint(800, 1000)
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
        prompt = "Can you rephrase this text:\n```\n{}\n```".format(prompt_text)
        temp = np.random.uniform(0.01, 0.4)
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_hf
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        elif LLM_MODEL == 'local_server':
            llm_code_generator = submit_local_server
        else:
            llm_code_generator = submit_mixtral_hf  # fallback
        output = llm_code_generator(prompt, temperature=temp).strip()
        if "```" in output:
            output = output.split("```")[0]
        output = output + "\n```python\n{}\n```"
        with open(os.path.join(path, "mutant{}.txt".format(i)), 'w') as file:
            file.write(output)


def submit_local_server(txt2llm, max_new_tokens=800, top_p=0.8, temperature=0.7, gene_id=None, **kwargs):
    """
    Submit a request to the local FastAPI server running on PACE-ICE cluster.
    
    Args:
        txt2llm (str): The prompt text to send to the LLM
        max_new_tokens (int): Maximum number of tokens to generate
        top_p (float): Nucleus sampling parameter
        temperature (float): Sampling temperature
        gene_id (str): Identifier for the individual this request belongs to
    
    Returns:
        str: Generated text from the local server
    """
    try:
        # Read the hostname from the file written by the server
        hostname_file = os.getenv("HOSTNAME_LOG_FILE", f"{ROOT_DIR}/hostname.log")

        if not os.path.exists(hostname_file):
            raise Exception("Server hostname file not found. Make sure the server is running.")
        
        with open(hostname_file, 'r') as f:
            server_hostname = f.read().strip()
        
        # Construct the API URL
        server_port = os.getenv("SERVER_PORT", "8000")
        api_url = f"http://{server_hostname}:{server_port}/generate"
        
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
            "job_id": job_id  # Add job identifier to match with slurm file
        }
        
        # Make the HTTP request
        response = requests.post(api_url, json=payload, timeout=None)  #  minute timeout
        
        if response.status_code == 200:
            result = response.json()
            return result.get("generated_text", "")
        else:
            raise Exception(f"Server returned status code {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to local server. Make sure the server is running.")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The server may be overloaded.")
    except Exception as e:
        raise Exception(f"Error calling local server: {str(e)}")
