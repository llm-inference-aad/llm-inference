import re
import time
import glob
import numpy as np
import transformers
from torch import bfloat16
import argparse
from cfg.constants import *
from utils.print_utils import box_print
from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, generate_augmented_code, 
                       extract_note, clean_code_from_llm, retrieve_base_code,
                       validate_module_source)
from pathlib import Path


def augment_network(input_filename='network.py', output_filename='network_x.py', template_txt=None,
                    top_p=0.15, temperature=0.1, apply_quality_control=False, inference_submission=False, gene_id=None):
    
    print(f'Loading {input_filename} code')
    parts = split_file(input_filename)
    augment_idx = np.random.randint(1, len(parts))
    # select code to be augmented randomly 
    code2llm = parts[augment_idx]
    original_parts = parts[:]
    # prompt_templates = glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt')
    # template_path = np.random.choice(prompt_templates)
    # template_path = f'{ROOT_DIR}/templates/{fname}'
    fname = template_txt
    with open(fname, 'r') as file:
        template_txt = file.read()
    # add code to be augmented 
    txt2llm = template_txt.format(code2llm.strip())
    note_txt = extract_note(code2llm)

    fallback_reason = None
    candidate_txt = None
    # Surya: Validate assembled module; fallback to parent if all retries fail
    for attempt in range(LLM_GENERATION_MAX_RETRIES):
        try:
            code_from_llm = generate_augmented_code(
                txt2llm,
                augment_idx-1,
                apply_quality_control,
                top_p,
                temperature,
                inference_submission=inference_submission,
                gene_id=gene_id,
                previous_error=fallback_reason,
            )
        except Exception as exc:
            fallback_reason = str(exc)
            break

        candidate_parts = parts[:]
        candidate_parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
        candidate_txt = '# --OPTION--'.join(candidate_parts)
        try:
            # Surya: Execute module to catch runtime errors before evaluation
            validate_module_source(
                candidate_txt,
                output_filename,
                module_name=f"_llmge_{gene_id}" if gene_id else None,
            )
            fallback_reason = None
            break
        except Exception as exc:
            fallback_reason = str(exc)
            box_print("Generated module failed validation", print_bbox_len=80, new_line_end=False)
            print(f"Attempt {attempt + 1} validation error: {fallback_reason}")
            candidate_txt = None

    # Surya: Fallback guarantees every individual yields a loadable module
    fallback_marker = Path(f"{output_filename}.fallback")
    if fallback_reason is not None or candidate_txt is None:
        box_print("Fallback to parent code triggered", print_bbox_len=80, new_line_end=False)
        print(f"Reason: {fallback_reason}")
        python_network_txt = '# --OPTION--'.join(original_parts)
        try:
            fallback_marker.write_text((fallback_reason or "unknown").strip() or "unknown")
        except OSError as marker_exc:
            print(f"[WARN] Unable to write fallback marker for {output_filename}: {marker_exc}")
    else:
        if fallback_marker.exists():
            try:
                fallback_marker.unlink()
            except OSError as marker_exc:
                print(f"[WARN] Unable to remove fallback marker for {output_filename}: {marker_exc}")
        python_network_txt = candidate_txt
    # Write the text to the file
    file = Path(output_filename)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, 'w') as file:
        file.write(python_network_txt)
        
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job Done')

    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('template_txt', type=str, help='Template txt')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')
    parser.add_argument('--inference_submission', type=str2bool, default=False, help='True to submit for inference remotely')
    parser.add_argument('--gene_id', type=str, default=None, help='Gene ID for tracking')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename=args.input_filename,
                    output_filename=args.output_filename,
                    template_txt=args.template_txt,
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                    inference_submission=args.inference_submission,
                    gene_id=args.gene_id
                   )
