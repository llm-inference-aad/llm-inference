#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os

repo_id = 'meta-llama/Llama-3.2-1B-Instruct'
target = '/home/hice1/jgil37/scratch/llm_models/meta-llama/Llama-3.2-1B-Instruct'

print(f'Downloading {repo_id} to {target}...')
try:
    snapshot_download(repo_id, local_dir=target)
    print('Download complete!')
    os.system(f'du -sh {target}')
    os.system(f'ls -lh {target}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
