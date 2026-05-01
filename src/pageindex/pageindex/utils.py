import tiktoken
import requests
import logging
import os
import re
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import yaml
from pathlib import Path
from types import SimpleNamespace as config

log = logging.getLogger("pageindex")


def _get_server_url():
    """Return the local server /generate URL, following the same env-var
    pattern as ``submit_local_server`` in ``llm_utils.py``."""
    use_load_balancer = os.getenv("USE_LOAD_BALANCER", "false").lower() in ['true', '1', 'yes']

    if use_load_balancer:
        lb_file = os.getenv("LOADBALANCER_LOG_FILE")
        if not lb_file or not os.path.exists(lb_file):
            raise RuntimeError("LOADBALANCER_LOG_FILE not set or file not found. Is the load balancer running?")
        with open(lb_file, 'r') as f:
            hostname = f.read().strip()
        port = os.getenv("LOAD_BALANCER_PORT", "9000")
    else:
        host_file = os.getenv("HOSTNAME_LOG_FILE")
        if not host_file or not os.path.exists(host_file):
            raise RuntimeError("HOSTNAME_LOG_FILE not set or file not found. Is the server running?")
        with open(host_file, 'r') as f:
            hostname = f.read().strip()
        port = os.getenv("SERVER_PORT", "8000")

    return f"http://{hostname}:{port}/generate"


def _post_to_server(prompt):
    """POST a prompt to the local server and return the raw JSON result dict.

    The local vLLM server runs at ``MAX_MODEL_LEN=8192`` (so prompt + output
    tokens must fit under that). ``PAGEINDEX_MAX_NEW_TOKENS`` controls the
    output budget; we default to 2048 so a ~6k-token prompt still fits.

    Server compatibility note: the default ``server_vllm.py`` prepends a
    ``SYSTEM_PROMPT`` that requires a fenced ``python`` code block of
    *runnable* Python — right for the LLMGE mutation path, wrong for
    PageIndex which needs raw JSON. We send ``system_prompt=""`` to opt
    out of the server default for this caller without affecting others.
    """
    url = _get_server_url()
    payload = {
        "prompt": prompt,
        "max_new_tokens": int(os.getenv("PAGEINDEX_MAX_NEW_TOKENS", 2048)),
        # Greedy decoding is more reliable here: at low non-zero
        # temperature combined with the server's hardcoded
        # ``repetition_penalty=1.1`` the EOS token can win the first
        # sampling step and produce 1-token responses.
        "temperature": 0.0,
        "system_prompt": "",
        "gene_id": "pageindex",
        "repetition_penalty": 1.0,
    }
    prompt_preview = prompt[:300].replace('\n', '\\n')
    log.debug("[REQ] url=%s prompt(%d chars)=%.300s…", url, len(prompt), prompt_preview)
    timeout_seconds = float(os.getenv("LOCAL_SERVER_TIMEOUT", 300))
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    result = response.json()
    text = result.get("generated_text", "")
    finish = result.get("finish_reason", "?")
    log.debug("[RESP] finish=%s len=%d text=%.500s", finish, len(text), text[:500].replace('\n', '\\n'))
    return result


def count_tokens(text, model=None):
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def ChatGPT_API_with_finish_reason(model=None, prompt="", api_key=None, chat_history=None):
    max_retries = 10
    for i in range(max_retries):
        try:
            result = _post_to_server(prompt)
            text = result["generated_text"]
            if result.get("finish_reason") == "length":
                return text, "max_output_reached"
            else:
                return text, "finished"
        except Exception as e:
            log.warning("Retry %d/%d: %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                time.sleep(1)
            else:
                log.error("Max retries reached. prompt=%.200s", prompt[:200])
                return "Error", "error"


def ChatGPT_API(model=None, prompt="", api_key=None, chat_history=None):
    max_retries = 10
    for i in range(max_retries):
        try:
            result = _post_to_server(prompt)
            return result["generated_text"]
        except Exception as e:
            log.warning("Retry %d/%d: %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                time.sleep(1)
            else:
                log.error("Max retries reached. prompt=%.200s", prompt[:200])
                return "Error"


async def ChatGPT_API_async(model=None, prompt="", api_key=None):
    max_retries = 10
    for i in range(max_retries):
        try:
            result = await asyncio.to_thread(_post_to_server, prompt)
            return result["generated_text"]
        except Exception as e:
            log.warning("Retry %d/%d: %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                log.error("Max retries reached. prompt=%.200s", prompt[:200])
                return "Error"  
            
            
def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content
         

def _strip_trailing_commas(text):
    """Remove trailing commas before } or ] (with optional whitespace)."""
    import re
    return re.sub(r',\s*([}\]])', r'\1', text)


def _fix_single_quoted_json(text):
    """Best-effort conversion of single-quoted JSON (common with local models) to double-quoted."""
    import re
    # Replace single-quoted keys:  'key':  ->  "key":
    text = re.sub(r"'(\w+)'\s*:", r'"\1":', text)
    # Replace single-quoted string values after colon:  : 'value'  ->  : "value"
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    return text


def _candidate_json_strings(content):
    """Yield best-effort JSON-text candidates from *content* in priority order.

    The local LLM server prepends a system prompt requiring a fenced
    ``python`` code block, which causes JSON requests to come back wrapped
    in `````python ... ````` (and sometimes inside an actual Python
    snippet that *prints* the JSON).  To survive that we look in order:
      1. The body of a `````json`` fence, if present.
      2. The body of any ``````` fence (e.g. ``python``).
      3. The raw content.
      4. The widest balanced ``[...]`` substring inside any of the above.
      5. The widest balanced ``{...}`` substring inside any of the above.
    """
    import re as _re

    seen: set[str] = set()

    def _emit(text: str):
        if not text:
            return
        s = text.strip()
        if s and s not in seen:
            seen.add(s)
            yield s

    # Fence variants
    for fence_match in _re.finditer(r"```(json|python)?\s*\n([\s\S]*?)```", content):
        for s in _emit(fence_match.group(2)):
            yield s

    # Raw content
    for s in _emit(content):
        yield s

    # Balanced bracket scans across fence-stripped body and raw content.
    bodies = [content]
    for fence_match in _re.finditer(r"```(json|python)?\s*\n([\s\S]*?)```", content):
        bodies.append(fence_match.group(2))
    for body in bodies:
        for opener, closer in (("[", "]"), ("{", "}")):
            start = body.find(opener)
            end = body.rfind(closer)
            if start != -1 and end != -1 and end > start:
                for s in _emit(body[start:end + 1]):
                    yield s


def _normalise_for_json(text):
    """Cleanup known parser pitfalls before ``json.loads``."""
    import re as _re
    text = text.replace('\\_', '_')                       # LaTeX-style escapes
    text = _re.sub(r'\bNone\b', 'null', text)             # Python None
    text = _re.sub(r'\bTrue\b', 'true', text)             # Python True
    text = _re.sub(r'\bFalse\b', 'false', text)           # Python False
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text


def extract_json(content):
    """Best-effort JSON extraction tolerant of Python-fenced LLM output.

    Returns the parsed JSON value, or ``{}`` on total failure (matches the
    upstream contract — caller checks for empty / wrong-shape result).
    """
    last_error: Exception | None = None
    for candidate in _candidate_json_strings(content):
        normalised = _normalise_for_json(candidate)
        for transform in (
            lambda s: s,
            _strip_trailing_commas,
            lambda s: _strip_trailing_commas(_fix_single_quoted_json(s)),
        ):
            try:
                return json.loads(transform(normalised))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
    log.error(
        "extract_json FAILED — last error=%s — raw LLM output:\n%s",
        last_error, content,
    )
    return {}

def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure):
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    
def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

    
def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure['nodes']:
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes

def is_leaf_node(data, node_id):
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    ###return text not list 
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text+=page.extract_text()
    return text

def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text

def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page

def get_last_start_page_from_text(text):
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace('/', replacement)

def get_pdf_name(pdf_path):
    # Extract PDF name
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)
    



def list_to_tree(data):
    def get_parent_structure(structure):
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)
    
    # Helper function to clean empty children arrays
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]

def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data



def get_page_tokens(pdf_path, model="local_server", pdf_parser="PyPDF2"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        enc = tiktoken.get_encoding("cl100k_base")
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")

        

def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num



def _sort_siblings_by_page(structure):
    """Sort flat TOC items by physical_index within each sibling group.

    Groups are determined by the parent prefix of the ``structure`` field
    (e.g. items "1.1", "1.2", "1.3" share parent "1").  Sorting only within
    groups preserves the hierarchy that ``list_to_tree`` relies on while
    fixing inverted page indices produced by the LLM.
    """
    from itertools import groupby

    def _parent_key(item):
        s = str(item.get('structure', ''))
        parts = s.split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else ''

    result = []
    for _key, grp in groupby(structure, key=_parent_key):
        items = list(grp)
        items.sort(key=lambda x: (x.get('physical_index') or 0))
        result.extend(items)
    return result


def _re_derive_structure_from_titles(items):
    """Re-derive hierarchy from document numbering embedded in titles.

    Academic papers often use numbering schemes like:
      I. INTRODUCTION / II. METHOD / A. Leaky ReLU / B. Convolutional layers

    If the majority of titles carry such prefixes, re-compute the ``structure``
    field from them so that ``list_to_tree`` parents children correctly —
    regardless of what the LLM assigned.

    Handles multi-column PDF extraction where sections may appear out of
    document order (e.g. "III. ARCHITECTURE" before "I. INTRODUCTION")
    by sorting top-level items by their detected number before assigning
    letter sub-sections to parents.
    """
    roman_re = re.compile(
        r'^([IVXLC]+)\.\s', re.IGNORECASE
    )
    letter_re = re.compile(
        r'^([A-Z])\.\s', re.IGNORECASE
    )
    arabic_re = re.compile(
        r'^(\d+)\.\s'
    )
    dot_numbered_re = re.compile(
        r'^(\d+(?:\.\d+)+)\s'
    )

    _ROMAN_MAP = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
    }

    # Classify each title
    classifications = []
    for item in items:
        title = item.get('title', '').strip()
        cls = None

        m = roman_re.match(title)
        if m:
            num = _ROMAN_MAP.get(m.group(1).upper())
            if num is not None:
                cls = ('roman', num)

        if cls is None:
            m = letter_re.match(title)
            if m:
                cls = ('letter', ord(m.group(1).upper()) - ord('A') + 1)

        if cls is None:
            m = dot_numbered_re.match(title)
            if m:
                cls = ('dot', m.group(1))

        if cls is None:
            m = arabic_re.match(title)
            if m:
                cls = ('arabic', int(m.group(1)))

        classifications.append(cls)

    # Only apply if at least half the items have detected numbering
    detected = sum(1 for c in classifications if c is not None)
    if detected < len(items) * 0.5:
        return items

    # Separate top-level (roman/arabic) and sub-level (letter) items.
    # For each letter item, record the highest-numbered top-level section
    # seen so far (not the *last* one in page order).  Multi-column PDF
    # extraction can scramble section order (e.g. III before I before II),
    # so tracking the maximum ensures letter items are parented under the
    # correct section regardless of extraction order.
    top_items = []      # (detected_number, original_index, item)
    letter_items = []   # (letter_number, original_index, item, parent_top_number)

    max_top_number = 0
    unnumbered_counter = 0
    for i, (item, cls) in enumerate(zip(items, classifications)):
        if cls is None:
            # No numbering — treat as top-level with synthetic number
            unnumbered_counter += 1
            synthetic_num = 1000 + unnumbered_counter
            top_items.append((synthetic_num, i, item))
        elif cls[0] in ('roman', 'arabic'):
            max_top_number = max(max_top_number, cls[1])
            top_items.append((cls[1], i, item))
        elif cls[0] == 'letter':
            letter_items.append((cls[1], i, item, max_top_number))
        elif cls[0] == 'dot':
            item['structure'] = cls[1]
            top_items.append((0, i, item))

    # Sort top-level items by their detected number (fixes column ordering)
    top_items.sort(key=lambda x: x[0])

    # Build the final sorted list: top-level items in number order,
    # with letter sub-items inserted after their parent.
    result = []
    for top_num, _, item in top_items:
        item['structure'] = str(top_num)
        result.append(item)
        # Append letter children that belong to this top-level section
        for letter_num, _, letter_item, parent_num in letter_items:
            if parent_num == top_num:
                letter_item['structure'] = f'{top_num}.{letter_num}'
                result.append(letter_item)

    # Any letter items whose parent wasn't found (edge case) go at the end
    assigned = {id(li) for _, _, li, _ in letter_items
                if any(pn == li_pn for pn, _, _, li_pn in letter_items
                       for top_num, _, _, _ in [(0, 0, None, 0)])}
    # (The loop above already handled all letter items via parent_num matching)

    return result


def post_processing(structure, end_physical_index):
    # Ensure siblings are in page order before computing end_index
    structure = _sort_siblings_by_page(structure)

    # Re-derive hierarchy from document numbering if present
    structure = _re_derive_structure_from_titles(structure)

    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree)!=0:
        return tree
    else:
        ### remove appear_start 
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure

def clean_structure_post(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data

def remove_fields(data, fields=['text']):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree, indent=0):
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    list = structure_to_list(structure)
    for node in list:
        num_tokens = count_tokens(node['text'], model='gpt-4o')
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node['start_index'])
            print("End Index:", node['end_index'])
            print("Title:", node['title'])
            print("\n")


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data):
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def fix_node_ranges(node):
    """Clamp inverted start_index/end_index to prevent empty text extraction."""
    if isinstance(node, dict):
        s, e = node.get('start_index'), node.get('end_index')
        if s is not None and e is not None and s > e:
            node['end_index'] = s
        if 'nodes' in node:
            fix_node_ranges(node['nodes'])
    elif isinstance(node, list):
        for item in node:
            fix_node_ranges(item)


def remove_bogus_nodes(tree):
    """Remove nodes with single-character alphabetic titles (figure-label artifacts).

    Children of removed nodes are re-parented into the removed node's position
    in its parent's children list.
    """
    if isinstance(tree, list):
        i = 0
        while i < len(tree):
            node = tree[i]
            title = (node.get('title') or '').strip().rstrip('.')
            if len(title) <= 1 and title.isalpha():
                children = node.get('nodes', [])
                log.debug("remove_bogus_nodes: dropping node '%s', re-parenting %d children",
                          node.get('title'), len(children))
                tree[i:i+1] = children if isinstance(children, list) else [children]
                # don't increment i — re-check the replacement nodes
            else:
                if 'nodes' in node and isinstance(node['nodes'], list):
                    remove_bogus_nodes(node['nodes'])
                i += 1
    elif isinstance(tree, dict):
        if 'nodes' in tree and isinstance(tree['nodes'], list):
            remove_bogus_nodes(tree['nodes'])


def fix_parent_child_page_ranges(tree):
    """Expand parent start_index/end_index to enclose all children.

    Prevents text-assignment failures when the LLM assigns a parent range
    that doesn't cover its children's pages.
    """
    if isinstance(tree, list):
        for item in tree:
            fix_parent_child_page_ranges(item)
    elif isinstance(tree, dict):
        if 'nodes' in tree and isinstance(tree['nodes'], list):
            fix_parent_child_page_ranges(tree['nodes'])
            child_starts = [c.get('start_index') for c in tree['nodes']
                            if isinstance(c, dict) and c.get('start_index') is not None]
            child_ends = [c.get('end_index') for c in tree['nodes']
                          if isinstance(c, dict) and c.get('end_index') is not None]
            if child_starts:
                parent_s = tree.get('start_index')
                if parent_s is None or min(child_starts) < parent_s:
                    tree['start_index'] = min(child_starts)
            if child_ends:
                parent_e = tree.get('end_index')
                if parent_e is None or max(child_ends) > parent_e:
                    tree['end_index'] = max(child_ends)


def find_title_offset(page_text, title):
    """Find character offset of a section title in page text using fuzzy whitespace matching.

    Tolerates extra whitespace *within* words (common PDF extraction artifact
    from multi-column layouts where e.g. "REDUCING" becomes "R EDUCING") as
    well as between words.

    Returns the index into *page_text* where *title* begins, or ``None`` if the
    title cannot be located.
    """
    words = title.split()
    if not words:
        return None

    # Guard: very short single-word titles (e.g. "A", "B") require a
    # line-boundary match to avoid matching random body-text occurrences.
    if len(words) == 1 and len(words[0]) <= 3:
        pattern = r'(?:^|\n)\s*' + re.escape(words[0]) + r'\s*(?:\n|$)'
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            # Return the offset of the title text itself, not the newline
            inner = re.search(re.escape(words[0]), match.group(0), re.IGNORECASE)
            return match.start() + inner.start() if inner else match.start()
        return None

    # First try: flexible whitespace between whole words (fast, avoids false
    # positives from the more permissive intra-word pattern).
    inter_word = r'\s+'.join(re.escape(w) for w in words)
    match = re.search(inter_word, page_text, re.IGNORECASE)
    if match:
        return match.start()

    # Second try: allow optional whitespace between characters within each
    # word to handle PDF extraction artifacts (e.g. "R EDUCING").
    # Only use this for titles with multiple words or a single long word
    # (>= 6 chars) to avoid false positives on short words.
    if len(words) >= 2 or len(words[0]) >= 6:
        intra_char = r'\s+'.join(
            r'\s*'.join(re.escape(c) for c in w) for w in words
        )
        match = re.search(intra_char, page_text, re.IGNORECASE)
        return match.start() if match else None

    return None


def compute_section_offsets(node, page_list):
    """Legacy shim — delegates to ``compute_all_section_offsets``."""
    compute_all_section_offsets(node, page_list)


def compute_all_section_offsets(node, page_list):
    """Annotate every node with character-level boundaries using a single
    global title search across the entire document.

    This prevents cross-level text duplication by assigning each character
    in the document to exactly one node.  All titles are located in
    document order; each node gets the text from its title to the next
    title (regardless of hierarchy level).
    """
    # Build single global combined text
    combined_text = ""
    page_char_starts = {}
    for pn in range(1, len(page_list) + 1):
        page_char_starts[pn] = len(combined_text)
        combined_text += page_list[pn - 1][0]

    if not combined_text:
        return

    # Collect all nodes (flat list)
    all_nodes = structure_to_list(node) if isinstance(node, list) else structure_to_list([node])

    # Sort nodes by document position (start_index, then structure depth)
    # so the sequential title search processes them in the order they
    # actually appear in the document, not depth-first tree order.
    # Use structure field length as a proxy for depth to break ties.
    def _doc_order_key(n):
        si = n.get('start_index') or 0
        struct = str(n.get('structure', ''))
        depth = struct.count('.') if struct else 0
        return (si, depth)
    all_nodes.sort(key=_doc_order_key)

    # Find each node's title in the global text, sequentially
    node_offsets = []
    search_start = 0
    for n in all_nodes:
        title = n.get('title', '')
        page_hint = n.get('start_index')
        hint_offset = page_char_starts.get(page_hint, search_start) if page_hint else search_start
        effective_start = max(search_start, hint_offset - 500) if hint_offset > 0 else search_start
        effective_start = max(0, effective_start)

        offset = find_title_offset(combined_text[effective_start:], title)
        if offset is not None:
            offset += effective_start
            search_start = offset + 1
        elif page_hint:
            # Fallback: multi-column PDFs can place headings out of
            # document order.  Search again from the page hint without
            # the sequential constraint.
            fallback_start = max(0, page_char_starts.get(page_hint, 0) - 500)
            offset = find_title_offset(combined_text[fallback_start:], title)
            if offset is not None:
                offset += fallback_start
                # Don't advance search_start — we went backward
        node_offsets.append((n, offset))

    # Sort by found offset (nodes whose titles weren't found go to end)
    found = [(n, off) for n, off in node_offsets if off is not None]
    not_found = [(n, off) for n, off in node_offsets if off is None]
    found.sort(key=lambda x: x[1])

    # Assign boundaries: each node gets text from its offset to the next
    for i, (n, off) in enumerate(found):
        n['_split_combined'] = combined_text
        n['_split_start'] = off
        if i + 1 < len(found):
            n['_split_end'] = found[i + 1][1]
        else:
            n['_split_end'] = len(combined_text)

    # Fallback for nodes whose titles weren't found: use page-level ranges
    for n, _ in not_found:
        sp = n.get('start_index')
        ep = n.get('end_index')
        n['_split_combined'] = combined_text
        n['_split_start'] = page_char_starts.get(sp, 0) if sp else 0
        if ep and ep + 1 in page_char_starts:
            n['_split_end'] = page_char_starts[ep + 1]
        else:
            n['_split_end'] = len(combined_text)


def _get_text_of_specific_pages(pdf_pages, page_numbers):
    """Extract text for a specific set of (1-based) page numbers."""
    text = ""
    for pn in sorted(page_numbers):
        text += pdf_pages[pn - 1][0]
    return text


def add_node_text_deduped(node, pdf_pages):
    """Assign text to nodes with page-level sibling dedup and parent-exclusive ranges.

    - Sibling dedup: among siblings, each physical page is assigned to the first
      sibling that claims it. Later siblings get only their unclaimed pages.
    - Parent text: a node with children gets only the pages NOT covered by any
      child (the exclusive prefix range). If there are no exclusive pages, text = "".
    - Leaf text: pages in range minus any already claimed by a prior sibling.
    """
    if isinstance(node, dict):
        _assign_text_single(node, pdf_pages)
        if 'nodes' in node:
            add_node_text_deduped(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        # Page-level dedup: track individual pages claimed within this sibling list
        seen_pages = set()
        for item in node:
            _assign_text_single(item, pdf_pages, seen_pages=seen_pages)
            if 'nodes' in item:
                add_node_text_deduped(item['nodes'], pdf_pages)


def _assign_text_single(node, pdf_pages, seen_pages=None):
    """Assign text to a single node using title-based split annotations when
    available, falling back to page-level dedup."""
    start_page = node.get('start_index')
    end_page = node.get('end_index')

    if start_page is None or end_page is None:
        node['text'] = ""
        return

    all_pages = set(range(start_page, end_page + 1))

    # --- Title-based splitting (preferred) ---
    split_combined = node.pop('_split_combined', None)
    split_start = node.pop('_split_start', None)
    split_end = node.pop('_split_end', None)
    node.pop('_split_pages', None)

    # Also pop legacy offset annotations if present
    node.pop('_start_offset', None)
    node.pop('_end_offset', None)

    if split_combined is not None and split_start is not None and split_end is not None:
        if 'nodes' in node and node['nodes']:
            # Parent node with children: text from this node's title to the
            # first child's title.  With the global split approach, all offsets
            # are in the same coordinate space.
            children = node['nodes'] if isinstance(node['nodes'], list) else [node['nodes']]
            first_child_start = None
            for c in children:
                cs = c.get('_split_start')
                if cs is not None:
                    if first_child_start is None or cs < first_child_start:
                        first_child_start = cs
            if first_child_start is not None and first_child_start > split_start:
                node['text'] = split_combined[split_start:first_child_start]
            else:
                node['text'] = ""
        else:
            # Leaf node: extract the title-to-title slice
            node['text'] = split_combined[split_start:split_end]

        if seen_pages is not None:
            seen_pages.update(all_pages)
        return

    # --- Fallback: page-level dedup (original behaviour) ---
    if 'nodes' in node and node['nodes']:
        children = node['nodes'] if isinstance(node['nodes'], list) else [node['nodes']]
        min_child_start = min(
            c.get('start_index', start_page) for c in children if isinstance(c, dict)
        )
        exclusive_pages = set(range(start_page, min_child_start))
        candidate_pages = exclusive_pages - seen_pages if seen_pages is not None else exclusive_pages
    else:
        candidate_pages = all_pages - seen_pages if seen_pages is not None else all_pages

    text_parts = []
    for pn in sorted(candidate_pages):
        text_parts.append(pdf_pages[pn - 1][0])

    node['text'] = ''.join(text_parts) if text_parts else ""

    if seen_pages is not None:
        seen_pages.update(all_pages)


def recover_empty_nodes(tree, page_list):
    """Safety-net pass: fill in text for any node left empty after dedup.

    For each node with ``text == ""``, attempts title-based offset search
    within its page range.  Falls back to the full page-range text so that
    no node is ever left completely empty (slightly wrong text is better than
    none for RAG retrieval).
    """
    if isinstance(tree, list):
        for item in tree:
            recover_empty_nodes(item, page_list)
    elif isinstance(tree, dict):
        has_children = bool(tree.get('nodes'))
        if not tree.get('text', '').strip() and not has_children:
            # Only recover leaf nodes — parent nodes with empty text
            # legitimately have all content in their children.
            start = tree.get('start_index')
            end = tree.get('end_index')
            if start is not None and end is not None and 1 <= start <= len(page_list):
                end = min(end, len(page_list))
                raw = get_text_of_pdf_pages(page_list, start, end)
                title = tree.get('title', '')
                offset = find_title_offset(raw, title) if title else None
                if offset is not None:
                    tree['text'] = raw[offset:]
                else:
                    tree['text'] = raw
                log.debug("recover_empty_nodes: filled node '%s' (%d chars)", title, len(tree['text']))
        if has_children:
            recover_empty_nodes(tree['nodes'], page_list)


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(node, model=None):
    title = node.get('title', 'this section')
    prompt = f"""Summarize the following section of a document in 2-3 sentences. Focus on the key points, methods, and findings. Write the summary as direct factual statements. Do NOT use phrases like "the partial document", "this section discusses", or "the document covers".

Section title: {title}

Section text: {node['text']}

Directly return the summary, do not include any other text."""
    response = await ChatGPT_API_async(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = []
    nodes_with_text = []
    for node in nodes:
        if node.get('text', '').strip():
            tasks.append(generate_node_summary(node, model=model))
            nodes_with_text.append(node)
        else:
            node['summary'] = ""
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes_with_text, summaries):
        node['summary'] = summary
    return structure


def create_clean_structure_for_description(structure):
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.
    """
    if isinstance(structure, dict):
        clean_node = {}
        # Only include essential fields for description
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        # Recursively process child nodes
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    """
    response = ChatGPT_API(model, prompt)
    return response


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)