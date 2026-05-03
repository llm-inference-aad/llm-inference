# RAG prompt POC (baseline vs augmented)

Small script: read **one mutation prompt per line** from a text file, build the
**RAG-augmented** version using your existing **`rag_data/`**, call the same
**`/generate`** server as the main stack, then write a **chart** and **HTML**
report.

## What you need

1. **`rag_data/`** populated (`uv run python scripts/setup_rag.py` once).
2. **LLM server running** — same discovery as everywhere else: `RUN_ID` +
   `runs/<RUN_ID>/logs/hostname.log`, or `LLM_SERVER_URL`, or `--server-url`.

## Run

```bash
cd /path/to/llm-inference
export PYTHONPATH=src

uv run python scripts/offline_rag_eval/poc_prompt_compare.py \
  --prompts-file scripts/offline_rag_eval/example_poc_prompts.txt \
  --output-dir runs/${RUN_ID}/rag_poc_out \
  --run-id "${RUN_ID}"
```

Outputs under `--output-dir`:

| File | Purpose |
|------|---------|
| `rendered.jsonl` | Baseline + RAG prompt rows (inspect RAG text here). |
| `results.jsonl` | Server responses + `evaluationScore` when the server sends it. |
| `poc_metrics.png` | Bar chart of scores per prompt index. |
| `report.html` | Opens in a browser: chart + table + expandable baseline/RAG text. |

**Scores in the chart:** prefer the server’s **`evaluationScore`** in the JSON
response; if missing, the script uses the same local **`OutputEvaluator`**
logic as `run_offline_eval.py` on the returned `generated_text`.

## Prompt file rules

- UTF-8, **one prompt per line** (multi-line prompts: join to one line or split into multiple entries).
- Empty lines and lines starting with **`#`** are ignored.
- Default cap **10** lines (`--max-prompts`).

## Options

`--rag-dir` overrides `RAG_DATA_DIR` / default `<repo>/rag_data`.  
`--no-code-context` / `--no-text-context` toggle retrieval namespaces.

See `poc_prompt_compare.py --help`.
