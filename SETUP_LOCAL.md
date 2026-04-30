# Local Setup Guide

After cloning the repository, the virtual environment has been removed to reduce download size. Follow these steps to set up locally:

## Quick Setup

```bash
# Navigate to project
cd llm-inference

# Create Python 3.10+ virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from pyproject.toml
pip install -e .

# Initialize RAG dataset (optional, for inference)
python setup_rag_local.py
```

## Requirements

- **Python 3.10+** (project requires 3.12+, but 3.10 is supported for RAG setup)
- **~5-10 GB** disk space for venv + dependencies
- **CUDA 12.x** or CPU (for PyTorch)

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Source code (LLM utils, RAG pipeline, evolution framework) |
| `runs/` | Benchmark results (vllm_300request has complete metrics) |
| `rag_data/` | FAISS vector database (51 mutations + 22 document chunks) |
| `rag_corpus/` | 3 research PDFs for RAG context |
| `sota/` | ExquisiteNetV2 model code reference |
| `docs/` | Documentation and architecture diagrams |
| `scripts/` | Utility scripts (setup_rag.py, run_dashboard.sh, etc.) |

## What's Included in the RAG Dataset

- **51 mutation records** from historical evolution runs
- **22 document chunks** from 3 PDFs (PyTorch docs, NAS survey, CIFAR-10 papers)
- Pre-built FAISS indices for code and text namespaces
- Golden query set for retrieval evaluation

## Key Files

- `pyproject.toml` — Dependencies and project metadata
- `setup_rag_local.py` — Bootstrap RAG vector database (no broken imports)
- `README.md` — Project overview and paper references
- `.env.example` — Template for environment variables

## Troubleshooting

**Import errors in RAG modules?**
Use `setup_rag_local.py` instead of `scripts/setup_rag.py` — it handles circular imports properly.

**Missing dependencies?**
```bash
pip install --upgrade pdfplumber sentence-transformers faiss-cpu
```

**Need to regenerate RAG database?**
```bash
rm -rf rag_data/
python setup_rag_local.py
```
