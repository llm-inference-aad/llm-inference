import os
import numpy as np
import torch

#: Root directory of the repository
ROOT_DIR = os.environ.get('LLM_INFERENCE_ROOT_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#: Active run identifier used to namespace logs/metrics.
RUN_ID = os.environ.get("RUN_ID", "server-only")
#: Directory for this run's artifacts.
RUN_DIR = os.environ.get("RUN_DIR", os.path.join(ROOT_DIR, "runs", RUN_ID))
#: Directory for consolidated logs (server, slurm, health checks, host files).
RUN_LOG_DIR = os.environ.get("RUN_LOG_DIR", os.path.join(RUN_DIR, "logs"))
#: Directory for analyzable metrics and evaluation outputs.
RUN_METRICS_DIR = os.environ.get("RUN_METRICS_DIR", os.path.join(RUN_DIR, "metrics"))
#: Directory for error logs (like validation_errors.csv)
RUN_ERRORS_DIR = os.environ.get("RUN_ERRORS_DIR", os.path.join(RUN_LOG_DIR, "errors"))
#: DATA_PATH absolute or relative to ExquisiteNetV2
DATA_PATH = "./cifar10"
#: Location where the current seed repo resides
SOTA_ROOT = os.path.join(ROOT_DIR, 'sota/ExquisiteNetV2')
#: Location where the network architecture for the seed resides
SEED_NETWORK = os.path.join(SOTA_ROOT, "network.py")
#: Directory for aggregated Slurm eval logs
SLURM_LOG_DIR = os.environ.get("SLURM_LOG_DIR", os.path.join(RUN_LOG_DIR, "eval"))
#: Directory for aggregated Slurm error logs
SLURM_ERROR_DIR = os.environ.get("SLURM_ERROR_DIR", RUN_ERRORS_DIR)
#: Whether to run llm-ge locally (True) or distribute across a slurm cluster  (False)
# For RAG TESTING: Set to True for simpler debugging, serial execution
# For BASELINE/PARALLEL: Set to False for parallel evaluation (requires Phase 2 modifications)
LOCAL = False
if LOCAL:
	RUN_COMMAND = 'bash'
	DELAYED_CHECK = False
else: 
	RUN_COMMAND = 'sbatch'
	DELAYED_CHECK = True
	
#: Whether to run LLM operations directly via HTTP (True) or submit via sbatch (False)
#: When running distributed (LOCAL=False), we want to avoid sbatch for simple HTTP requests
LLM_DIRECT_HTTP = os.getenv("LLM_DIRECT_HTTP", str(not LOCAL)).lower() in ("true", "1", "yes")

#: Whether host uses macOS (True) and should use mps, or not (False) and should use cpu or cuda depending on what is available
MACOS = False
if torch.mps.is_available():
	DEVICE = 'mps'
	MACOS = True
elif torch.cuda.is_available():
	DEVICE = 'cuda'
else:
	DEVICE = 'cpu'

#LLM_MODEL = 'mixtral'
#LLM_MODEL = 'llama3'
#: LLM Model to use. Choices currently include ['gemini', 'mixtral', 'llama3', 'local_server']
LLM_MODEL = 'local_server'
try:
	GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
except:
	GEMINI_API_KEY = ''
# Surya: Retry budget for LLM code generation (tune to trade off diversity vs. reliability)
LLM_GENERATION_MAX_RETRIES = int(os.environ.get('LLM_GENERATION_MAX_RETRIES', 3))
# Centralized epoch configuration (overrides scattered hard-coded values)
try:
	TRAIN_EPOCHS = int(os.environ.get('EPOCHS', '24').strip())
except Exception:
	TRAIN_EPOCHS = 24
# SEED_PACKAGE_DIR = "./sota/ExquisiteNetV2/divine_seed_module"


def _parse_optional_float(value: str | None) -> float | None:
	if value is None or value == "":
		return None
	try:
		return float(value)
	except ValueError:
		return None


#: Retrieval-Augmented Generation (RAG) configuration
RAG_ENABLED = os.environ.get("RAG_ENABLED", "true").lower() in {"1", "true", "yes"}
RAG_DATA_DIR = os.environ.get("RAG_DATA_DIR", os.path.join(ROOT_DIR, "rag_data"))
RAG_CODE_EMBED_MODEL = os.environ.get("RAG_CODE_EMBED_MODEL", "microsoft/codebert-base")
RAG_TEXT_EMBED_MODEL = os.environ.get("RAG_TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", 5))
RAG_MIN_ACCURACY = float(os.environ.get("RAG_MIN_ACCURACY", 0.9))
RAG_MAX_PARAMETERS = _parse_optional_float(os.environ.get("RAG_MAX_PARAMETERS"))
RAG_MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIMILARITY", 0.3))  # Minimum similarity threshold for filtering irrelevant results
RAG_TEXT_TOP_K = int(os.environ.get("RAG_TEXT_TOP_K", 3))  # Number of text chunks (PDFs, docs) to retrieve
RAG_TEXT_CANDIDATE_K = int(os.environ.get("RAG_TEXT_CANDIDATE_K", 24))
RAG_TEXT_TOP_K_API = int(os.environ.get("RAG_TEXT_TOP_K_API", 2))
RAG_TEXT_TOP_K_PDF = int(os.environ.get("RAG_TEXT_TOP_K_PDF", 1))
RAG_USE_CODE_CONTEXT = os.environ.get("RAG_USE_CODE_CONTEXT", "true").lower() in {"1", "true", "yes"}
RAG_USE_TEXT_CONTEXT = os.environ.get("RAG_USE_TEXT_CONTEXT", "true").lower() in {"1", "true", "yes"}
RAG_RERANKER_ENABLED = os.environ.get("RAG_RERANKER_ENABLED", "false").lower() in {"1", "true", "yes"}
RAG_RERANKER_MODEL = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
#: Retrieval backend selector for RagService default construction.
#: Supported values: "faiss", "graph" (pageindex still pending port).
RAG_BACKEND = os.environ.get("RAG_BACKEND", "faiss").strip().lower()
#: Memory backend: episodic summaries of past mutations (successes + failures).
RAG_MEMORY_STORE_ENABLED = os.environ.get("RAG_MEMORY_STORE_ENABLED", "false").lower() in {"1", "true", "yes"}
RAG_MEMORY_TOP_K = int(os.environ.get("RAG_MEMORY_TOP_K", 3))
RAG_MEMORY_MIN_SIMILARITY = float(os.environ.get("RAG_MEMORY_MIN_SIMILARITY", 0.5))

# --- Pareto-aware mutation logging policy ---------------------------------- #
#: Controls which events get the is_pareto_eligible=True flag.
#: "pareto"   — per-generation percentile windows (default, recommended).
#: "absolute" — falls back to RAG_MIN_ACCURACY / RAG_MAX_PARAMETERS thresholds.
RAG_LOG_POLICY: str = os.environ.get("RAG_LOG_POLICY", "pareto").lower()
#: Top-N% of test_accuracy within the generation that are marked eligible.
#: Uses math.ceil for inclusivity (e.g. 10% of 7 = ceil(0.7) = 1).
RAG_LOG_TOP_ACCURACY_PCT: float = float(os.environ.get("RAG_LOG_TOP_ACCURACY_PCT", 10.0))
#: Bottom-N% of total_params within the generation that are marked eligible.
#: Uses math.ceil for inclusivity.
RAG_LOG_BOTTOM_PARAMS_PCT: float = float(os.environ.get("RAG_LOG_BOTTOM_PARAMS_PCT", 10.0))

# Evolution Constants/Params
# --------------------------

#: Tuple of fitness weights of length equal to the number of objectives.
#: 1.0 indicates objective will be maximized, -1.0 for objective to by minimized.
FITNESS_WEIGHTS = (1.0, -1.0)
INVALID_FITNESS_MAX = tuple([float(x*np.inf*-1) for x in FITNESS_WEIGHTS])
# this is just a unique value
PLACEHOLDER_FITNESS = tuple([int(x*9999999999*-1) for x in FITNESS_WEIGHTS])

#: Number of elite individuals to utilize within the Evolution of Thought (EOT) operation
NUM_EOT_ELITES = 2

#: Cycle in the optimization and output directory where intermediate data will be stored.
GENERATION = 0

PROB_QC = 0.0
PROB_EOT = 0.25

# =============================================================================
# BASELINE CONFIGURATION FOR LLM INFERENCE OPTIMIZATION
# =============================================================================
# Recommended settings for establishing baseline metrics before applying
# optimization techniques (RAG, speculative decoding, etc.)
#
# For baseline run:
# - num_generations: 10-15 (enough for statistical significance)
# - population_size: 8-16 (matches batch size for efficient batching)
# - LOCAL: False (parallel evaluation, better for throughput measurement)
# ========================================================

#: Number of generations to run for
num_generations = int(os.environ.get("NUM_GENERATIONS", 15))  # BASELINE: 15 generations for initial experiments

#: Population size for launching optimization
start_population_size = int(os.environ.get("START_POPULATION_SIZE", 16))  # BASELINE: 8 genes (matches BATCH_SIZE in server.py)

#: Population size to utilize in each generation after optimization begins
population_size = int(os.environ.get("POPULATION_SIZE", 16))  # BASELINE: Keep consistent with start_population_size

crossover_probability = 0.35  #: Probability of mating two individuals
mutation_probability = 0.8 	  #: Probability of mutating an individual

#: Number of elites to consider
num_elites = 8
#: Number of individuals to keep in the hall of fame across the optimization
hof_size = 8


# Job Sub Constants/Params
# ------------------------


#: Whether (True) or not (False) you wish to run quality control checks on responses from the LLM
QC_CHECK_BOOL = False
#: Whether (True) or not (False) to submit LLM prompts remotely to sources such as hugging face.
INFERENCE_SUBMISSION = False  # Use local server
#LLM_GPU = 'NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100-PCIE-32GB|QuadroRTX4000|GeForceGTX1080Ti|GeForceGTX1080|TeslaV100-PCIE-32GB|TeslaV100S-PCIE-32GB'
#LLM_GPU = 'NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe|TeslaV100-PCIE-32GB|TeslaV100S-PCIE-32GB|NVIDIARTX6000AdaGeneration|NVIDIARTXA6000|NVIDIARTXA5000|NVIDIARTXA4000|GeForceGTX1080Ti|QuadroRTX4000|QuadroP4000|GeForceGTX1080|TeslaP4'
#: If using slurm, this string will be used to request GPUs for the submission of prompts to the LLM.
LLM_GPU = 'A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S'

#: Template script for submitting job for evaluation.
PYTHON_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "nvidia-gpu"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output={slurm_log_dir}/eval-%j.out
#SBATCH --error={slurm_error_dir}/eval-%j.err
echo "Launching Python Evaluation"
hostname

# Load GCC version 9.2.0
# module load gcc/13.2.0
module load cuda
# module load anaconda3
# Activate Conda environment
# conda activate llm_guided_env
# export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# conda info

export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.13/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
source "$VENV_PATH/bin/activate"

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Change to repository root directory to ensure consistent paths
cd "${{LLM_INFERENCE_ROOT_DIR:-{root_dir}}}"

# Time the evaluation
EVAL_START=$(date +%s)

# Run Python script
{python_runline}
EXIT_CODE=$?

EVAL_END=$(date +%s)
EVAL_ELAPSED=$((EVAL_END - EVAL_START))
echo "EVAL_TIME_SECONDS=$EVAL_ELAPSED" >> {slurm_log_dir}/eval-$SLURM_JOB_ID.time
echo "EXIT_CODE=$EXIT_CODE" >> {slurm_log_dir}/eval-$SLURM_JOB_ID.time

exit $EXIT_CODE
"""

# modify the script to use .env 
#: Template script for submitting a prompt to the LLM
LLM_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C {gpu_constraint}
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --output={slurm_log_dir}/llm-%j.out
#SBATCH --error={slurm_error_dir}/llm-%j.err
echo "Launching AIsurBL"
hostname

# Load modules
module load cuda
module load python/3.12.5


# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate virtual environment and set library paths
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

# Run Python script with uv
{python_runline}
"""


# Helper Functions
# ----------------

def ensure_slurm_log_dir():
    """
    Ensure run-scoped logging/metrics directories exist.
    This should be called during program startup/configuration rather than at import time.
    """
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    os.makedirs(RUN_METRICS_DIR, exist_ok=True)
    os.makedirs(RUN_ERRORS_DIR, exist_ok=True)
    os.makedirs(SLURM_LOG_DIR, exist_ok=True)
    os.makedirs(SLURM_ERROR_DIR, exist_ok=True)


# Misc. Non-sense
# ---------------

DNA_TXT = """
⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣶⣶⠶⣶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣀⣹⣟⣛⣛⣻⣿⣿⣿⡾⠟⢉⣴⠟⢁⣴⠋⣹⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠈⠛⠛⣿⠉⢉⣩⠵⠚⠁⢀⡴⠛⠁⣠⠞⠁⣰⠏⠸⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢻⣷⠋⠁⠀⢀⡴⠋⠀⢀⡴⠋⠀⣼⠃⠀⡼⢿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢻⣆⣠⡴⠋⠀⠀⣠⠟⠀⢀⡾⠁⠀⡼⠁⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠻⣯⡀⠀⢀⡼⠃⠀⢠⡟⠀⢀⡾⠁⢀⣾⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠻⣶⣟⡀⠀⣰⠏⠀⢀⡾⠁⠀⣼⢹⣿⣀⣤⣤⣴⠶⢿⡿⠛⢛⣷⢶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠶⠶⠾⠷⠶⠿⠛⢻⣟⠉⣥⠟⠁⣠⠟⠀⢠⠞⠁⣄⡿⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⠞⠁⢀⡴⠋⠀⣴⠋⠀⣰⠟⠀⣤⡾⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡄⢠⠞⠁⢀⡾⠁⢀⡼⠃⢀⡴⠋⠀⢸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣷⠋⠀⣰⠏⠀⣠⠟⠀⣰⠟⠁⢀⡴⠛⣿⠀⠀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣧⡼⠃⢀⡼⠋⢠⡞⠁⣠⣞⣋⣤⣶⣿⡟⠛⣿⠛⠛⣻⠟⠷⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⣦⣾⣤⣴⣯⡶⠾⠟⠛⠉⠉⠉⣿⡇⢠⡏⠀⣰⠏⠀⢀⣼⠋⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⡾⠀⢰⠏⠀⢠⡞⠁⠀⣠⠞⢻⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣷⠇⢠⠏⠀⣰⠋⠀⣠⠞⠁⠀⢀⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⢠⠟⢀⡼⠁⣠⠞⠁⣀⣴⢾⣿⣤⣿⣦⣄⣀⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡟⣠⠏⣠⠞⣁⣴⣾⣿⣿⣿⣿⣿⣿⡏⢹⡏⠛⠳⣦⣄⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢷⣾⣷⠿⠿⠛⠉⠀⠀⠈⠳⣬⣿⡟⣾⠁⠀⣼⠃⠉⠻⠆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣧⡏⠀⣼⠃⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠁⡼⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⡼⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⣃⠀⠀
"""
