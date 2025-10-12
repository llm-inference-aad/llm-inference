# Onboarding Document for ExquisiteNetV2 LLMGE on CIFAR10

## Principle Assumptions

- **Assumption 1 – GA manipulates ExquisiteNetV2 on CIFAR10**: Confirmed via the DEAP-driven loop that mutates networks under `sota/ExquisiteNetV2`. Validation: Individuals are ExquisiteNet variants evaluated with CIFAR10 data.
    - `run_improved.py:900`
    - `run_improved.py:913`
    - `src/cfg/constants.py:10`
    - `src/cfg/constants.py:8`

- **Assumption 2 – “Code blocks” can yield valid or invalid individuals**: True; each block is a `# --OPTION--` segment selected from the seed network for LLM rewriting. Validation: `slurm-3101714.out` shows several LLM responses that saved malformed Python yet were marked complete, demonstrating invalid individuals emerge when QC is off.
    - `src/llm_mutation.py:20`
    - `src/llm_utils.py:113`
    - `slurm-3101714.out`

- **Assumption 3 – Fitness drives validity**: Confirmed; evaluation reads `{gene}_results.txt`, assigns tuple fitness, or penalizes failures with `INVALID_FITNESS_MAX`. Validation: Individuals without results are forced to max-penalty fitness.
    - `run_improved.py:414`
    - `run_improved.py:424`
    - `run_improved.py:480`
    - `src/cfg/constants.py:48`

- **Assumption 4 – Crossover of high-performers**: Mostly true; selection leverages SPEA2/NSGA-II to bias toward Pareto-strong parents before applying crossover with probability 0.35. Validation: While not restricted to elites, the selection pressure favors better individuals.
    - `run_improved.py:874`
    - `run_improved.py:897`
    - `run_improved.py:900`
    - `src/cfg/constants.py:72`

- **Assumption 5 – Mutations maintain diversity**: Confirmed by 0.8 mutation probability and LLM-based rewriting of selected code blocks. Validation: Mutation pipeline resubmits altered blocks, replacing prior genes.
    - `src/cfg/constants.py:73`
    - `run_improved.py:929`
    - `src/llm_mutation.py:31`

- **Assumption 6 – Objectives maximize/minimize as defined**: Yes; `FITNESS_WEIGHTS` control objective direction (test accuracy maximized, params minimized) and downstream reporting respects these weights. Validation: Fitness tuples follow the configured weight signs.
    - `src/cfg/constants.py:47`
    - `src/utils/print_utils.py:24`

## Code Block Role

- Network files are partitioned by `# --OPTION--`; mutation/crossover selects one segment, formats it into an LLM prompt template, and writes the model back after replacement. Validation: Observed prompts and outputs in `slurm-3101714.out` match this segmentation strategy.
    - `src/llm_mutation.py:31`
    - `src/llm_crossover.py:40`
    - `templates/ConstantRules.txt:1`
    - `slurm-3101714.out`

## Architecture Overview

- **Orchestration** (`run_improved.py:840`, `run_improved.py:189`) drives population lifecycle, job submission, checkpointing, and ancestry tracking.
- **LLM interfaces** (`src/llm_mutation.py:16`, `src/llm_crossover.py:17`, `src/llm_utils.py:72`) encapsulate prompt construction, local/remote inference, optional QC, and prompt mutation.
- **Evaluation harness** `sota/ExquisiteNetV2/train.py:279` trains candidate models, emits metrics for fitness, and stores artifacts under `sota/ExquisiteNetV2/results/`.
- **Configuration layer** blends `.env` overrides with defaults in `src/cfg/constants.py:5` to control data paths, GA hyperparameters, and job templates.
- **Local inference server** (`server.py:11`) loads the HF model defined by environment variables, batches requests, and services `submit_local_server`.

## Evolutionary Workflow

1.  **Initialize or resume population**, seeding individuals from the ExquisiteNet baseline (`run_improved.py:860`).
2.  For each individual, **submit mutation scripts** to produce `network_{gene}.py` via LLM prompts (`run_improved.py:151`, `src/llm_mutation.py:31`).
3.  **Train each generated network** and capture fitness metrics into results files (`run_improved.py:414`, `sota/ExquisiteNetV2/train.py:264`).
4.  **Select next-generation parents** with SPEA2/NSGA-II, apply crossover/mutation probabilistically, and requeue LLM jobs (`run_improved.py:900`, `run_improved.py:915`, `run_improved.py:929`).
5.  **Update fitness**, maintain ancestry/history, mutate prompt templates for exploration, and checkpoint state (`run_improved.py:964`, `run_improved.py:972`, `src/llm_utils.py:190`).

## Key Elements Table

| **Component** | **Location** | **Purpose** | **Notes** |
| --- | --- | --- | --- |
| GA Driver | `run_improved.py:840` | Controls population loop, selection, variation, fitness checks | Uses DEAP NSGA-II tooling |
| LLM Mutation | `src/llm_mutation.py:16` | Randomly selects a code block and rewrites it via prompt template | Relies on generated `template_txt` saved per gene |
| LLM Crossover | `src/llm_crossover.py:17` | Combines differing blocks from two parent networks | Pulls paired templates from `templates/CrossOver/` |
| LLM Utilities | `src/llm_utils.py:72` | Provides prompt execution, QC hooks, prompt mutation routines | References missing QC template path |
| Evaluation Script | `sota/ExquisiteNetV2/train.py:279` | Trains candidate model, logs test acc, params, val acc, runtime | Outputs CSV-like metrics per gene |
| Local LLM Server | `server.py:41` | Hosts HF model for low-latency prompt execution | Persists hostname to coordinate with clients |

## Configuration & Parameters

- **Environment defaults** for root paths, model server, and GA flags derive from `.env` with fallbacks in `src/cfg/constants.py:5`; see `.env.example` for cluster-specific overrides.
- **GA hyperparameters**—population size, generation count, crossover/mutation rates—are defined in constants and can be environment-patched if needed (`src/cfg/constants.py:65`, `src/cfg/constants.py:72`).
- **Job submission templates** embed GPU constraints and environment activation, customizable via `.env` variables like `VENV_PATH`, `RUN_COMMAND` (`src/cfg/constants.py:93`, `run.sh:33`).
- **LLM behavior toggles** include `LLM_MODEL`, `INFERENCE_SUBMISSION`, QC probability, and prompt mutation count (`src/cfg/constants.py:35`, `src/cfg/constants.py:84`, `src/llm_utils.py:190`).

## Dependencies & I/O

- **Core libraries**: DEAP, PyTorch, Transformers, FastAPI, requests, huggingface-hub (`pyproject.toml:9`).
- **Inputs**: seed network `sota/ExquisiteNetV2/network.py`, CIFAR10 dataset (`src/cfg/constants.py:8`), LLM templates under `templates/`.
- **Outputs**: mutated networks in `sota/ExquisiteNetV2/models/`, training metrics in `sota/ExquisiteNetV2/results/`, per-gene prompt logs in `0/<gene>_model.txt`, checkpoints in `checkpoints/`.
- **Log artifacts**: Slurm outputs (`slurm-3101714.out`) capture prompt transcripts, QC status, and job completion markers.

## Findings & Risks

- **Quality control templates** referenced in code are absent (`src/llm_utils.py:135`), so QC toggles currently fail; reinstate or update template paths.
- **LLM responses often return natural-language prose** instead of code (see `slurm-3101714.out`), causing invalid networks yet still being evaluated; cleaning or stricter validation is essential.
- **Mutation relies on random block selection**; no safeguard ensures coverage or structural soundness, which may explain large invalid ratios.
- **Timeout handling** in `check_and_update_fitness` uses long loops; consider shorter `loop_delay` for responsive monitoring in local runs.

## Immediate Goal – Pareto Front Visualization

- **Fitness tuples** already capture (test accuracy, parameter count) per gene (`run_improved.py:418`); aggregate per-generation results into a dataframe and plot the non-dominated set (e.g., using matplotlib/scikit-learn’s `approximate_front`). Harvest data from checkpoints or `GLOBAL_DATA_HIST` snapshots.
- **Ensure consistent logging** by storing `(accuracy, params, gene_id, generation)` after each evaluation; a simple CSV export in `check_and_update_fitness` would enable Pareto plotting without reruns.

## Long-Term Initiative Alignment

- **Baseline trials** across LLM backends can toggle `LLM_MODEL` and log corresponding Pareto fronts to compare inference-guided evolution outcomes.
- **Introduce RAG** by enriching prompt templates with retrieved documentation snippets prior to LLM calls; store retrieval metadata to compare against baseline fronts.

## Next Steps

1.  **Reintroduce/author the missing QC prompt** (`templates/llm_quality_control.txt`) and enforce syntactic validation before accepting LLM outputs, reducing invalid individuals.
2.  **Instrument evolution runs** to emit structured metrics (CSV/JSON) per generation, enabling Pareto front plotting and downstream comparison across LLM/RAG configurations.
