# LLMGE Pipeline Architecture

LLM-Guided Evolution (LLMGE) is an evolutionary algorithm that uses an LLM to generate and improve neural network architectures. This document describes the per-individual mutation/evaluation cycle.

---

## 1. Run Launch

**Entry point:** `/storage/ice1/6/6/bcharest3/llm-inference/llm-inference/run.sh`

The launch sequence:

1. **run.sh** (lines 70–177):
   - Loads `.env` for config (RAG_ENABLED, NUM_GENERATIONS, POPULATION_SIZE, etc.; see lines 100–122)
   - Allows environment variable overrides (sbatch --export)
   - Submits **server.sh** via sbatch and records SERVER_JOB_ID
   - Calls `uv run python run_improved.py ${RUN_DIR}/checkpoints` (line 177)

2. **server.sh** (lines 25–259):
   - Launches the LLM server (vLLM or HuggingFace backend) on a dedicated GPU job
   - Registers itself in SERVER_REGISTRY_FILE (JSON) for load balancing
   - Auto-assigns port if not specified
   - Runs until main job completes, then is killed by run.sh cleanup (lines 225–275)

3. **run_improved.py** waits for server readiness (lines 1300–1330):
   - `wait_for_server_ready()` polls HOSTNAME_LOG_FILE or LOADBALANCER_LOG_FILE
   - Blocks until HTTP server responds (port from SERVER_PORT env var, default 8000)

**Key environment variables** (sourced from `.env`, overrideable):
- `RAG_ENABLED`: bool, enables RAG context injection
- `RAG_USE_CODE_CONTEXT`, `RAG_USE_TEXT_CONTEXT`, `RAG_RERANKER_ENABLED`: RAG feature flags
- `NUM_GENERATIONS`, `POPULATION_SIZE`, `START_POPULATION_SIZE`: Evolution params
- `EXPERIMENT_SEED`: int, seeds RNGs for reproducibility
- `LLM_DIRECT_HTTP`: bool (default True if not LOCAL), runs LLM ops via thread pool vs sbatch
- `LOCAL`: bool, if True runs serially on CPU; if False distributes jobs via slurm

---

## 2. Individual Representation

**Individual = gene_id** (a 24-character random string, e.g., `xXx<random>`)

### ID Generation
`generate_random_string(length=24)` (run_improved.py:514–520):
- Generates `'xXx' + random_letters_and_digits`
- Called for each new individual (mutation, crossover, initial creation)

### File paths
- **Code:** `${SOTA_ROOT}/models/network_{gene_id}.py`
  - SOTA_ROOT = `${ROOT_DIR}/sota/ExquisiteNetV2`
  - Seed network at `${SOTA_ROOT}/network.py`
- **Results:** `${RUN_DIR}/results/{gene_id}_results.txt` (fitness values)
- **Logs:** `${RUN_LOG_DIR}/llm/gene_{gene_id}.log` (mutation prompts, LLM responses)
- **Fallback marker:** `${SOTA_ROOT}/models/network_{gene_id}.py.fallback` (if fallback triggered)

### Ancestry tracking
Global `GLOBAL_DATA_ANCESTRY[gene_id]`:
```python
{
  'GENES': [parent_gene_id, grandparent_id, ..., gene_id],
  'MUTATE_TYPE': ['SEED' | 'CREATED' | mutation_type_str | 'CrossOver', ...]
}
```
Maintained by `update_ancestry()` (src/evolution/ancestry.py:15–56).

---

## 3. Mutation Operators

Each mutation: parent code → LLM prompt → child code. Two modes:

### 3.1 FixedPrompts (probability 1 - PROB_EOT, default 75%)
**Template selection:**
- `generate_template()` (run_improved.py:152–209) randomly selects from:
  - `templates/FixedPrompts/concise/*/*.txt` (18 templates total)
  - `templates/FixedPrompts/roleplay/*/*.txt` (18 templates total)
- Appends `templates/ConstantRules.txt` (shared constraints)

**Prompt assembly:**
1. Load template (e.g., `templates/FixedPrompts/concise/Param.txt`)
2. Load parent code part at random section index (lines 20–27 of llm_mutation.py)
3. Template filled: `template.format(parent_code_part.strip())`
4. Add ConstantRules.txt suffix
5. Apply RAG context if enabled → `_apply_rag_context()` (run_improved.py:74–95)

**Entrypoint:** `src/llm_mutation.py:augment_network()` (lines 20–111)

### 3.2 EoT (Evolution of Thought)
**Trigger:** `PROB_EOT > random()` and generation > 0 (line 179)

**Mechanism:**
1. Randomly pick a top-performing parent from `TOP_N_GENES` (selected via SPEA2, line 1443)
2. Find first differing code section between top parent and seed network
3. Load `templates/EoT/EoT.txt` and fill: `template.format(differing_from_elite, differing_from_seed, "{}")`
4. Apply RAG context

**Entrypoint:** Same as FixedPrompts (both handled in `generate_template()`)

### 3.3 RAG Context Injection (optional)
**Call site:** run_improved.py:196 (for EoT) and run_improved.py:207 (for FixedPrompts)

**Function:** `_apply_rag_context(template_txt, mutation_type, query_code)` (run_improved.py:74–95)

**Inside:**
- Calls `get_runtime().enhance_template(template, mutation_type, query_code, gene_id=None)`
- Returns augmented template + retrieved mutations list
- Records metric: rag_generation_context (lines 86–94)

---

## 4. Crossover

**Function:** `src/llm_crossover.py:augment_network()` (lines 12–67)

**Inputs:**
- `input_filename_x`, `input_filename_y`: paths to two parent networks
- `output_filename`: path to write child code

**Mechanism:**
1. Split both parents into code sections (via `split_file()`)
2. Find first differing section between parents
3. Randomly select crossover template: `templates/CrossOver/crossover.txt` or `crossover_s.txt`
4. Format: `template.format(parent_x_section, parent_y_section)`
5. Generate augmented code via LLM
6. Write child to output_filename

**Entrypoint in evolution loop:** `customCrossover()` (run_improved.py:1029–1113)
- Called for each pair of offspring (line 1493)
- Generates two children per mating event
- Returns updated individuals

**Note:** Crossover does NOT use RAG context (no call to `_apply_rag_context`).

---

## 5. Evaluation

**Evaluation pipeline:**
1. **Code generation** → `src/llm_mutation.py` or `src/llm_crossover.py` (validates syntax, retries if needed)
2. **Validation** → `validate_module_source()` (llm_utils.py:125–155) executes code to catch runtime errors
3. **Training** → `src/sota/ExquisiteNetV2/train.py` (via SLURM or local subprocess)
4. **Results collection** → `check4results()` (run_improved.py:734–818)

### 5.1 Code Generation with Validation

**Entrypoint:** `generate_augmented_code()` (llm_utils.py:158–225)

**Retry loop (up to LLM_GENERATION_MAX_RETRIES=3):**
1. Submit prompt to LLM (via `submit_local_server()`, `submit_mixtral()`, etc.)
2. Clean response: `clean_code_from_llm()` (lines 66–98)
3. Validate syntax: `_validate_python_snippet()` (lines 101–111)
4. On error, re-prompt with: `_format_retry_prompt()` appends stricter instructions (lines 114–122)
5. On success, return code

**Fallback mechanism:**
- If all retries fail, mutation code (line 88–102) writes `.fallback` marker
- Returns original parent code to guarantee loadability
- Sets `GLOBAL_DATA[gene_id]['fallback'] = True`

### 5.2 Module Instantiation Check
`validate_module_source(source_code, module_path, module_name)` (llm_utils.py:125–155):
- Executes code via `exec(compile(source))`
- Attempts instantiation of all nn.Module subclasses (catches NameError, AttributeError)
- Raises exception if NameError/ImportError found

### 5.3 Training Submission
**Entrypoint:** `check4model2run()` (run_improved.py:645–732) and `submit_run()` (lines 577–637)

**Fitness inheritance optimization (lines 669–726):**
- If child is fallback clone of parent, inherit parent's fitness instead of re-evaluating
- For seed network fallbacks: load from `${SOTA_ROOT}/results/network_results.txt`
- For evolved gene fallbacks: copy from GLOBAL_DATA[parent_gene_id]['fitness']

**Training script invocation:**
- Calls `sota/ExquisiteNetV2/train.py` with network module name: `models.network_{gene_id}`
- Passes data path, epochs (TRAIN_EPOCHS env var), batch size, etc.
- Logs: `${SLURM_LOG_DIR}/eval-{job_id}.out` and `.err`

### 5.4 Results Collection

**Entrypoint:** `check4results()` (run_improved.py:734–818)

**Result file format:** `${SOTA_ROOT}/results/{gene_id}_results.txt`
- Format: comma-separated fitness values, e.g., "0.95,12345" (accuracy, neg_parameters)

**Fitness assignment:**
```python
fitness = (float(results[0]), float(results[1]))  # (accuracy, neg_param_count)
GLOBAL_DATA[gene_id]['fitness'] = fitness
GLOBAL_DATA[gene_id]['status'] = 'completed'
```

**Metrics:**
- Objective 1 (maximize): test accuracy (weight: 1.0)
- Objective 2 (minimize): parameter count (weight: -1.0, negated in results)

**Error handling:**
- Checks `eval-{job_id}.err` for "traceback" → marks fitness INVALID_FITNESS_MAX
- Checks `eval-{job_id}.out` for "Job Done" or error keywords
- Timeout (30 hours): assign INVALID_FITNESS_MAX

---

## 6. Per-Generation Loop

**Main loop:** run_improved.py:1440–1552

```
FOR each generation gen in range(start_gen, num_generations):
  
  A. VERIFY POPULATION
     - check_and_update_fitness(population) [BLOCKING]
       ├─ For each gene with status='subbed file':
       │  ├─ check4model2run() → submit training job or inherit fitness
       │  └─ check4results() → poll results, assign fitness
       └─ Repeat until all fitness values assigned or timeout
  
  B. SELECT ELITES
     - TOP_N_GENES = tools.selSPEA2(population, NUM_EOT_ELITES=2)
     - elites = tools.selSPEA2(population, num_elites=8)
     - offspring = toolbox.select(population, population_size) [NSGA2]
  
  C. MATE (line 1491–1495)
     FOR each pair (child1, child2) in offspring:
       IF random() < crossover_probability (0.35):
         child1, child2 = toolbox.mate(child1, child2)
           → customCrossover(child1, child2)
              ├─ new_gene_id_1 = generate_random_string()
              ├─ new_gene_id_2 = generate_random_string()
              ├─ For each: submit_direct_llm_task('src/llm_crossover.py', ...)
              ├─ Wait for LLM completion if not DELAYED_CHECK
              └─ Update GLOBAL_DATA[new_gene_id] = {'sub_flag', 'job_id', 'status':'subbed file'}
         del child1.fitness.values, del child2.fitness.values
     
     IF DELAYED_CHECK:
       delayed_mate_check(offspring) → poll job completion, update individuals
  
  D. MUTATE (line 1506–1509)
     FOR each mutant in offspring:
       IF random() < mutation_probability (0.8):
         toolbox.mutate(mutant)
           → customMutation(individual)
              ├─ new_gene_id = generate_random_string()
              ├─ generate_template() → select FixedPrompt or EoT
              ├─ submit_direct_llm_task('src/llm_mutation.py', template_txt, ...)
              ├─ Wait for completion if not DELAYED_CHECK
              └─ Update GLOBAL_DATA[new_gene_id] = {'status':'subbed file'}
         del mutant.fitness.values
     
     IF DELAYED_CHECK:
       delayed_mutate_check(offspring) → poll job completion
  
  E. MERGE & DEDUP
     - offspring.extend(elites)
     - offspring = remove_duplicates(offspring)
     - Restore elite history from GLOBAL_DATA_HIST
  
  F. ASSIGN PLACEHOLDER & EVALUATE
     FOR each ind in offspring:
       ind.fitness.values = PLACEHOLDER_FITNESS
     check_and_update_fitness(offspring) [BLOCKING] → repeats step A
  
  G. UPDATE & CHECKPOINT
     - population[:] = offspring
     - hof.update(population)
     - save_checkpoint(gen, checkpoints_dir)
     - mutate_prompts()
```

---

## 7. RAG Hook Points

LLMGE uses RAG to augment prompts with examples of successful prior mutations.

### Import Sites

1. **run_improved.py:18** `from rag.data_ingestion import build_mutation_description, calculate_fitness_improvement`
2. **run_improved.py:19** `from rag.runtime import get_runtime`
3. **src/llm_utils.py:17** `from rag.runtime import get_runtime`

### Call Sites

1. **run_improved.py:74–95** `_apply_rag_context(template_txt, mutation_type, query_code)`
   - Called at line 196 (EoT) and 207 (FixedPrompts)
   - **Operation:** `runtime.enhance_template(template, mutation_type, query_code, gene_id=None)`
   - **Returns:** (augmented_template: str, mutations: List[RetrievedMutation])

2. **run_improved.py:99–149** `_log_mutation_result(gene_id, fitness)`
   - Called at line 876 after successful evaluation
   - **Condition:** fitness ≥ RAG_MIN_ACCURACY (default 0.9) and not fallback
   - **Operations:**
     - `build_mutation_description(gene_id, mutation_type, fitness, improvement)` → narrative
     - `calculate_fitness_improvement(fitness, parent_fitness)` → accuracy_delta, parameters_delta
     - `runtime.log_mutation_code(content, metadata)` → stores code + metadata in vector DB
   - **Metadata:** gene_id, parent_id, mutation_type, fitness, improvement, description

3. **src/llm_utils.py:246–270** `_augment_template_with_rag(template_text, mutation_label, query_code)` (unused in current flow)
   - Same operation as _apply_rag_context but with different name

4. **src/llm_utils.py:273–299** `_prepend_rag_context_to_prompt(prompt_text, mutation_label)` (unused)
   - Calls `runtime.collect_context(mutation_type, query_code)`
   - **Returns:** List[RetrievedMutation]
   - **Operation:** `runtime.format_context(mutations)` → formatted instruction block

### RAG Runtime Internals (reference only)

**File:** `src/rag/runtime.py`

**Class:** `RagRuntime`

**Methods:**
- `__init__()` (lines ~20–60): Initialize embeddings, vector store, prompt enhancer
- `enhance_template(template, mutation_type, query_code, gene_id)` (line 65–75): Retrieve relevant mutations, inject into prompt
- `log_mutation_code(content, metadata)` (line 77–80): Embed code, store in vector DB
- `collect_context(mutation_type, query_code)` (line 86–88): Retrieve mutations without prompt injection
- `format_context(mutations)` (line 90–92): Convert mutations to readable instruction block

**Singleton:** `get_runtime() -> RagRuntime | None` (line 99+)
- Returns None if RAG_ENABLED=false
- Thread-safe singleton via `_runtime_lock`, `_runtime_instance`

---

## 8. Templates Inventory

### FixedPrompts (Concise, 10 templates)
- `templates/FixedPrompts/concise/Complex.txt` – Modify complex component
- `templates/FixedPrompts/concise/Param.txt` – Adjust hyperparameters
- `templates/FixedPrompts/concise/ParamWeird.txt` – Unconventional hyperparameter tweaks
- `templates/FixedPrompts/concise/RemoveParams.txt` – Reduce parameter count
- `templates/FixedPrompts/concise/Significant.txt` – Make significant architectural changes
- `templates/FixedPrompts/concise/Weird.txt` – Experimental/creative mutations
- `templates/FixedPrompts/concise/mutant{0,1,2,3,4}.txt` – Generic mutation variants

### FixedPrompts (Roleplay, 15 templates)
Role-based prompts (Expert, MadScientist, Helper, MrMagoo) × 5 mutation categories:
- `Expert_{Complex,Params,ParamsWeird,ReduceParams,Significant,Weird}.txt`
- `MadScientist_{...}.txt`
- `Helper_{x,y,z}.txt` (3 helper prompts)
- `MrMagoo_{...}.txt`
- `mutant{0,1,2,3,4}.txt` (roleplay variants)

### CrossOver (2 templates)
- `templates/CrossOver/crossover.txt` – Standard crossover instruction
- `templates/CrossOver/crossover_s.txt` – Short/simplified crossover

### EoT (1 template)
- `templates/EoT/EoT.txt` – Evolution of Thought comparison template

### Shared
- `templates/ConstantRules.txt` – Appended to all FixedPrompts (shared constraints/instructions)

---

## 9. Logging & Outputs

### Per-Individual Artifacts

**LLM Prompt Log:**
- Path: `${RUN_LOG_DIR}/llm/gene_{gene_id}.log`
- Content: Timestamped entries for each stage:
  - `[PROMPT TO LLM]` – Full assembled prompt
  - `[TEXT FROM LLM (RAW)]` – Raw LLM response
  - `[CODE FROM LLM (VALID)]` – Extracted & validated code
  - `[INVALID LLM OUTPUT (Attempt N)]` – Validation errors (if retries occur)
- Created by: `log_llm_interaction(gene_id, stage, content, is_error)` (llm_utils.py:35–56)

**Generated Child Code:**
- Path: `${SOTA_ROOT}/models/network_{gene_id}.py`
- Format: Python module with nn.Module subclass
- Marker: `${SOTA_ROOT}/models/network_{gene_id}.py.fallback` (if fallback triggered; contains error reason)

**Validation Errors Log:**
- Path: `${RUN_LOG_DIR}/validation_errors.csv`
- Format: CSV with columns (datetime, gene_id, augment_idx, exception_type, error_message)
- Created by: llm_mutation.py:77–81 (on validation failure)

**Evaluation Results:**
- Path: `${SOTA_ROOT}/results/{gene_id}_results.txt` (created by train.py)
- Format: "accuracy,parameter_count" (e.g., "0.9512,45678")

**RAG Logging (if RAG_ENABLED=true):**
- **Code Storage:** Vector DB under `RAG_DATA_DIR` with code + metadata
  - Metadata: gene_id, parent_id, mutation_type, fitness, improvement, description
  - Stored by: `_log_mutation_result()` → `runtime.log_mutation_code()`
- **Metrics:**
  - `rag_generation_context`: retrieval latency, retrieved count
  - `rag_mutation_logged`: fitness, improvement deltas
  - `rag_prompt_enhancement`: retrieval time, mutation count

### Per-Run Outputs

**Checkpoint (every generation):**
- Path: `${RUN_DIR}/checkpoints/checkpoint_gen_{gen}.pkl`
- Contents: GLOBAL_DATA, GLOBAL_DATA_HIST, population, hof, GLOBAL_DATA_ANCESTRY

**Population Summary (console):**
- Via `print_population()` (src/utils/print_utils.py): gene IDs and fitness per generation

**SLURM Logs:**
- Main job: `${RUN_LOG_DIR}/slurm-main-{SLURM_JOB_ID}.out/err`
- Server job: `${RUN_LOG_DIR}/slurm-server-{SERVER_JOB_ID}.out/err`
- LLM jobs: `${RUN_LOG_DIR}/llm-{job_id}.out/err` (or `slurm-{job_id}` if LOCAL=true)
- Eval jobs: `${SLURM_LOG_DIR}/eval-{job_id}.out/err`

**Run Metadata:**
- Path: `${RUN_DIR}/run_metadata.json`
- Created by: run.sh (before & after execution)
- Updated by: run_improved.py (enriches with RAG/evolution config)

**GPU Monitoring (server job):**
- Path: `${RUN_METRICS_DIR}/gpu/server-{SLURM_JOB_ID}.csv`
- Format: CSV of nvidia-smi snapshots (timestamp, index, name, utilization, memory, etc.)

---

## Summary Table: Mutation/Crossover Entrypoints

| Operation | Function | File | Lines | Input | Output |
|-----------|----------|------|-------|-------|--------|
| **Mutation (code gen)** | `augment_network()` | src/llm_mutation.py | 20–111 | parent_code, template | child_code ∥ fallback |
| **Mutation (call)** | `customMutation()` | run_improved.py | 1115–1200 | individual | mutated_individual |
| **Crossover (code gen)** | `augment_network()` | src/llm_crossover.py | 12–67 | parent_x, parent_y | child_code |
| **Crossover (call)** | `customCrossover()` | run_improved.py | 1029–1113 | ind1, ind2 | (offspring1, offspring2) |
| **Template selection** | `generate_template()` | src/evolution/templates.py | 13–55 | prob_eot, gen_count, top_genes | (template_str, mutation_type_str) |
| **RAG augmentation** | `_apply_rag_context()` | run_improved.py | 74–95 | template, mutation_type | (augmented_template, mutations) |
| **LLM code validation** | `generate_augmented_code()` | src/llm_utils.py | 158–225 | prompt, top_p, temperature | valid_code ∥ RuntimeError |
| **Module validation** | `validate_module_source()` | src/llm_utils.py | 125–155 | source_code | None ∥ Exception |
| **Evaluation submit** | `submit_run()` | run_improved.py | 577–637 | gene_id | job_id |
| **Results collect** | `check4results()` | run_improved.py | 734–818 | gene_id | fitness_tuple |

