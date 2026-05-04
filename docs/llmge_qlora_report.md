# LLM-Guided Evolution (LLMGE): Infrastructure & QLoRA Fine-Tuning Report

**Author:** Rohan Kansal  
**Date:** May 2026  
**Cluster:** PACE-ICE, Georgia Tech  
**Base Model:** Meta Llama 3.3-70B-Instruct

---

## 1. System Overview

LLMGE is a pipeline that uses a large language model as the mutation operator inside an evolutionary neural architecture search loop. Instead of hand-crafted mutation rules, a locally-served LLM reads a code block from a seed network and proposes a modified version. Those mutations are compiled, trained on CIFAR-10, and evaluated for accuracy, parameter count, and inference latency. The results feed back into a dataset that is used to fine-tune the LLM itself — closing the loop so the model progressively improves as a mutation operator.

The system runs entirely on the PACE-ICE SLURM cluster with no external API dependency.

---

## 2. Infrastructure Built

### 2.1 Local LLM Inference Server

A FastAPI server (`server.py`) wraps HuggingFace Transformers and exposes a `/generate` endpoint. Key design decisions:

- **Batching**: Configurable batch size and wait time for request coalescing
- **LoRA hot-loading**: Server accepts `ADAPTER_PATH` at startup, merges the adapter into the base weights with `merge_and_unload()` so inference runs at full speed with no adapter overhead
- **Run-scoped metrics**: Every request is logged to a per-session JSON file under `runs/<run_id>/metrics/`, keyed by a run hash, enabling latency analysis across evolution rounds
- **Multi-server registry**: A `servers.json` registry with file locking (`flock`) allows multiple server instances to register themselves, supporting a load balancer layer for parallel evolution runs
- **Thinking token stripping**: Post-processes model output to remove `<think>...</think>` blocks (for R1-style models), returning only the code

### 2.2 Evolution Mutation Pipeline

`scripts/generate_mutations.py` drives the outer loop:

- Parses the seed network (`sota/ExquisiteNetV2/network.py`) on `# --OPTION--` markers, extracting **7 independently mutable code blocks**
- Samples from **37 prompt templates** across 2 styles (concise Q&A, roleplay persona) and **3 temperatures** (0.3, 0.7, 1.0) — 777 total unique (block, template, temperature) combinations per round
- Validates LLM output: must parse as Python, must compile (`compile()` with `exec` mode), and must preserve the original class/function name
- Submits a SLURM eval job per accepted mutation, which trains the mutated network for a short schedule on CIFAR-10 and records top-1 accuracy, parameter count, top-5 accuracy, and inference latency
- Writes a `synthetic_manifest.json` per run so results can be joined without relying on SLURM log filenames

**Mutation validation gates** (`validate_code()`): LLM output must pass three sequential checks before it enters the pipeline. First, the extracted code must be at least 20 characters (rejects empty or trivially short responses). Second, it must compile cleanly via `compile(code, "<string>", "exec")` (rejects syntactically invalid Python). Third, the original class or function name must still be present in the mutated code (rejects outputs where the LLM renamed the entrypoint, which would break EMADE's class-lookup). Only mutations passing all three gates get a model file written and a SLURM eval job submitted.

**Combo-key deduplication**: each (block index, template filename, temperature) triple is recorded as a `combo_key` in `synthetic_manifest.json`. Subsequent generation runs skip any combo already present, ensuring the dataset does not contain duplicate (prompt, temperature) pairings regardless of how many times the generator is run against the same template pool.

**Code assembly**: the mutated block is a surgical replacement — only `seed_parts[block_to_part_idx[block_idx]]` is swapped; the other six `# --OPTION--` blocks remain from the seed network verbatim. This ensures the assembled network file is always a structurally complete, importable Python module even when the mutated block itself is minimal.

**Current run stats (run `my_run_20260428_011918`):**

| Metric | Value |
|---|---|
| Mutations generated & submitted | 53 |
| Unique templates used | 27 / 37 |
| Code blocks covered | All 7 |
| Temperature distribution | 0.3: 9 (17%), 0.7: 18 (34%), 1.0: 26 (49%) |

### 2.3 Evaluation & Dataset

Each eval job trains the mutated network and writes a fitness tuple:

```
(top-1 accuracy, parameter count, top-5 accuracy, inference latency ms)
```

`join_metrics.py` collects completed eval results and writes `dataset.json` for QLoRA training.

**Current dataset (16 evaluated entries):**

| Metric | Value |
|---|---|
| Total entries | 16 |
| Top-1 accuracy range | 0.287 – 0.533 |
| Top-1 accuracy mean | 0.384 |
| Parameter range | 516,766 – 687,910 |
| Inference latency range | 32.5 – 119.6 ms |
| Best mutation (acc) | `xXxfCXJnehcr`, **53.3% top-1**, 518K params, 87ms |

---

## 3. QLoRA Fine-Tuning

### 3.1 Standard QLoRA Setup

The fine-tuning pipeline (`scripts/train_qlora.py`) follows the QLoRA recipe from Dettmers et al. (2023):

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base model | Llama 3.3-70B-Instruct | Strongest available locally |
| Quantization | 4-bit NF4 + double quant | Fits 70B on a single H100 80GB |
| Compute dtype | bfloat16 | H100 native dtype, stable for long sequences |
| LoRA rank (r) | 16 | Balances expressivity vs. parameter efficiency |
| LoRA alpha | 32 | α/r = 2, standard effective scaling |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + MLP projections — full transformer coverage |
| Trainable parameters | **207,093,760** (0.293% of 70.76B) | |
| Learning rate | 2e-4 | Standard QLoRA LR |
| LR schedule | Cosine with 5% warmup | |
| Optimizer | paged_adamw_32bit | Memory-efficient AdamW for QLoRA |
| Gradient accumulation | 8 steps | Effective batch = 8 with single GPU |
| Gradient checkpointing | Enabled | Required for 2048-token EMADE prompts on 1 GPU |
| Max sequence length | 2048 | Covers full prompt + generated code |
| Epochs | 3 | |
| `min_accuracy` threshold | 0.0 (current) → 0.45 (target) | Fitness gate: only samples whose top-1 CIFAR-10 accuracy meets the floor enter training |
| `packing` | False | SFTTrainer default packs sequences to fill context; disabled here to avoid cross-example contamination |
| Hardware | NVIDIA H100 80GB HBM3 | PACE-ICE cluster |

**Adapter file size:** 791 MB (stored as `adapter_model.safetensors`)

### 3.2 Training Run (qlora_v1)

Training was conducted on April 28, 2026 (job `5100618`), completing in **~94 seconds** over 6 gradient steps on 11 samples.

| Metric | Value |
|---|---|
| Training samples used | 11 |
| Total gradient steps | 6 |
| Final training loss | **1.0486** |
| Mean token accuracy (final) | **76.77%** |
| Gradient norm | 0.206 (healthy, no explosion) |
| Entropy (final) | 1.074 |
| Train runtime | 93.9 seconds |

The checkpoint-resume system was validated across two consecutive SLURM jobs (`5100618` → `5100947`), confirming correct state restoration with no loss spike.

> **Honest caveat:** 11 samples and 6 gradient steps is insufficient for meaningful generalization. The training script itself emits a warning: *"QLoRA typically needs 200+ for meaningful generalization."* The current adapter captures the signal present in the 11 samples but should be considered a proof-of-concept run. The mutation pipeline is actively generating more data (53 mutations submitted, evaluations pending).

---

## 4. Innovations Over Standard QLoRA

The following adaptations are specific to using QLoRA as an evolutionary operator rather than a general instruction-following fine-tune:

### 4.1 Fitness-Gated Dataset Curation

Standard QLoRA SFT trains on all available (prompt, response) pairs. Here, training samples are filtered by a `min_accuracy` threshold on the fitness column before being passed to the trainer. Only mutations that produced a network meeting the accuracy floor are included.

**Fitness tuple format**: each `dataset.json` entry carries a `fitness` string of the form `"top1_acc,param_count,top5_acc,latency_ms"` (e.g., `"0.533,518000,0.991,87.3"`). The dataset loader (`load_and_filter_dataset`) splits on comma and reads index 0 as the quality signal. Only column 0 (top-1 accuracy) is used for filtering; the other columns are preserved in the record but not used during training.

**Threshold values and their meaning**:

| Phase | `min_accuracy` | Rationale |
|---|---|---|
| v1 (current, 11 samples) | **0.0** | Too few samples to filter any — dropping entries would leave too little signal |
| v2 target (200+ samples) | **≥ 0.45** | Filters out mutations that perform below the dataset mean (0.384), training only on above-average outcomes |

The training script enforces two additional data-volume guards: if fewer than 50 samples pass the filter it emits a `WARNING` ("QLoRA typically needs 200+ for meaningful generalization") and continues; if zero samples pass, it raises `ValueError` and exits rather than training on nothing. The 50-sample number is a soft floor and 200 is the recommended minimum for the kind of code-domain SFT being done here.

**Training text format**: each training example is assembled as:

```
{prompt.strip()}\n{generated_text.strip()}
```

There is no chat-template wrapper (no `[INST]`/`[/INST]` or `<|user|>`/`<|assistant|>` tokens). The model is being trained to continue a specific structured text — system rules followed by a code block — not to engage in dialogue. Using the raw concatenation keeps the training distribution consistent with the inference-time prompt format used by `generate_mutations.py`.

**Why this matters:** The LLM should learn from its own *successful* mutations, not from all outputs including bad ones that produced 28.7% accuracy. As the dataset grows and the threshold is raised (current: 0.0 → target: ≥0.45), the model will be progressively trained on a higher-quality curriculum.

### 4.2 Self-Referential Training Loop

The training data is generated by the same model being fine-tuned. This is a form of online policy improvement: the model proposes mutations → the good ones become training examples → the model is updated → it proposes better mutations. This is architecturally similar to RLHF's self-improvement loop, but uses fitness-evaluated code correctness as the reward signal instead of human preference.

### 4.3 Checkpoint Resume Across SLURM Job Slots

PACE-ICE imposes a 16-hour hard wall-time limit. The training script implements a **15.5-hour soft budget**: it writes a clean checkpoint before the wall clock expires, and the next job submission auto-resumes from the latest checkpoint. This allows training to span multiple cluster job slots without manual intervention and without losing gradient state.

### 4.4 Task-Specific Sequence Structure

The training sequences are structured as:

```
[System: roleplay/Q&A prompt] → [Code block from seed network] → [Mutated code output]
```

This is meaningfully different from standard instruction-following fine-tuning. The model is not learning to chat or answer questions — it is learning the specific distribution of *valid PyTorch mutations of ExquisiteNetV2 code blocks*. The 7-block structure means the model sees all architectural components (optimizer, SE attention, conv layers, etc.) as training targets.

### 4.5 Sequence Packing Disabled

`SFTTrainer` packs multiple training examples end-to-end into a single context window by default, which maximises GPU utilization. That default is explicitly disabled here (`packing=False`). The reason is that EMADE mutation prompts are long (system rules + full code block + constant rules ≈ 1,000–1,800 tokens) and variable in length. Packing would concatenate two or more prompts into one sequence, causing the model to attend across example boundaries and potentially treat the end of one mutation as context for the next. Since the goal is to train the model on the specific (prompt → code) mapping, cross-example attention would corrupt the learned association. At the current data volume (11–60 examples) each sequence also fits comfortably within the 2048-token limit, so packing provides no throughput benefit that would justify the risk.

### 4.6 Temperature-Diverse Training Data

Mutations were generated at 3 temperatures (0.3, 0.7, 1.0). Higher temperatures produced more novel (if more error-prone) code. Including all three in the training set ensures the model learns from both conservative modifications (lr tweaks, parameter changes) and more creative structural changes (new helper functions, architectural rewrites). This covers the full diversity spectrum the prompt templates were designed to elicit.

---

## 5. Template Quality Improvements

During analysis of the mutation pipeline, 10 of the 37 templates (`mutant0-4.txt` in both `concise/` and `roleplay/`) were identified as producing near-zero valid output. The root causes were:

1. **No instruction**: Templates showed example Python code followed directly by `{}` with no question or directive. The LLM had no task to complete.
2. **Model artifacts**: Templates contained `</think>` tokens (artifacts of a DeepSeek-R1–style model) that Llama does not interpret as continuation signals.
3. **Broken example code**: One template referenced `self.out_relu` which does not exist on the model — a copy-paste error that caused confusion.

All 10 were rewritten to include a clear instruction, with the example code properly framed as a few-shot demonstration. Additionally, a separate bug was fixed in `generate_mutations.py`: `tmpl_text.format(block)` was replaced with `tmpl_text.replace("{}", block, 1)`. The original `.format()` call caused `KeyError` whenever a template contained literal Python curly braces (e.g., `{result_c}`, `{'id': item['id']}`), crashing generation mid-run.

**Expected impact:** The 10 fixed templates represent 27% of the template pool. Prior to the fix, these contributed ~0 valid mutations despite consuming server time. After the fix, the effective template coverage increases from ~27 working templates to all 37.

---

## 6. Current State & Expected Next Steps

### What is complete
- Full inference server with LoRA adapter support, metrics logging, and multi-server load balancing
- Mutation generation pipeline with manifest tracking and SLURM eval submission
- QLoRA fine-tuning pipeline with fitness filtering, checkpoint resumption, and self-referential loop
- First adapter trained and ready to serve (`runs/qlora_v1/qlora/adapters/`)
- 16-entry dataset with real fitness evaluations; 53 additional mutations submitted and evaluating

### What is in progress
- Eval jobs for the 53 new mutations are running; once `join_metrics.py` completes, the dataset will grow to ~60+ entries
- Template fixes take effect on the next generation run, expected to meaningfully reduce the skip rate

### Expected trajectory (projected, not yet measured)

With dataset growth to 200+ quality samples (filtering at acc ≥ 0.45), a second QLoRA run (`qlora_v2`) would train for the full 3 epochs with ~180 gradient steps rather than 6. At that scale:

- Token accuracy is expected to reach **>85%** based on typical SFT convergence curves at this data volume for code-domain fine-tuning
- Training loss expected to fall below **0.6** (from current 1.05), indicating the model is learning the specific structural patterns of valid EMADE mutations
- The fine-tuned model serving as the mutation operator is expected to produce fewer syntactically invalid responses and more mutations that preserve the seed network's design patterns (correct class names, compatible tensor shapes), reducing the skip rate and increasing the fraction of submitted mutations that score above the fitness floor

The key validation metric will be the **fraction of LLM-generated mutations that exceed the mean baseline accuracy (0.384)** when served with vs. without the adapter. That comparison is the planned next experiment once the 53 pending evals complete.

---

## 7. Summary

| Component | Status | Key Number |
|---|---|---|
| Inference server | Running | H100 80GB, Llama 3.3-70B |
| Mutation pipeline | Active | 53 mutations in flight |
| Template quality | Fixed | 10/37 templates repaired |
| Dataset | Growing | 16 evaluated, 53 pending |
| QLoRA adapter (v1) | Trained, ready to serve | 791 MB, 207M trainable params (0.29%) |
| Training loss | Measured | 1.0486, token acc 76.8% |
| Best mutation found | Measured | **53.3% top-1 on CIFAR-10** |
| Self-improvement loop | Designed, first cycle done | v2 training pending data |
