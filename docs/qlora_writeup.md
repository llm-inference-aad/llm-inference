# QLoRA Fine-Tuning for LLM-Guided Evolution: Implementation & Purpose

---

## Part 1: How Code Mutations Connect to the Original LLMGE Loop

### The Original Evolutionary Loop

The original LLMGE system (`run_improved.py`) is a full multi-objective evolutionary algorithm built on DEAP. It maintains a population of neural network variants and evolves them across generations using two LLM-powered genetic operators:

**Mutation** (`customMutation` → `llm_mutation.py`):
A parent network file is split on `# --OPTION--` markers into independent code blocks. One block is randomly selected, inserted into a prompt template from `templates/FixedPrompts/`, and sent to the LLM. The LLM returns a modified version of that block, which is spliced back into the full network file and saved as a new individual. If the LLM output fails validation, the system falls back to the parent's code (the fallback mechanism in `check4model2run`) so the individual is still evaluable — it just inherits the parent's fitness.

**Crossover** (`customCrossover` → `llm_crossover.py`):
Two parent networks are passed to the LLM together. The LLM is asked to synthesize a new network that combines elements of both. This is a semantic crossover — not a random code splice, but an LLM-reasoned merge.

**End-of-Tree (EoT)**:
A special operator that fires with probability `PROB_EOT` after the first generation. It picks one of the current top-N genes, finds a block that differs from the seed network, and asks the LLM to write a version that synthesizes the best of both — the evolved variant and the original. This allows the system to revisit and refine earlier mutations.

After each generation, `mutate_prompts()` is called, which evolves the prompt templates themselves. The selection pressure is multi-objective NSGA-II / SPEA2, optimizing simultaneously for accuracy and parameter efficiency.

### What the Synthetic Mutation Script Does Differently

`scripts/generate_mutations.py` performs the same fundamental operation as `customMutation` — it picks a block, fills a template, sends it to the LLM, validates the result — but it runs *outside* the evolutionary loop with two key differences:

1. **Exhaustive sampling instead of generational pressure.** The evolutionary loop only calls the mutation operator on selected parents with probability `mutation_probability`. The synthetic script systematically samples across all 7 blocks × 37 templates × 3 temperatures = 777 combinations, then shuffles and pulls from that pool until it has 50 valid mutations. No selection pressure is applied; the goal is diversity, not fitness.

2. **Purpose is data collection, not evolution.** Each accepted mutation is submitted to a SLURM eval job and the (prompt, generated_code, fitness) tuple is stored in `dataset.json`. This data is not fed back into the evolutionary population — it is fed into the QLoRA trainer.

The two systems are deliberately identical at the mutation level. They use the same seed network, the same `# --OPTION--` parsing, the same template files, the same `ConstantRules.txt` appended to each prompt, and the same code validation logic (`compile()` + class/function name check). This is intentional: the training data must match the exact distribution of inputs the LLM will see when operating as the live mutation operator.

---

## Part 2: QLoRA Implementation

### What QLoRA Is

QLoRA (Quantized Low-Rank Adaptation) is a parameter-efficient fine-tuning method. Instead of updating all 70 billion weights of Llama 3.3-70B — which would require ~140GB of GPU memory at bf16 — it freezes the base model in 4-bit NF4 quantization and trains a small set of low-rank adapter matrices injected into the attention and MLP layers. Only those adapter weights are updated during training.

The math: for a weight matrix W, LoRA introduces two small matrices A (d × r) and B (r × d) where r << d. The update to the forward pass is W·x + (B·A)·x · (α/r), where α/r is a fixed scaling factor. Gradients only flow through A and B.

### Configuration

| Parameter | Value | Why |
|---|---|---|
| Base model | Llama 3.3-70B-Instruct | Strongest model available locally on PACE-ICE |
| Quantization | 4-bit NF4 + double quantization | Fits the full 70B model on a single H100 80GB |
| Compute dtype | bfloat16 | H100's native dtype; numerically stable for long sequences |
| LoRA rank (r) | 16 | Sufficient expressivity for code-domain adaptation at this data scale |
| LoRA alpha | 32 | α/r = 2; standard effective learning rate scaling for LoRA |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention projections + full MLP — covers every transformer sublayer |
| Trainable parameters | 207,093,760 out of 70,760,800,256 | **0.293% of total weights** |
| Optimizer | paged_adamw_32bit | Offloads optimizer states to CPU RAM; required to fit training on one GPU |
| Gradient accumulation | 8 steps | Simulates effective batch size of 8 with single-sample batches |
| Gradient checkpointing | On | Trades compute for memory; needed for 2048-token EMADE prompts |
| Max sequence length | 2048 | Covers the full prompt + generated code block |
| Learning rate | 2×10⁻⁴ with cosine decay + 5% warmup | Standard QLoRA recommendation |
| Epochs | 3 | |
| Hardware | NVIDIA H100 80GB HBM3 | PACE-ICE `ice-gpu` partition |

### Training Data Format

Each training sample is a (prompt, completion) pair where:

- **Prompt**: a filled template from `FixedPrompts/` — the same text that gets sent to the LLM during a live LLMGE mutation. It contains a roleplay or Q&A instruction followed by the actual code block from ExquisiteNetV2.
- **Completion**: the generated code that the LLM produced for that prompt, but only for samples where the evaluated network exceeded the `min_accuracy` fitness threshold.

This means the model is not learning to chat, summarize, or answer general questions. It is learning the specific conditional distribution: *given this mutation prompt and this code block, produce code that looks like a high-fitness mutation of ExquisiteNetV2.*

### Training Pipeline Steps

**Step 1 — Dataset preparation (`join_metrics.py`)**  
Collects completed SLURM eval results, parses the fitness tuple `(top-1 accuracy, param count, top-5 accuracy, latency ms)`, and writes `dataset.json`. Samples below `min_accuracy` are filtered out.

**Step 2 — Tokenization**  
The tokenizer is Llama 3.3-70B's SentencePiece tokenizer. Each (prompt, completion) pair is formatted using the Llama 3 chat template and truncated/padded to 2048 tokens. The loss is masked on the prompt tokens — only the completion tokens contribute to the gradient.

**Step 3 — 4-bit model loading**  
The base model is loaded with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True)`. `prepare_model_for_kbit_training()` freezes the base weights and enables gradient flow through the adapter.

**Step 4 — LoRA injection**  
`get_peft_model()` wraps the quantized model and injects the A/B adapter matrices into all 7 target module types across all 80 transformer layers.

**Step 5 — Training with SFTTrainer**  
HuggingFace TRL's `SFTTrainer` runs the training loop. A custom `TimeBudgetCallback` monitors wall-clock time and triggers a clean checkpoint save if the job is within 30 minutes of the SLURM 16-hour hard limit. This enables training to resume across consecutive SLURM job submissions without losing gradient state.

**Step 6 — Adapter save**  
The trained LoRA weights are saved via `save_pretrained()` to `runs/qlora_v1/qlora/adapters/`. The base model is not modified.

### First Training Run Results (qlora_v1)

Trained April 28, 2026 on PACE-ICE (job `5100618`), H100 80GB.

| Metric | Value |
|---|---|
| Training samples | 11 |
| Gradient steps completed | 6 |
| Training time | 93.9 seconds |
| Final training loss | **1.0486** |
| Mean token accuracy | **76.77%** |
| Gradient norm | 0.206 |
| Adapter file size | 791 MB |

The checkpoint-resume mechanism was validated: a second job (`5100947`) correctly picked up from `checkpoint-6` with no loss discontinuity.

The training script itself emits a warning at this data volume: *"QLoRA typically needs 200+ for meaningful generalization."* 11 samples and 6 steps is a proof-of-concept run. The adapter has learned something from the 11 examples (token accuracy of 76.77% on the training set indicates it is not at random), but it has not seen enough variation to generalize across the full template × block space. The active synthetic mutation run is collecting the data needed for a second, substantive training round.

---

## Part 3: How the Adapter Is Meant to Improve LLMGE

### Serving the Adapter

At server startup, if `ADAPTER_PATH` is set in `.env`, the server loads the adapter with `PeftModel.from_pretrained()` and immediately calls `merge_and_unload()`. This fuses the LoRA matrices back into the base weight matrices algebraically: the effective weight becomes W + B·A·(α/r). The adapter is dissolved into the base model. From this point forward the model runs at full inference speed with no adapter overhead — identical latency to the base model, but with updated weights.

### The Feedback Loop

The intended mechanism is a closed self-improvement cycle:

```
Evolutionary loop generates mutations
        ↓
Eval jobs score each mutation on CIFAR-10
        ↓
High-fitness mutations collected into dataset.json
        ↓
QLoRA fine-tunes Llama 3.3-70B on those (prompt, good_mutation) pairs
        ↓
Adapted model serves as the mutation operator in the next evolution run
        ↓
Repeat
```

In the original loop, the LLM generates mutations from a cold prior — it has no knowledge of which types of changes tend to improve ExquisiteNetV2 on CIFAR-10. It might suggest changes that are syntactically valid but have no bearing on accuracy (e.g., renaming variables, changing unrelated hyperparameters). The fallback rate in `check4model2run` captures how often LLM output is entirely unusable; the fraction of evaluated mutations with fitness near the seed baseline captures how often the change was valid but unhelpful.

After fine-tuning on a quality-filtered dataset, the model has seen examples of what a productive mutation looks like for each of the 7 code blocks. The expected improvements are:

- **Lower fallback rate**: Fewer generated code blocks fail `compile()` or omit the required class/function name, because the model has seen many examples of correctly structured mutations.
- **Higher fraction of above-baseline mutations**: The model has been shown that, for this network on this task, certain types of changes (e.g., modifications to the SE attention ratio, optimizer parameter groups, activation functions) tend to produce evaluable and competitive networks.
- **Better use of templates**: Because fine-tuning exposed the model to all 37 templates across all 7 blocks, it has practice completing the specific instruction style of each template for each type of code block, rather than treating each prompt as novel.

The key experiment — comparing the fraction of mutations that exceed the mean baseline accuracy (0.384) when using the base model versus the fine-tuned adapter as the mutation operator — is the planned validation once the current 53 eval jobs complete and a second QLoRA round (`qlora_v2`) is trained on the expanded dataset.
