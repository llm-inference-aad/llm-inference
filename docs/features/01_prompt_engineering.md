# Prompt Engineering: Inspiring Creative Architectural Exploration

## Overview

This document describes the **revised** prompt engineering philosophy that enables the LLM to explore novel CNN architectures through **creative reasoning** rather than prescriptive compliance. The goal is to discover unconventional, high-performing solutions that a human designer might not consider.

## Philosophy Shift

### ❌ Before: Prescriptive Compliance Engine
**Over-specification killed creativity:**
- Explicit checklists ("estimate parameter delta", "explain trade-offs")
- Pre-specified architectural lever lists telegraphing "expected" solutions
- Mandatory reasoning scaffolds (Context → Constraints → Levers → Desired Change → Trade-off Analysis)
- Directive language ("must justify", "reference how each lever influences", "explain why")

**Result:** LLM became a form-filler executing instructions, not an architecture explorer discovering novelty.

### ✅ After: Guided Creative Exploration
**Nudge, don't direct:**
- **Awareness over prescription** - Mention objectives, don't mandate how to achieve them
- **Inspire over instruct** - Suggest possibilities, don't enumerate checklists  
- **Trust LLM creativity** - Remove scaffolding requirements, let reasoning emerge naturally
- **Preserve guardrails** - Keep technical constraints (tensor shapes, compilation) but drop "explain your reasoning" demands

## Core Principles

### 1. Research Goal Alignment
This is a **research project exploring novel architectures**. The LLM's imaginative reasoning is the primary discovery mechanism. Over-constraining prompts defeats the purpose, but under-constraining them prevents effective context retrieval. 
- We want **unexpected** solutions, but within the bounds of verifiable code.
- Creative mutations drive evolutionary diversity.
- Aligning the LLM's imagination with explicit PyTorch vocabulary (`torch.nn.SiLU`, `AdamW`) bridges the semantic gap for our RAG database.

### 2. Gentle Guidance + RAG-Bounded Action Subspaces
Prompts should:
- State the multi-objective goal (accuracy ≥90%, efficient parameters) as **awareness context**, not compliance mandates.
- Preserve technical constraints (tensor shapes must remain `(batch, 3, 32, 32)` → `(batch, 10)`, code must compile).
- Encourage creative exploration by **suggesting specific, high-value PyTorch operators** (e.g., `Mish`, `OneCycleLR`, `AdaptiveAvgPool2d`). This bounds the action space to valid API calls and crucially **enables FAISS vector similarity search** to retrieve the correct PyTorch documentation sheets from our ingested corpus.

### 3. Persona-Driven Diversity
Different personas change **tone and creative direction**, not objectives:
- **Expert**: Sophisticated, research-backed innovations.
- **MadScientist**: Energetic, bold, chaotic experimentation.  
- **MrMagoo**: Whimsical, serendipitous, understated discoveries.

## Changes Made

### 1. Simplified ConstantRules.txt (Preserved)

**Before (Rules 9-12):** 4 prescriptive rules with explicit scaffolds, lever lists, and trade-off mandates

**After (Rules 9-10):**
```
9. This network targets high CIFAR-10 accuracy (≥90%) with efficient parameter usage. 
   Consider how architectural choices influence both objectives.

10. Explore creative modifications that push performance boundaries—novel layer 
    compositions, unconventional activation patterns, or emergent architectural motifs 
    are encouraged.
```

### 2. Transformed Concise Templates: Bridging the RAG Semantic Gap

Initially, templates were reduced from 28 lines of prescriptive text to 3 lines of pure inspiration. However, pure inspiration lacked the syntactic keywords necessary for the FAISS index to retrieve the correct PyTorch API docs for RAG-Text augmentation. We successfully addressed this by injecting specific keywords into the creative framing.

**Example: Complex.txt**

**Initial State (Prescriptive - 28 lines):**
```text
Desired Change:
- Propose complex architectural enhancements (e.g., channel grouping strategies, 
  attention placement, activation upgrades) that deliver higher accuracy while 
  staying within a justified parameter budget. Reference how each lever influences 
  both objectives.
```

**Second Iteration (Pure Creativity - 3 lines):**
```text
Q: How would you reimagine this CIFAR-10 architecture to achieve breakthrough 
performance? Consider creative combinations of layer types, attention mechanisms, 
activation functions, or novel architectural patterns. The network seeks high 
accuracy (≥90%). Input/output tensors must remain `(batch, 3, 32, 32)` → `(batch, 10)`.
```

**Current State (Bounded Creativity for FAISS RAG - March 2026):**
```text
Q: How would you reimagine this CIFAR-10 architecture to achieve breakthrough performance? Explore structural and functional combinations: replace ReLU with advanced activations like `torch.nn.SiLU` or `Mish`, integrate Squeeze-and-Excitation (`SE` blocks) mechanisms or `AdaptiveAvgPool2d`, or leverage depthwise (`groups`) and dilated convolutions. The network seeks high accuracy (≥90%) with thoughtful parameter management. Input/output tensors must remain `(batch, 3, 32, 32)` → `(batch, 10)`.
```

**Key Improvements:**
- Maintains the short 3-line format.
- Removes prescriptive checklists and compliance scoring constraints.
- Crucially embeds valid PyTorch vocabulary (`torch.nn.SiLU`, `Mish`, `AdaptiveAvgPool2d`, `groups`). When these templates are embedded and queried against the FAISS RAG corpus, the vector search retrieves the precise mathematical formulation and hyperparameter docs for those explicitly named modules, reducing LLM syntax hallucinations.

### 3. Updated RAG Context Integration

**File**: `src/llm_utils.py` - `_prepend_rag_context_to_prompt()`

**Before:**
```python
rag_prefix = (
    "Reference the following mutations that improved the accuracy/parameter Pareto frontier. "
    "Compare their recorded accuracy_delta and parameter_count to prioritize edits that "
    "deliver the highest accuracy gain per parameter added (or that preserve accuracy while "
    "reducing parameters). Preserve compilation requirements, tensor shapes, and the current "
    "parameter budget unless a clear trade-off justification is provided.\n"
    f"{context_block}\n\n"
)
```

**After:**
```python
rag_prefix = (
    "Here are some successful mutations from prior generations. "
    "Consider how their approaches might inspire your own creative solution, but feel "
    "free to explore novel directions.\n"
    f"{context_block}\n\n"
)
```

**Key Changes:**
- "Reference... Compare... prioritize edits... deliver highest gain" → "Consider how their approaches might inspire"
- "Preserve... unless clear justification provided" → removed compliance language
- RAG context becomes **inspiration**, not **directive examples to follow**

### 5. Architectural Lever Reference (Maintained for Human Use)

**File**: `templates/ARCHITECTURAL_LEVERS.md`

**Status**: Kept as a **reference document for humans** (researchers, template writers), but **removed from prompts**

**Rationale**: 
- Useful for understanding the codebase
- Helpful for template writers to maintain consistency
- **NOT** shown to the LLM (prevents constraining its creative exploration)

## Example Transformations

### Before: Prescriptive Roleplay Template
```
As a leading authority, apply expert-level reasoning to evolve this CIFAR-10 architecture 
under a multi-objective mandate (≥90% accuracy while minimizing parameters).

Context:
- Operate within the existing ExquisiteNet-style modules; mutate `# --OPTION--` blocks responsibly.
- Objective 1: Increase test accuracy via well-founded architectural upgrades.
- Objective 2: Control parameter growth to keep inference efficient.

Constraints:
- Preserve input `(batch, 3, 32, 32)` and output `(batch, 10)` tensor sizes.
- Code must compile, train, and adhere to ConstantRules.

Architectural Levers for Experts:
- Redistribute channel widths/expansion ratios and insert bottleneck helpers.
- Layer SE / SE_LN attention strategically to boost feature quality with minimal params.
- Upgrade activation pipelines (SiLU, Mish, Hardswish, GELU) and mixed pooling.
- Calibrate optimizer/normalization settings that complement structural changes.

Desired Change:
- Design a sophisticated sequence of modifications (e.g., staged attention, cross-stage 
  feature fusion, activation factories) that meaningfully shifts the accuracy/parameter 
  Pareto front. Justify each step in terms of gains per parameter added.

Trade-off Analysis Required:
- Estimate parameter deltas and highlight mitigation tactics.
- Explain the expected accuracy lift, referencing receptive fields, gradient flow, or 
  regularization improvements.
- Outline guardrails: residual integrity, initialization considerations, validation checks.
```

### After: Inspiring Roleplay Template
```
As a leading ML authority, how would you evolve this CIFAR-10 architecture to breakthrough 
performance? Draw on your expertise to propose creative architectural innovations that 
balance accuracy (≥90%) with parameter efficiency. Input/output: `(batch, 3, 32, 32)` → 
`(batch, 10)`.
```

**Transformation:**
- **30 lines → 3 lines** (90% reduction)
- Removed: Context section, Constraints list, Lever enumeration, Desired Change specifics, Trade-off Analysis checklist
- Kept: Persona ("leading ML authority"), dual objective awareness ("balance accuracy with parameter efficiency"), technical constraint (tensor shapes)
- Added: Creative framing ("how would you evolve", "propose creative innovations")

## Success Criteria (Revised)

✅ **Every template mentions multi-objective awareness without prescription**
- Changed from "Optimize for dual objective" to "seeks high accuracy with thoughtful parameter management"
- Changed from "must justify parameter increases" to "consider how choices influence both objectives"

✅ **Architectural levers removed from prompts**
- Lever lists removed from all templates (moved to reference doc for humans only)
- LLM discovers architectural components organically rather than choosing from a menu

✅ **Reasoning scaffolds eliminated**
- No more "Context → Constraints → Levers → Desired Change → Trade-off Analysis" structure
- LLM generates reasoning naturally as part of creative process

✅ **Trade-off mandates replaced with awareness**
- Removed "Estimate parameter delta", "Explain accuracy gain", "Highlight guardrails"
- Kept awareness of dual objective as contextual nudge, not compliance requirement

✅ **RAG context inspires rather than directs**
- Changed from "Reference... Compare... prioritize" to "Consider how approaches might inspire"
- Examples become inspiration, not templates to follow

## Metrics & Expected Impact

### Quantitative Improvements Expected:
1. **Higher architectural diversity** - Measured via unique network topologies per generation
2. **More creative mutations** - Fewer "standard" SE block additions, more novel compositions
3. **Faster Pareto frontier expansion** - Creative exploration finds unconventional high-performers faster
4. **Reduced prompt compliance overhead** - LLM spends less reasoning on "checking boxes", more on architecture

### Qualitative Improvements Expected:
1. **Unexpected solutions** - Architectures that wouldn't emerge from prescribed lever lists
2. **Emergent patterns** - Discovery of novel architectural motifs through experimentation
3. **Better persona differentiation** - Expert/MadScientist/MrMagoo produce genuinely different approaches
4. **Natural reasoning** - LLM explanations flow organically rather than filling required sections

### Track via:
- `metrics/rag_metrics.jsonl`: Mutation success rates, diversity metrics
- Fitness logs: Distribution of accuracy/parameter combinations
- Template usage analysis: Success rates by persona and mutation type
- Manual inspection: Review generated architectures for novelty and creativity

## Comparison: Prescriptive vs. Creative vs. Bounded (Current)

| Aspect | Prescriptive (Before) | Creative (Mid) | Bounded for RAG (Current) |
|--------|----------------------|----------------|--------------------------|
| **Prompt Length** | 25-30 lines | 2-4 lines | 3-6 lines |
| **Structure** | Rigid 5-section scaffold | Open-ended question | Open-ended question with semantic keywords |
| **Lever Guidance** | Explicit enumeration | None (organic discovery) | Suggestive API hooks (`SiLU`, `AdaptiveAvgPool2d`) |
| **FAISS Retrieval**| N/A | Fails due to vague keywords | Succeeds, retrieves relevant PyTorch documentation |
| **LLM Role** | Compliance executor | Creative explorer | Informed Creative Explorer |
| **Diversity** | Constrained by lists | Unbounded exploration | Bounded by valid PyTorch functionality |

## Implementation Summary

### Files Modified:
1. **`templates/ConstantRules.txt`**: Simplified rules 9-12.
2. **`templates/FixedPrompts/concise/*.txt`**: Reduced layout but injected exact PyTorch operator strings.
3. **`src/llm_utils.py`**: Simplified RAG context prefix from directive to inspirational.
4. **`src/rag/data_ingestion.py`**: Implemented `hashlib.sha256(content.encode('utf-8')).hexdigest()` to deduplicate identical code blocks across mutated runs and prevent the FAISS database from being flooded with convergent evolution duplicate solutions.

### Verification Steps:
```bash
# Check ConstantRules simplified
cat templates/ConstantRules.txt | grep -A2 "^9\."

# Verify RAG keyword inclusion
cat templates/FixedPrompts/concise/Complex.txt

# Check RAG context updated
grep -A5 "rag_prefix =" src/llm_utils.py

# Verify FAISS deduplication
grep -A3 hashlib src/rag/data_ingestion.py
```

## Future Considerations

### What We Kept (Deliberately):
1. **Technical constraints** - Tensor shapes, compilation requirements.
2. **Dual objective awareness** - Mention of accuracy target and parameter efficiency.
3. **Keyword Binding** - Explicitly providing correct PyTorch operators to bridge the semantic gap for RAG retrieval.

### What We Might Adjust:
1. **Embedding Models** - We have upgraded to `gemini-embedding-2` to further improve semantic retrieval over open-source HuggingFace standard sentence transformers.
2. **RAG Database Size** - Continuously monitoring `rag_data/metadata/code.jsonl` to see how the code namespace populates.
3. **Persona Intensity** - Ensuring Expert/MadScientist roles don't drift away from utilizing the retrieved FAISS documentation.

### Monitoring Plan:
1. **FAISS Ingestion Logs**: Confirm that identical code mutations are rejected by the sha256 hashing.
2. **Evaluate Validation Logs**: Track `logs/errors/validation_errors.csv` to ensure the bounded prompts actually reduce syntax error rates compared to completely blank open-ended prompts.
3. **Monitor SOTA**: Verify if RAG+Bounded Prompts actually find Pareto expansions faster than Open-Ended prompts.

## Related Documentation

- `templates/ARCHITECTURAL_LEVERS.md`: Lever reference (for humans, not LLM prompts)
- `docs/07_rag_feature.md`: RAG pipeline integration details
- `src/llm_utils.py`: RAG context integration and prompt mutation logic
- `run_improved.py`: Template generation and mutation logging

## Professor's Guidance

> "The prompts should not be specific as if we are telling the LLM what to do. Rather the prompt gently guides the query letting the LLM's imaginative reasoning handle the mutation. Remember the goal of this research project - to explore novel architectures. This can only be done with a creative element."

**This revision implements that guidance**: Prompts now **inspire** rather than **instruct**, trusting the LLM's creativity to discover unexpected architectural innovations.

