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
This is a **research project exploring novel architectures**. The LLM's imaginative reasoning is the primary discovery mechanism. Over-constraining prompts defeats the purpose:
- We want **unexpected** solutions, not prescribed ones
- Creative mutations drive evolutionary diversity
- Novel architectures emerge from open-ended exploration

### 2. Gentle Guidance
Prompts should:
- State the multi-objective goal (accuracy ≥90%, efficient parameters) as **awareness context**, not compliance mandates
- Preserve technical constraints (tensor shapes must remain `(batch, 3, 32, 32)` → `(batch, 10)`, code must compile)
- Encourage creative exploration without prescribing specific approaches
- Trust the LLM to find unexpected paths to the Pareto frontier

### 3. Persona-Driven Diversity
Different personas change **tone and creative direction**, not objectives:
- **Expert**: Sophisticated, research-backed innovations
- **MadScientist**: Energetic, bold, chaotic experimentation  
- **MrMagoo**: Whimsical, serendipitous, understated discoveries

All personas share awareness of the dual objective but approach it from completely different creative angles.

## Changes Made

### 1. Simplified ConstantRules.txt

**Before (Rules 9-12):** 4 prescriptive rules with explicit scaffolds, lever lists, and trade-off mandates

**After (Rules 9-10):**
```
9. This network targets high CIFAR-10 accuracy (≥90%) with efficient parameter usage. 
   Consider how architectural choices influence both objectives.

10. Explore creative modifications that push performance boundaries—novel layer 
    compositions, unconventional activation patterns, or emergent architectural motifs 
    are encouraged.
```

**Key Changes:**
- "Optimize for the dual objective... changes must justify" → "Consider how choices influence both"
- Removed mandatory reasoning scaffold
- Removed explicit architectural lever enumeration
- "Prefer Pareto-optimal modifications: explain why" → "Explore creative modifications"

### 2. Transformed Concise Templates

**Example: Complex.txt**

**Before (28 lines):**
```
Context:
- You are modifying a CIFAR-10 convolutional network using multi-objective optimization.
- Objective 1: Maximize test accuracy (target ≥90%).
- Objective 2: Minimize parameter count to keep inference lightweight.

Constraints:
- Input tensor must remain `(batch, 3, 32, 32)` and output `(batch, 10)`.
- Code must compile/run without errors and respect ConstantRules.

Architectural Levers Available:
- Channel widths / expansion ratios, bottlenecks, and skip paths.
- [... 3 more bullet points]

Desired Change:
- Propose complex architectural enhancements (e.g., channel grouping strategies, 
  attention placement, activation upgrades) that deliver higher accuracy while 
  staying within a justified parameter budget. Reference how each lever influences 
  both objectives.

Trade-off Analysis Required:
- Estimate the parameter delta for every suggested change.
- [... 2 more mandatory analysis points]
```

**After (3 lines):**
```
Q: How would you reimagine this CIFAR-10 architecture to achieve breakthrough 
performance? Consider creative combinations of layer types, attention mechanisms, 
activation functions, or novel architectural patterns. The network seeks high 
accuracy (≥90%) with thoughtful parameter management. Input/output tensors must 
remain `(batch, 3, 32, 32)` → `(batch, 10)`.
```

**Key Changes:**
- From **28 lines of instructions** to **3 lines of inspiration**
- Removed "Context / Constraints / Levers / Desired Change / Analysis" scaffold
- Removed explicit lever enumeration
- Changed "Propose... Reference how... Estimate... Explain..." to "How would you reimagine... Consider..."
- Kept awareness ("seeks high accuracy ≥90% with thoughtful parameter management") without prescription ("must justify parameter increases")

### 3. Simplified All Template Types

**Template Coverage:**
- ✅ **Concise/** (6 files): Complex, Param, RemoveParams, Significant, Weird, ParamWeird
- ✅ **Roleplay/** (18 files): Expert, MadScientist, MrMagoo × 6 mutation types each

**Before/After Comparison:**

| Template | Before | After | Change |
|----------|--------|-------|--------|
| Complex.txt | 28 lines, 5 sections, explicit lever list | 3 lines, open-ended question | -89% |
| Param.txt | 26 lines, mandatory analysis framework | 2 lines, exploratory prompt | -92% |
| Expert_Complex.txt | 30 lines, structured reasoning scaffold | 2 lines, persona + inspiration | -93% |
| MadScientist_Weird.txt | 27 lines, guardrail checklists | 2 lines, energetic exploration | -93% |

**Consistent Pattern:**
1. **Remove scaffolding** - No more "Context / Constraints / Levers / Desired Change / Analysis"
2. **Remove checklists** - No more "Estimate delta / Explain trade-offs / List guardrails"
3. **Remove lever lists** - No pre-specified "available options" that constrain thinking
4. **Maintain awareness** - Keep dual objective context without making it prescriptive
5. **Maintain constraints** - Keep tensor shapes and compilation requirements
6. **Inspire creativity** - Use question framing ("How would you...", "What if...", "Explore...")

### 4. Updated RAG Context Integration

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

## Comparison: Prescriptive vs. Creative Approaches

| Aspect | Prescriptive (Before) | Creative (After) |
|--------|----------------------|------------------|
| **Prompt Length** | 25-30 lines | 2-4 lines |
| **Structure** | Rigid 5-section scaffold | Open-ended question |
| **Lever Guidance** | Explicit enumeration | None (organic discovery) |
| **Trade-off Analysis** | Mandatory checklists | Implied awareness |
| **LLM Role** | Compliance executor | Creative explorer |
| **Expected Output** | Predictable variations | Novel discoveries |
| **Reasoning Style** | Form-filling | Natural ideation |
| **Diversity** | Constrained by lists | Unbounded exploration |

## Implementation Summary

### Files Modified:
1. **`templates/ConstantRules.txt`**: Simplified rules 9-12 (removed 3 prescriptive rules, simplified 1)
2. **`templates/FixedPrompts/concise/*.txt`** (6 files): Reduced from 25-30 lines to 2-4 lines each
3. **`templates/FixedPrompts/roleplay/*.txt`** (18 files): Reduced from 25-30 lines to 2-4 lines each
4. **`src/llm_utils.py`**: Simplified RAG context prefix from directive to inspirational
5. **`templates/ARCHITECTURAL_LEVERS.md`**: Kept as human reference, removed from prompts

### Lines of Code Changed:
- **Before**: ~700 lines of prescriptive prompt text across all templates
- **After**: ~80 lines of inspirational prompt text
- **Reduction**: 89% fewer prompt tokens, 100% more creative freedom

### Verification Steps:
```bash
# Check ConstantRules simplified
cat templates/ConstantRules.txt | grep -A2 "^9\\."

# Verify concise templates shortened
wc -l templates/FixedPrompts/concise/*.txt

# Verify roleplay templates shortened  
wc -l templates/FixedPrompts/roleplay/Expert_*.txt

# Check RAG context updated
grep -A5 "rag_prefix =" src/llm_utils.py
```

## Future Considerations

### What We Kept (Deliberately):
1. **Technical constraints** - Tensor shapes, compilation requirements (necessary for valid code)
2. **Dual objective awareness** - Mention of accuracy target and parameter efficiency (provides context)
3. **Persona differentiation** - Expert/MadScientist/MrMagoo voices (drives creative diversity)

### What We Might Adjust:
1. **Objective target specificity** - Could soften "≥90%" to "high accuracy" for more flexibility
2. **Persona intensity** - Could experiment with more/less pronounced persona characteristics
3. **RAG example count** - Could vary number of retrieved examples to balance inspiration vs. anchoring

### Monitoring Plan:
1. **Generation 0-5**: Baseline diversity and creativity metrics
2. **Compare to pre-revision runs**: Evaluate if simplified prompts produce more novel architectures
3. **A/B test specific personas**: Identify which personas benefit most from creative freedom
4. **Track fitness progression**: Verify Pareto frontier still advances (or accelerates)

## Related Documentation

- `templates/ARCHITECTURAL_LEVERS.md`: Lever reference (for humans, not LLM prompts)
- `docs/07_rag_feature.md`: RAG pipeline integration details
- `src/llm_utils.py`: RAG context integration and prompt mutation logic
- `run_improved.py`: Template generation and mutation logging

## Professor's Guidance

> "The prompts should not be specific as if we are telling the LLM what to do. Rather the prompt gently guides the query letting the LLM's imaginative reasoning handle the mutation. Remember the goal of this research project - to explore novel architectures. This can only be done with a creative element."

**This revision implements that guidance**: Prompts now **inspire** rather than **instruct**, trusting the LLM's creativity to discover unexpected architectural innovations.

