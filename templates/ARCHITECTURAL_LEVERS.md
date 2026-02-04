# Architectural Lever Reference

Use this guide to ground every template, persona, and RAG context in concrete architectural options that affect the accuracy/parameter trade-off.

## Mutation-Type → Lever Map

| Mutation Type | Primary Objective | Suggested Levers | Trade-off Notes |
| --- | --- | --- | --- |
| `Complex` | Increase accuracy with minimal param growth | Channel width rebalancing, staged SE / SE_LN attention, advanced activations (SiLU, Hardswish, Mish, GELU), mixed pooling strategies, cross-stage fusion helpers | Justify any parameter increase with expected accuracy gains; reuse helpers to contain growth |
| `Param` | Improve convergence/generalization without structural edits | Optimizer schedules (SGD ↔ AdamW), LR warmup + cosine decay, BatchNorm momentum tuning, dropout/stochastic-depth adjustments, gradient clipping | No parameter change allowed; emphasize monitoring + rollback strategies |
| `RemoveParams` | Reduce total parameters while preserving accuracy | Depthwise/grouped convolutions, bottleneck/1x1 reduction, channel pruning, weight sharing/tied classifiers, low-rank projections | Highlight mitigation tactics (attention placement, activation upgrades) to retain accuracy |
| `Significant` | Major structural refactors + helper introduction | Stage builder helpers, activation/attention factories, hierarchical residual stacks, observability hooks | Document helper interfaces, testing plans, and parameter/complexity impacts |
| `Weird` / `ParamsWeird` | Explore unconventional yet defensible ideas | Dynamic pooling/activation routing, reversible blocks, oscillating hyperparameters, curriculum-based dropout, layer-wise optimizer cocktails | Always connect creativity back to the Pareto frontier and define guardrails |
| `Roleplay Personas (Expert / Helper / MadScientist / MrMagoo)` | Persona-specific tone layered on the above mutation types | Align persona guidance with the same lever set; emphasize trade-off reasoning and safeguards regardless of tone | Personas change storytelling, not objectives—always cite accuracy vs. parameter impact |

## Core Lever Details

- **Channel Widths / Expansion Ratios:** Adjust ME/EVE/DFSEB modules to redistribute capacity. Increasing width boosts accuracy but costs parameters; balance with bottlenecks or shared heads.
- **Depthwise Separable vs. Standard Convs:** Replace heavy convolutions with depthwise + pointwise pairs to slash parameters while retaining receptive field coverage.
- **Attention Mechanisms (SE / SE_LN):** Lightweight squeeze/excitation blocks add minimal parameters yet improve feature calibration; ideal for accuracy-focused mutations.
- **Activation Functions:** Swapping ReLU for SiLU, Mish, Hardswish, or GELU impacts accuracy without changing parameter counts; ensure consistency throughout blocks.
- **Pooling Strategies:** Combine max/min/adaptive pooling or learned pooling kernels to enhance feature diversity at zero parameter cost.
- **Optimizer & Normalization Tweaks:** Learning-rate schedules, momentum, weight decay, BatchNorm momentum/epsilon, and dropout/stochastic depth control convergence/regularization without structural changes.
- **Helper Utilities:** Encapsulate repeated logic (stage builders, attention factories, activation pipelines) to keep code modular and experimentation-friendly while managing parameter usage.

## Trade-off Checklist

1. **Quantify Impact:** For every modification, estimate relative parameter change and expected accuracy shift.
2. **Guardrails:** Define monitoring metrics, rollback toggles, and validation steps (unit tests, tensor assertions).
3. **Pareto Mindset:** Favor changes that improve accuracy without adding parameters, or justify additions with measurable accuracy gains.
4. **Documentation:** Update prompts, personas, and RAG context with lever references so the LLM always understands the available toolkit.


