# Sprint Planning: Next Iteration

**Date:** October 20, 2025  
**Sprint Goal:** Optimize retry logic, implement error-driven prompt engineering, and fine-tune evolutionary parameters based on empirical data.

---

## Questions to Address

### 1. **Track API Retry Attempts Per Individual**

**Goal:** Measure the effectiveness of our retry loop to find the optimal `LLM_GENERATION_MAX_RETRIES` value.

**Current State:**
- We have **two levels of retries**:
  1. **Server-level retries** (in `submit_local_server()`): Currently 3 attempts with exponential backoff for transient network issues.
  2. **Validation-level retries** (in `llm_mutation.py`): Currently 3 attempts (`LLM_GENERATION_MAX_RETRIES`) to generate code that passes validation.

**Problem:**
- We don't currently track **which attempt** produces a valid individual.
- This prevents us from answering: "Does retry #3 meaningfully improve success rate, or does it just waste compute?"

**Proposed Solution:**

Add a `retry_attempts` field to `GLOBAL_DATA` to track how many validation retries were needed:

```python
# In src/llm_mutation.py, line ~39:
for attempt in range(LLM_GENERATION_MAX_RETRIES):
    try:
        code_from_llm = generate_augmented_code(...)
        # ... validation logic ...
        validate_module_source(...)
        
        # SUCCESS: Record the attempt number
        fallback_reason = None
        retry_attempts = attempt + 1  # 1-indexed for human readability
        break
    except Exception as exc:
        fallback_reason = str(exc)
        retry_attempts = attempt + 1
        # ... continue retry loop ...

# After loop, store in GLOBAL_DATA (passed from run_improved.py)
# This would require passing GLOBAL_DATA reference or gene_id to mutation function
```

**Better approach:** Return retry count from mutation/crossover functions:

```python
# In src/llm_mutation.py:
def mutate_from_llm(...):
    retry_attempts = 0
    for attempt in range(LLM_GENERATION_MAX_RETRIES):
        retry_attempts = attempt + 1
        try:
            # ... mutation logic ...
            break
        except:
            continue
    
    # Return both the success flag and retry count
    return {
        'success': fallback_reason is None,
        'retry_attempts': retry_attempts,
        'fallback_reason': fallback_reason
    }

# In run_improved.py, after calling mutation:
result = mutate_from_llm(...)
GLOBAL_DATA[gene_id]['retry_attempts'] = result['retry_attempts']
GLOBAL_DATA[gene_id]['fallback'] = not result['success']
```

**Analysis Script:**

Create `scripts/analyze_retry_effectiveness.py`:

```python
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# Load checkpoints
retry_counts = []
for checkpoint in glob("runs/*/checkpoints/*.pkl"):
    data = pickle.load(open(checkpoint, 'rb'))
    for gene_id, gene_data in data['GLOBAL_DATA'].items():
        if 'retry_attempts' in gene_data:
            retry_counts.append(gene_data['retry_attempts'])

# Analyze distribution
counter = Counter(retry_counts)
print("Retry Distribution:")
for attempt in sorted(counter.keys()):
    count = counter[attempt]
    pct = count / len(retry_counts) * 100
    print(f"  Attempt {attempt}: {count} individuals ({pct:.1f}%)")

# Recommendation
success_on_first = counter.get(1, 0)
success_on_second = counter.get(2, 0)
success_on_third = counter.get(3, 0)

if success_on_third < success_on_second * 0.1:
    print("\n💡 RECOMMENDATION: Reduce LLM_GENERATION_MAX_RETRIES to 2")
    print(f"   Reason: Only {success_on_third} individuals succeeded on 3rd attempt")
```

**Expected Insight:**

If we find that 90% of valid individuals succeed on attempt 1-2, but only 5% on attempt 3, we can reduce `LLM_GENERATION_MAX_RETRIES=2` and save ~30% of LLM compute time.

---

### 2. **Error-Driven Prompt Engineering**

**Goal:** Use the context from previous failures to guide the LLM toward valid code generation.

**Current State:**

The `fallback_reason` field in `GLOBAL_DATA` contains:
- **Network errors:** `"Error calling local server after 3 attempts: ..."`
- **Validation errors:** `"RuntimeError: Given normalized_shape=[12], expected input with shape [*, 12], but got input of size[216, 12, 1, 1]"`
- **Syntax errors:** `"SyntaxError: invalid syntax (<string>, line 42)"`

**Problem:**

We currently **discard** this valuable error information. The LLM makes the same mistakes repeatedly because it has no memory of what didn't work.

**Proposed Solution:**

Implement **error-aware prompting** by appending failure context to the LLM prompt:

```python
# In src/llm_mutation.py, modify the retry loop:

previous_errors = []

for attempt in range(LLM_GENERATION_MAX_RETRIES):
    # Construct prompt with error history
    if attempt == 0:
        txt2llm = template_txt.format(code2llm.strip())
    else:
        # Add error context for retries
        error_context = "\n\n".join([
            f"### Previous Attempt {i+1} Error:\n{err}"
            for i, err in enumerate(previous_errors)
        ])
        
        txt2llm = template_txt.format(code2llm.strip()) + f"""

The previous attempts to modify this code failed with the following errors:

{error_context}

Please generate code that avoids these specific errors. Pay special attention to:
- Tensor shape compatibility
- Layer input/output dimensions
- PyTorch syntax correctness
"""
    
    try:
        code_from_llm = generate_augmented_code(txt2llm, ...)
        # ... validation ...
        validate_module_source(...)
        break  # Success!
    except Exception as exc:
        fallback_reason = str(exc)
        previous_errors.append(fallback_reason)
```

**Expected Benefits:**

1. **Faster convergence:** LLM learns from its mistakes within the same individual.
2. **Higher success rate:** Reduces fallback rate from 4.4% to potentially <2%.
3. **Better code quality:** LLM becomes aware of common pitfalls in this specific architecture.

**Alternative Approach (Simpler):**

Instead of modifying prompts, we could **analyze fallback patterns** across the entire run and create a **global "lessons learned" document** that gets appended to all future prompts:

```python
# After analyzing past runs, create a static guide:
COMMON_PITFALLS = """
### Common Errors to Avoid:

1. **Tensor Shape Mismatches:**
   - Always verify input shapes before applying LayerNorm, BatchNorm, etc.
   - Example: LayerNorm(12) expects input shape [..., 12], not [216, 12, 1, 1]

2. **Dimension Reduction Errors:**
   - When using pooling/stride, ensure output dimensions are valid
   - Example: Don't apply stride=2 to a 1x1 feature map

3. **Invalid Layer Sequences:**
   - Don't apply adaptive pooling after output layer
   - Activation functions must precede normalization in most cases
"""

# Append to all prompts:
txt2llm = template_txt.format(code2llm.strip()) + COMMON_PITFALLS
```

This is **simpler to implement** and provides value immediately without complex retry logic.

---

### 3. **Should We Tweak Constants for the Next Run?**

**Current Configuration:**

```python
num_generations = 8
start_population_size = 16
population_size = 8
crossover_probability = 0.35
mutation_probability = 0.8
PROB_EOT = 0.25
LLM_GENERATION_MAX_RETRIES = 3
```

**Analysis from `auto_20251017_175557`:**

| Metric | Value | Assessment |
|--------|-------|------------|
| **Goodput** | 95.6% average | ✅ Excellent |
| **Fallback Rate** | 4.4% | ✅ Very good |
| **Generations** | 8 | ⚠️ May be too short to see convergence |
| **Population Size** | 14-16 → 8 | ✅ Working well |
| **Server Availability** | Offline Gen 0-1 | ⚠️ External factor |

**Recommendations:**

#### **Option A: Conservative (Recommended for Next Run)**

Keep current settings to **validate the ancestry bug fix** and measure fitness inheritance impact:

```python
num_generations = 8           # UNCHANGED
start_population_size = 16    # UNCHANGED
population_size = 8           # UNCHANGED
crossover_probability = 0.35  # UNCHANGED
mutation_probability = 0.8    # UNCHANGED
```

**Rationale:** We need a **clean A/B comparison** between the buggy run and the fixed run. Changing multiple variables makes it impossible to isolate the impact of fitness inheritance.

#### **Option B: Longer Run (For Subsequent Runs)**

After validating the bug fix, increase generation count to see if the population converges:

```python
num_generations = 15          # INCREASE to observe Pareto front stabilization
start_population_size = 16    # UNCHANGED
population_size = 8           # UNCHANGED
```

**Expected Outcome:** Pareto front should stabilize around Gen 10-12, indicating we've explored the search space sufficiently.

#### **Option C: Larger Population (Future Experiment)**

To increase diversity and reduce the risk of premature convergence:

```python
num_generations = 10          # Moderate increase
start_population_size = 24    # INCREASE for more diverse Gen 0
population_size = 12          # INCREASE to maintain diversity
```

**Trade-off:** More diversity, but **50% more GPU hours** per generation.

#### **Option D: Higher Mutation Rate (Not Recommended)**

```python
mutation_probability = 0.9    # INCREASE from 0.8
```

**Concern:** We already have 95.6% goodput with 0.8 mutation rate. Increasing it risks generating more invalid individuals without meaningful diversity gains.

---

### **Recommendation Matrix:**

| Parameter | Current | Next Run | Subsequent Runs |
|-----------|---------|----------|-----------------|
| `num_generations` | 8 | **8** (validate bug fix) | 15 (observe convergence) |
| `start_population_size` | 16 | **16** | 16 or 24 (if diversity needed) |
| `population_size` | 8 | **8** | 8 or 12 (if diversity needed) |
| `crossover_probability` | 0.35 | **0.35** | 0.35 |
| `mutation_probability` | 0.8 | **0.8** | 0.8 |
| `PROB_EOT` | 0.25 | **0.25** | 0.25-0.35 (if elites stagnate) |
| `LLM_GENERATION_MAX_RETRIES` | 3 | **3** (collect data) | 2 (if analysis shows diminishing returns) |

---

## Action Items for Next Sprint

### **High Priority**

1. **✅ Implement Server Shutdown Automation**
   - Add server job ID tracking to `server.sh`
   - Add `scancel` logic to `run.sh` after job completes
   - Move server logs to `runs/{RUN_ID}/logs/`

2. **❌ Enable Remote Fallback**
   - Set `ENABLE_LLM_REMOTE_FALLBACK=true` in `.env`
   - Verify Mixtral HuggingFace credentials are configured

3. **✅ Validation Run with Bug Fix**
   - Launch new run with **unchanged parameters** (Option A)
   - Monitor for `"Inheriting fitness"` log messages
   - Measure GPU hours saved by fitness inheritance

### **Medium Priority**

4. **📊 Add Retry Tracking Metric**
   - Modify `llm_mutation.py` to return `retry_attempts`
   - Store in `GLOBAL_DATA[gene_id]['retry_attempts']`
   - Create analysis script to determine optimal `LLM_GENERATION_MAX_RETRIES`

5. **🧠 Implement Error-Aware Prompting (Simple Version)**
   - Create `COMMON_PITFALLS` guide from past fallback reasons
   - Append to all LLM prompts
   - Measure impact on fallback rate

### **Low Priority (Future Work)**

6. **📈 Extended Run for Convergence Analysis**
   - After validating bug fix, run with `num_generations=15`
   - Analyze when Pareto front stabilizes
   - Determine optimal run length

7. **🔬 Diversity Experiments**
   - Test larger population sizes (`population_size=12`)
   - Measure diversity metrics (genotype uniqueness, phenotype coverage)
   - Compare compute cost vs. solution quality

---

## Success Metrics for Next Run

| Metric | Current Baseline | Target | Method |
|--------|------------------|--------|--------|
| **Fitness Inheritance Events** | 0 (bug prevented it) | ≥5 (one per fallback) | `grep -c "Inheriting fitness" logs/slurm-main-*.out` |
| **GPU Hours Saved** | 0 | ~8 hours | `(inheritance_count * avg_eval_time) - overhead` |
| **Goodput** | 95.6% | ≥95% (maintain) | `scripts/plot_latency_vs_goodput.py` |
| **Fallback Rate** | 4.4% | <4% (with error prompting) | `scripts/plot_latency_vs_goodput.py` |
| **Server Uptime** | Offline Gen 0-1 | 100% (with remote fallback) | Check server logs for connection errors |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Server downtime during Gen 0** | Medium | High | Enable remote fallback to Mixtral |
| **Fitness inheritance still broken** | Low | High | Verify ancestry structure in checkpoint after Gen 0 |
| **Error prompting increases latency** | Medium | Low | Monitor LLM response time; revert if >20% slower |
| **Retry tracking adds overhead** | Low | Low | Store only after mutation completes |

---

## Questions for Discussion

1. **Should we implement error-aware prompting now, or wait until we have retry attempt data?**
   - **Recommendation:** Start with the simple `COMMON_PITFALLS` guide (quick win), then implement full error history if needed.

2. **How much GPU budget do we have for extended runs?**
   - If limited: Stick to 8 generations for validation.
   - If ample: Run 15 generations to measure convergence.

3. **Do we want to compare multiple configurations in parallel?**
   - Could launch 2 runs simultaneously: one with current settings, one with higher diversity.
   - **Recommendation:** No, validate the bug fix first with a clean A/B comparison.

---

## Timeline Estimate

| Task | Effort | Dependencies |
|------|--------|--------------|
| Server shutdown automation | 2 hours | None |
| Enable remote fallback | 30 min | None |
| Validation run | 16-24 hours | Server shutdown |
| Retry tracking implementation | 3 hours | None |
| Error-aware prompting (simple) | 1 hour | None |
| Analysis scripts | 2 hours | Validation run complete |

**Total Development Time:** ~8 hours  
**Total Run Time:** 16-24 hours (depends on server speed)  
**Sprint Duration:** 2-3 days
