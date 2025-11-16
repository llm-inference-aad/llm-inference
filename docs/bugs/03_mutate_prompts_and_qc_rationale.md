# Bug Fix: mutate_prompts() Code Generation Issue & PROB_QC Rationale

**Date:** November 2024  
**Status:** Fixed  
**Severity:** Medium (affects template mutation quality, not core evolution)

## Issue Summary

The `mutate_prompts()` function was generating mutant template files (`mutant0.txt`, `mutant1.txt`, etc.) that sometimes contained Python code examples instead of pure prompt text. Additionally, there was confusion about why quality control (QC) uses probabilistic sampling (`PROB_QC`) instead of running on every mutation/crossover.

## Problem 1: mutate_prompts() Generating Code Examples

### Description

The `mutate_prompts()` function asks an LLM to rephrase existing prompt templates to create variant templates. However, the LLM sometimes interpreted this as a request to generate code examples, resulting in mutant files like:

```python
# mutant0.txt (incorrect output)
python
def train_model():
    model = Model()
    # ... code example ...
    
```python
{}
```
```

Instead of:

```
# mutant0.txt (expected output)
Q: What modifications can be made to enhance this code snippet's performance?

The current code block:
```python
{}
```
```

### Root Cause

The original prompt was too vague:
```python
prompt_base = "Can you rephrase this text:\n```\n{}\n```".format(prompt_text)
```

This didn't explicitly instruct the LLM to:
1. Return prompt text only (not code)
2. Avoid including code examples
3. Preserve the `{}` placeholder format

### Impact

- **Low severity**: These mutant files are **not used as input templates** during evolution
- Only the original templates in `concise/` and `roleplay/` directories are used
- However, it creates confusion and pollutes the template directory with malformed files
- If someone manually selects a mutant file, it could cause issues

### Fix

Updated `mutate_prompts()` in `src/llm_utils.py` to explicitly request prompt text only:

```python
prompt_base = (
    "Rephrase the following prompt template text. "
    "Return ONLY the rephrased prompt text, do NOT include any code examples or code blocks. "
    "The output should be a prompt template that can be used to instruct an LLM to modify code. "
    "Preserve the placeholder {} where code should be inserted.\n\n"
    "Original prompt template:\n```\n{}\n```\n\n"
    "Rephrased prompt template (text only, no code):"
).format(prompt_text)
```

Additionally, added post-processing to strip any code blocks that might slip through:
```python
# Remove any code blocks that LLM might have generated
if "```" in output:
    output = output.split("```")[0].strip()
```

### Files Modified

- `src/llm_utils.py`: `mutate_prompts()` function

---

## Problem 2: Why PROB_QC Instead of Always QC?

### Question

Why use probabilistic quality control (`PROB_QC = 0.0` by default) instead of running QC on every mutation/crossover?

### Answer: Cost vs. Quality Tradeoff

**Quality Control (QC) is expensive:**

1. **Double LLM calls**: QC requires a second LLM inference call after the initial mutation/crossover
   - First call: Generate mutated code
   - Second call: Validate and refine the generated code
   - **2x inference cost** per operation

2. **Latency impact**: Each QC check adds significant latency
   - Mutation: ~5-10 seconds
   - QC check: ~5-10 seconds additional
   - **Doubles the time** for each operation

3. **API costs**: If using paid LLM APIs (Hugging Face, OpenAI), QC doubles the cost

4. **Throughput reduction**: With QC enabled, the system processes half as many mutations per unit time

### Current Implementation

```python
# In run_improved.py:write_bash_script()
qc_check = random.random() < PROB_QC  # Default: PROB_QC = 0.0 (disabled)
```

**Default: `PROB_QC = 0.0`** means QC is disabled by default.

### When to Enable QC

QC should be enabled (`PROB_QC > 0.0`) when:

1. **High mutation failure rate**: If many mutations produce invalid code
2. **Debugging**: When investigating why mutations fail
3. **Quality over speed**: When you prioritize code quality over throughput
4. **Experimentation**: Testing whether QC improves overall evolution quality

### Recommended Settings

- **Baseline runs**: `PROB_QC = 0.0` (disabled) - maximize throughput
- **Quality-focused runs**: `PROB_QC = 0.3-0.5` - balance quality and speed
- **Debugging**: `PROB_QC = 1.0` (always on) - catch all issues

### How QC Works

When `apply_quality_control=True` is passed to `llm_mutation.py` or `llm_crossover.py`:

1. LLM generates initial mutated code
2. QC function (`llm_code_qc()` or `llm_code_qc_hf()`) is called
3. QC sends the generated code + original code to LLM with a validation prompt
4. LLM refines/validates the code
5. Refined code is returned

**Location**: `src/llm_utils.py:llm_code_qc_hf()`

```python
def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    template_path = os.path.join(ROOT_DIR, f'templates/{template_name}')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    prompt2llm = template_txt.format(code_from_llm, base_code)
    # LLM validates and refines the code
    code_from_llm = submit_mixtral_hf(prompt2llm, ...)
    return code_from_llm
```

### Alternative: Syntax-Only Validation

Instead of expensive LLM-based QC, the system uses **syntax validation** on every mutation:

```python
# In src/llm_utils.py:generate_augmented_code()
is_valid, validation_error = _validate_python_snippet(candidate_code)
if is_valid:
    return candidate_code
# Retry if invalid
```

This catches syntax errors without the cost of LLM QC.

---

## Historical Context

### Incorrect Documentation

The file `docs/critical_fixes_oct15_2025.md` (lines 24-42) incorrectly claims that mutant template files were "fixed" by restoring them to persona-based format. However:

1. **The actual issue** was in `mutate_prompts()` function generating code examples
2. **The mutant files** are outputs, not inputs - they don't affect evolution
3. **The real fix** needed was in the prompt engineering, not the template files themselves

### What Was Actually Fixed in Oct 2024

The Oct 2024 fixes addressed:
- ✅ SLURM log polling paths
- ✅ Token limits for API endpoints
- ✅ Probabilistic QC sampling (making `PROB_QC` functional)
- ❌ **NOT** the `mutate_prompts()` code generation issue (fixed in Nov 2024)

---

## Testing

### Verify mutate_prompts() Fix

1. Run `mutate_prompts()`:
   ```python
   from src.llm_utils import mutate_prompts
   mutate_prompts(n=3)
   ```

2. Check generated files:
   ```bash
   cat templates/FixedPrompts/concise/mutant0.txt
   ```

3. Verify output:
   - ✅ Should contain prompt text only
   - ✅ Should include `{}` placeholder
   - ✅ Should NOT contain Python code examples
   - ✅ Should end with `\n```python\n{}\n````

### Verify PROB_QC Behavior

1. Set `PROB_QC = 0.5` in `src/cfg/constants.py`
2. Run a short evolution (1-2 generations)
3. Check SLURM logs for `--apply_quality_control` flag:
   ```bash
   grep "apply_quality_control" slurm-results/llm-*.out
   ```
4. Approximately 50% of mutations should have `'True'` and 50% should have `'False'`

---

## Related Files

- `src/llm_utils.py`: `mutate_prompts()` function
- `src/cfg/constants.py`: `PROB_QC` constant (default: 0.0)
- `run_improved.py`: QC sampling logic in `write_bash_script()`
- `src/llm_mutation.py`: QC application in mutation pipeline
- `src/llm_crossover.py`: QC application in crossover pipeline
- `docs/critical_fixes_oct15_2025.md`: Historical documentation (partially incorrect)

---

## Future Improvements

1. **Template validation**: Add validation to ensure mutant files are valid prompt templates
2. **QC effectiveness metrics**: Track whether QC actually improves mutation success rate
3. **Adaptive QC**: Automatically adjust `PROB_QC` based on mutation failure rate
4. **Syntax-only QC**: Consider expanding syntax validation to catch more errors without LLM cost

