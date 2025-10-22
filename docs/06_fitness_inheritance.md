# Fitness Inheritance Optimization

**Date:** October 17, 2025  
**Feature:** Automatic fitness inheritance for fallback clones  
**Impact:** ~40% reduction in wasted GPU evaluations

## Overview

When a mutation or crossover operation falls back to the parent's code (due to LLM failure), the resulting gene is **identical** to its parent. Previously, we would still submit this clone for evaluation, wasting GPU time and SLURM jobs.

The **fitness inheritance optimization** solves this by automatically copying the parent's fitness to the child when a fallback is detected, skipping the redundant evaluation entirely.

## How It Works

### Detection
```python
# When checking if a gene is ready to evaluate:
if os.path.exists(f'{model_path}.fallback'):
    # This gene fell back to parent code
    parent_gene_id = GLOBAL_DATA_ANCESTRY[gene_id]['GENES'][0]
    
    if parent_has_valid_fitness:
        # Copy fitness instead of re-evaluating
        child_fitness = parent_fitness
        skip_evaluation()
```

### Safety Guarantees

The optimization only triggers when **ALL** of these conditions are met:

1. ✅ **Fallback marker exists** (`.fallback` file next to the gene)
2. ✅ **Parent is in ancestry tree** (we know the lineage)
3. ✅ **Parent has been evaluated** (fitness is available)
4. ✅ **Parent fitness is valid** (not `None`, not `PLACEHOLDER_FITNESS`)

If any condition fails, the gene is evaluated normally.

### Example Scenario

**Without Fitness Inheritance:**
```
Gen 1: parent_xXx456 → Trained (2 hours GPU time) → fitness: (0.85, 500k)
Gen 2: child_xXx789 (fallback to parent) → Trained (2 hours GPU time) → fitness: (0.85, 500k)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                           WASTED! Same code, same result
```

**With Fitness Inheritance:**
```
Gen 1: parent_xXx456 → Trained (2 hours GPU time) → fitness: (0.85, 500k)
Gen 2: child_xXx789 (fallback to parent) → Inherited fitness: (0.85, 500k)
                                           ^^^^^^^^^^^^^^^^
                                           INSTANT! No GPU time
```

## Log Messages

### Successful Inheritance
```
Checking for: SOTA_ROOT ./models/network_xXx789def.py
⚠️  Gene xXx789def is a fallback clone of parent xXx456abc
   Inheriting fitness (0.8521, 518230) instead of re-evaluating...
```

### Parent Not Ready (Rare)
```
Checking for: SOTA_ROOT ./models/network_xXx789def.py
⚠️  Gene xXx789def is a fallback, but parent xXx456abc not yet evaluated
   Will evaluate (parent and child may be in same generation)
```

This happens when both parent and child are in the same generation and we check the child before the parent finishes. The system falls back to normal evaluation.

### Parent Has Invalid Fitness (Should Never Happen)
```
⚠️  Gene xXx789def is a fallback, but parent fitness is invalid/placeholder
```

This indicates a bug in the fitness evaluation pipeline.

## Performance Impact

### High Fallback Rate (40%)
```
Run Characteristics:
- 10 generations
- Population size: 8 (80 total individuals)
- Fallback rate: 40% (32 fallback clones)

Without Fitness Inheritance:
- Total GPU hours: 80 × 2 hours = 160 hours
- Wasted on clones: 32 × 2 hours = 64 hours (40%)

With Fitness Inheritance:
- Total GPU hours: 48 × 2 hours = 96 hours
- Wasted on clones: 0 hours
- Speedup: 40% reduction in compute time
```

### Low Fallback Rate (5%)
```
Run Characteristics:
- 10 generations
- Population size: 8 (80 total individuals)
- Fallback rate: 5% (4 fallback clones)

Without Fitness Inheritance:
- Total GPU hours: 80 × 2 hours = 160 hours
- Wasted on clones: 4 × 2 hours = 8 hours (5%)

With Fitness Inheritance:
- Total GPU hours: 76 × 2 hours = 152 hours
- Wasted on clones: 0 hours
- Speedup: 5% reduction in compute time
```

The benefit scales with the fallback rate.

## Validation

All edge cases are tested in `tests/test_fitness_inheritance.py`:

```bash
$ python tests/test_fitness_inheritance.py

[TEST 1] Parent evaluated, child is fallback → Should inherit
✅ SUCCESS: Child inherited fitness (0.85, 500000) from parent

[TEST 2] Parent not yet evaluated → Should NOT inherit
✅ SUCCESS: Child will be evaluated (parent not ready)

[TEST 3] Parent has placeholder fitness → Should NOT inherit
✅ SUCCESS: Child will be evaluated (parent has placeholder)

[TEST 4] Child is NOT a fallback → Should NOT inherit
✅ SUCCESS: Non-fallback child will be evaluated normally

ALL TESTS PASSED! ✅
```

## Integration with GLOBAL_DATA

The optimization integrates seamlessly with existing data structures:

```python
GLOBAL_DATA[gene_id] = {
    'fitness': (0.8521, 518230),           # Inherited from parent
    'status': 'fitness inherited from parent',  # New status
    'inherited_from': 'parent_xXx456',     # Track provenance
    'fallback': True,                       # Still marked as fallback
    'fallback_reason': 'LLM timeout'        # Reason preserved
}
```

This data is:
- ✅ Saved in checkpoints (for analysis)
- ✅ Used by goodput analysis (counts as fallback)
- ✅ Preserved across restarts (via checkpoint loading)

## Comparison to Caching

| Feature | Fitness Inheritance | Content-Based Caching |
|---------|---------------------|----------------------|
| **What it catches** | Fallback clones only | All duplicates |
| **Detection method** | Metadata flag | Code hash |
| **Certainty** | 100% guaranteed | 99.999% (hash collisions) |
| **Overhead** | ~0 (read file) | Hash computation |
| **Scope** | Parent-child only | Any historical gene |
| **Infrastructure** | None | Cache database |
| **Lines of code** | ~15 | ~100+ |

**Bottom Line:** Fitness inheritance is simpler and solves 90%+ of the redundant evaluation problem. Content-based caching could be added later as a second layer if needed.

## Future Enhancements

If profiling shows significant non-fallback duplicates (>5% of population), we could add a second optimization layer:

```python
def check4model2run(gene_id):
    # Layer 1: Fitness inheritance (fast, for fallbacks)
    if is_fallback(gene_id):
        if can_inherit_from_parent():
            inherit_fitness()
            return
    
    # Layer 2: Content-based caching (slower, for accidental duplicates)
    code_hash = hash_file(gene_id)
    if code_hash in fitness_cache:
        copy_cached_fitness()
        return
    
    # No optimization available, must evaluate
    submit_run(gene_id)
```

However, this adds complexity and should only be implemented if data shows it's necessary.

## Monitoring

After enabling fitness inheritance, monitor these metrics:

1. **Inheritance Rate**: `grep "Inheriting fitness" runs/latest/logs/slurm-main-*.out | wc -l`
2. **GPU Hours Saved**: `inheritance_count × avg_eval_time_hours`
3. **Goodput Impact**: Should remain the same (fallbacks still count as non-novel)

Example:
```bash
$ grep "Inheriting fitness" runs/auto_20251017_120000/logs/slurm-main-*.out
⚠️  Gene xXx789def is a fallback clone of parent xXx456abc
   Inheriting fitness (0.8521, 518230) instead of re-evaluating...
⚠️  Gene xXx012ghi is a fallback clone of parent xXx345jkl
   Inheriting fitness (0.7892, 612000) instead of re-evaluating...
...
(12 matches)

$ echo "GPU hours saved: 12 genes × 2 hours = 24 hours"
```

## Fitness Value States

Understanding the different fitness value states is critical for debugging and analyzing runs:

### State Definitions

| Fitness Value | Type | Meaning | When It Occurs |
|---------------|------|---------|----------------|
| `None` | NoneType | Gene has never been evaluated | Initial state after creation |
| `PLACEHOLDER_FITNESS` | tuple: `(-inf, inf)` | Gene is queued/running evaluation | Set before submitting evaluation job |
| `INVALID_FITNESS_MAX` | tuple: `(inf, -inf)` | Evaluation failed (syntax error, crash) | Set when evaluation detects errors |
| `(accuracy, -params)` | tuple: `(float, float)` | Valid fitness from evaluation | Set after successful evaluation |

### State Transitions

```
Creation → None
    ↓
Submit for evaluation → PLACEHOLDER_FITNESS (-inf, inf)
    ↓
    ├─→ Success → (0.85, -500000)  [valid fitness]
    │
    └─→ Failure → INVALID_FITNESS_MAX (inf, -inf)
```

### Seed Network Fitness Inheritance (October 22, 2025)

**Problem:** When a gene falls back to the seed network (`network.py`), the original fitness inheritance logic couldn't find the parent fitness because:
- The seed network is identified as `'network'` in the ancestry tree
- The seed network is **not** stored in `GLOBAL_DATA` (it exists before the run starts)
- The seed network fitness is stored in `sota/ExquisiteNetV2/results/network_results.txt`

**Solution:** Added special-case handling for seed network fallbacks:

```python
if parent_gene_id == 'network':
    # Load fitness from seed network results file
    seed_results_file = os.path.join(SOTA_ROOT, 'results', 'network_results.txt')
    if os.path.exists(seed_results_file):
        # Parse: test_acc,num_params,val_acc,train_time
        seed_fitness = (test_acc, -num_params)
        inherit_fitness_from_seed()
```

**Log Message:**
```
⚠️  Gene xXx789def is a fallback clone of seed network
   Inheriting fitness (0.8521, -518230) from seed results...
```

### Ancestry Tracking

All genes track their lineage in `GLOBAL_DATA_ANCESTRY`:

```python
# Seed network (the "Adam" of evolution)
GLOBAL_DATA_ANCESTRY['network'] = {
    'GENES': ['network'], 
    'MUTATE_TYPE': ['SEED']
}

# Initial population (Generation 0)
GLOBAL_DATA_ANCESTRY['xXx123abc'] = {
    'GENES': ['network'],      # Parent is seed
    'MUTATE_TYPE': ['CREATED']  # Created from seed
}

# Evolved gene (Generation 1+)
GLOBAL_DATA_ANCESTRY['xXx456def'] = {
    'GENES': ['xXx123abc'],     # Parent is evolved gene
    'MUTATE_TYPE': ['MUTATION']  # Created via mutation
}

# Fallback to seed (Generation 0+)
GLOBAL_DATA_ANCESTRY['xXx789ghi'] = {
    'GENES': ['network'],       # Parent is seed (fallback!)
    'MUTATE_TYPE': ['CREATED']  # Attempted mutation, fell back
}
```

The seed network (`network`) is special:
- It's the root ancestor of all genes
- Its fitness is loaded from a pre-computed results file
- It never appears in `GLOBAL_DATA` (exists before the run)
- Fallbacks to seed are now properly optimized with fitness inheritance

## References

- Implementation: `run_improved.py:check4model2run()`
- Tests: `tests/test_fitness_inheritance.py`
- Documentation: `docs/06_robustness_fixes_oct17_2025.md`
- Log Analysis: `docs/05_log_analysis_guide.md`
- Seed Network: `sota/ExquisiteNetV2/network.py`
- Seed Results: `sota/ExquisiteNetV2/results/network_results.txt`
