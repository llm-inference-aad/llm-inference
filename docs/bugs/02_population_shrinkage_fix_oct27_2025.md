# Population Shrinkage Robustness Fix

**Date:** October 27, 2025  
**Issue:** Run failure when too many genes fail evaluation  
**Status:** Fixed  
**Severity:** Critical (causes run termination)

## Problem Description

### Error Encountered
```
ValueError: selTournamentDCD: k must be less than or equal to individuals length
```

### Root Cause
When a high percentage of genes fail evaluation (syntax errors, runtime crashes, etc.), the population shrinks after the "Invalid Removal" step. If the population becomes smaller than `population_size`, the selection operator fails because it tries to select more offspring than available parents.

### Example Failure Scenario
```
Generation 0:
- Started with: 8 genes
- Failed evaluation: 4 genes (50% failure rate)
- After invalid removal: 4 genes remain
- Selection step: Tries to select 8 offspring from 4 parents
- Result: ValueError and run termination
```

## Impact

### Run: auto_20251022_193902
- **Status**: Failed in Generation 0
- **Failure rate**: 4/8 genes (50%)
- **No checkpoints saved**: Run terminated before first checkpoint
- **No logs moved**: Cleanup steps never executed

### Failed Genes
All 4 failed genes appear to have had syntax errors from LLM-generated code based on the partial output in the logs.

## Solution

Added dynamic population size adjustment to handle shrinkage gracefully:

### Before (Broken)
```python
# Remove invalid genes
population = [ind for ind in population if ind.fitness.values != INVALID_FITNESS_MAX]

# Selection - FAILS if len(population) < population_size
elites = tools.selSPEA2(population, num_elites)
offspring = toolbox.select(population, population_size)
```

### After (Fixed)
```python
# Remove invalid genes
population = [ind for ind in population if ind.fitness.values != INVALID_FITNESS_MAX]

# Handle complete population failure
if len(population) == 0:
    recreate_population()

# Adjust selection sizes based on actual population
actual_pop_size = len(population)
target_num_elites = min(num_elites, actual_pop_size)
target_offspring_size = min(population_size, actual_pop_size)

if actual_pop_size < population_size:
    log_warning(f"Population shrinkage: {actual_pop_size}/{population_size}")

# Selection with adjusted sizes
elites = tools.selSPEA2(population, target_num_elites)
offspring = toolbox.select(population, target_offspring_size)
```

## Key Changes

1. **Empty population handling**: If all genes fail, attempt to recreate population (up to 5 attempts)
2. **Dynamic size adjustment**: Selection size adapts to actual population size
3. **Clear logging**: Warns when population shrinkage is detected
4. **Graceful degradation**: Run continues with smaller population instead of crashing

## Expected Behavior After Fix

### Scenario 1: Moderate Failure (25-50%)
```
Generation 0:
- Started with: 8 genes
- Failed: 4 genes
- After removal: 4 genes
⚠️  Population shrinkage detected: 4/8 genes survived
   Adjusting selection: elites=4, offspring=4
- Continues with reduced population
- Population can recover in later generations through mutation/crossover
```

### Scenario 2: High Failure (>75%)
```
Generation 0:
- Started with: 8 genes
- Failed: 7 genes
- After removal: 1 gene
⚠️  Population shrinkage detected: 1/8 genes survived
   Adjusting selection: elites=1, offspring=1
- Continues with minimal population
- Will rely heavily on mutation to increase diversity
```

### Scenario 3: Complete Failure (100%)
```
Generation 0:
- Started with: 8 genes
- Failed: 8 genes
- After removal: 0 genes
⚠️  CRITICAL: All genes failed evaluation - recreating population
- Attempts to create fresh population from seed
- If successful, continues run
- If 5 recreation attempts fail, exits with clear error message
```

## Prevention Strategies

To reduce the likelihood of high failure rates:

1. **Improve LLM prompts**: More explicit instructions about syntax requirements
2. **Quality control**: Enable `QC_CHECK_BOOL` to validate code before submission
3. **Fallback detection**: Current fallback system helps reduce wasted evaluations
4. **Temperature tuning**: Lower temperature = more conservative/correct code
5. **Model selection**: Some LLM models produce more syntactically correct code than others

## Monitoring

After this fix, monitor these metrics in future runs:

```bash
# Check for population shrinkage warnings
grep "Population shrinkage" runs/latest/logs/slurm-main-*.out

# Count invalid genes per generation
grep "Invalid Removal" -A 20 runs/latest/logs/slurm-main-*.out | grep "Population Size"

# Calculate failure rate
python scripts/analyze_failure_rate.py runs/latest/
```

## Testing

This fix should be validated with:
1. **Normal run**: Verify no regression when all genes succeed
2. **Moderate failure**: Simulate 50% failure rate and verify graceful handling
3. **High failure**: Simulate 90% failure rate and verify recovery
4. **Complete failure**: Simulate 100% failure and verify recreation logic

## Related Issues

- Fitness inheritance optimization (reduces wasted evaluations)
- Fallback detection (helps identify duplicate genes)
- LLM quality control (prevents syntax errors)

## References

- Error log: `slurm-results/slurm-main-3450563.err`
- Failed run: `runs/auto_20251022_193902/`
- Implementation: `run_improved.py` lines 1042-1070
