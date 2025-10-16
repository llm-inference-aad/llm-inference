# Log Analysis Guide

This guide provides a comprehensive overview of the logging system for the LLM Guided Evolution (LLMGE) framework. Understanding these logs is crucial for debugging runs, analyzing experimental results, and monitoring the health of the system.

## Table of Contents
1.  [Log File Structure](#log-file-structure)
2.  [Interpreting Core Log Files](#interpreting-core-log-files)
    -   [Main SLURM Output (`slurm-main-*.out`)](#main-slurm-output-slurm-main-out)
    -   [Evaluation Logs (`eval-*.out`, `eval-*.err`)](#evaluation-logs-eval-out-eval-err)
    -   [LLM Operation Logs (`llm-*.out`, `llm-*.err`)](#llm-operation-logs-llm-out-llm-err)
    -   [Results Files (`*_results.txt`)](#results-files-_resultstxt)
3.  [Common Issues and Debugging](#common-issues-and-debugging)
    -   [Identical Fitness Scores in a Generation](#identical-fitness-scores-in-a-generation)
    -   [Job Timeouts and Log Migration](#job-timeouts-and-log-migration)
    -   [`NameError` in Evaluation Logs](#nameerror-in-evaluation-logs)
    -   [Fallback Messages in LLM Logs](#fallback-messages-in-llm-logs)

## Log File Structure

When you execute a run (e.g., via `run.sh`), a unique directory is created in the `runs/` folder, named with a timestamp, like `runs/auto_20251015_180811`. All artifacts for that run are stored within this directory.

```
runs/
└── auto_20251015_180811/
    ├── genes/              # Contains the Python code for each individual gene
    ├── logs/               # Aggregated SLURM logs for the run
    │   ├── slurm-main-3403227.out
    │   ├── eval-3403230.out
    │   ├── eval-3403230.err
    │   ├── llm-3403235.out
    │   └── ...
    ├── population.pkl      # Pickled DEAP population object
    └── results/            # Fitness and metric results for each individual
        ├── 0_results.txt
        └── ...
```

-   **`genes/`**: Stores the generated Python code for each neural network architecture.
-   **`logs/`**: This is where all the SLURM output files (`.out` and `.err`) are moved after the main job completes. If a job times out, you may need to move these logs manually using the `scripts/migrate_slurm_logs.sh` script.
-   **`results/`**: Contains text files with the final evaluation metrics for each individual.

## Interpreting Core Log Files

### Main SLURM Output (`slurm-main-*.out`)

This is the primary log file for the entire genetic algorithm process (`run_improved.py`). It provides a high-level view of the evolution.

**Key Information to Look For:**
-   **Generation Status**: `GENERATION 0`, `GENERATION 1`, etc.
-   **Job Submission**: Lines like `running eval for gene 1 with job id 3403230` show that a child SLURM job has been submitted for evaluation.
-   **Fitness Updates**: The log shows the population's fitness scores at each generation.
    ```
    STATS:
    gen	nevals	avg     	std     	min     	max
    0  	8     	(0.4, 5.2e+05)	(0.0, 0.0)	(0.4, 5.2e+05)	(0.4, 5.2e+05)
    ```
-   **Fitness Tuple `(accuracy, parameters)`**: The fitness is a multi-objective value.
    -   The first value is **Test Accuracy** (higher is better).
    -   The second value is the **Number of Model Parameters** (lower is better).
    -   The weights `(1.0, -1.0)` in `src/cfg/constants.py` guide the algorithm to maximize accuracy and minimize parameters.
-   **"Delayed..." Messages**:
    ```
    Delayed checking for job 3403230's completion by 30 minutes
    ```
    This message is normal. The system polls for job completion in a loop. To avoid overwhelming the SLURM scheduler with frequent checks, it waits for a significant period (e.g., 30 minutes) between polling cycles. It does **not** mean the job itself is delayed.
-   **`GLOBAL_DATA_ANCESTRY`**:
    ```
    GLOBAL_DATA_ANCESTRY
    {
        "9": {
            "id": "9",
            "parent_id": "1",
            "operation": "mutation"
        }
    }
    ```
    This section logs the lineage of each new gene. It tracks which individuals were created through `mutation` or `crossover` and identifies their parent(s). This is essential for tracing the evolution of successful architectures.

### Evaluation Logs (`eval-*.out`, `eval-*.err`)

These logs capture the output from training and evaluating a single neural network architecture.

-   **`eval-*.out`**: Contains the standard output of the training script. Look for training progress, epoch results, and the final accuracy and parameter count.
-   **`eval-*.err`**: Contains error messages. It is the first place to look if an evaluation job fails. Common errors include `NameError` or `SyntaxError` if the LLM generated invalid code.

### LLM Operation Logs (`llm-*.out`, `llm-*.err`)

These logs show the interaction with the Large Language Model during mutation and crossover operations.

-   **`llm-*.out`**:
    -   **Prompt**: Shows the exact prompt sent to the LLM.
    -   **Generated Code**: Shows the raw Python code returned by the LLM.
    -   **Fallback Trigger**: Messages like `Fallback to parent code triggered` indicate that the LLM failed to produce valid code after multiple retries.
-   **`llm-*.err`**: Captures errors from the LLM operation script, such as API timeouts or connection issues.

### Results Files (`*_results.txt`)

Located in `runs/<run_id>/results/`, these files provide a concise summary of each individual's performance. Each file corresponds to one individual in the population and contains a single, comma-separated line with four values:

1.  **Test Accuracy**: The model's accuracy on the test dataset.
2.  **Total Parameters**: The total number of trainable parameters in the model.
3.  **Validation Accuracy**: The model's accuracy on the validation dataset.
4.  **Train Time (seconds)**: The total time taken to train the model.

## Common Issues and Debugging

### Identical Fitness Scores in a Generation

-   **Symptom**: You observe that most or all individuals in a generation have the exact same fitness values (e.g., `(0.40209, 518230)`).
-   **Cause**: This is a direct result of the **fallback mechanism**. If an LLM-driven mutation or crossover fails to produce valid, runnable code within its retry budget (`LLM_GENERATION_MAX_RETRIES`), the system reverts the individual's gene to its parent's code. This ensures the population size remains stable, but if the LLM consistently fails (due to timeouts, restrictive prompts, or other issues), the new generation will be filled with clones of the previous one.
-   **How to Verify**:
    1.  Check the `llm-*.out` logs for messages like `Fallback to parent code triggered`.
    2.  Check the `eval-*.err` logs for `NameError` or other fatal errors that would cause the evaluation to fail and trigger the fallback.

### Job Timeouts and Log Migration

-   **Symptom**: The main SLURM job (e.g., `slurm-main-3403227.out`) ends abruptly, often due to reaching its time limit (e.g., 16 hours). The `runs/<run_id>/logs` directory is empty or incomplete because the main script was terminated before it could collect the child job logs.
-   **Solution**: Use the `scripts/migrate_slurm_logs.sh` script. This script will find all `.out` and `.err` files in the top-level `slurm-results/` directory that belong to your run and move them to the correct `runs/<run_id>/logs` folder.
    ```bash
    # Example usage:
    bash scripts/migrate_slurm_logs.sh <run_id>
    # e.g., bash scripts/migrate_slurm_logs.sh auto_20251015_180811
    ```

### `NameError` in Evaluation Logs

-   **Symptom**: The `eval-*.err` file contains an error like `NameError: name 'SE_LN' is not defined`.
-   **Cause**: The LLM generated Python code for a neural network that uses a variable, function, or class that was not defined or imported. This is a common failure mode for code generation.
-   **Impact**: This failed evaluation will trigger the fallback mechanism, causing the individual to revert to its parent's code.

### Fallback Messages in LLM Logs

-   **Symptom**: The `llm-*.out` log contains the message `Fallback to parent code triggered`.
-   **Cause**: This indicates that the LLM operation (mutation or crossover) failed repeatedly. The script attempted to generate code up to `LLM_GENERATION_MAX_RETRIES` times, and each attempt resulted in an error (e.g., it produced unrunnable code, timed out, or returned an empty response).
-   **Impact**: The resulting individual is a clone of its parent, leading to the "Identical Fitness Scores" issue described above.
-   **How to Debug**:
    1.  Examine the prompts in the `llm-*.out` file to see if they are too restrictive or ambiguous.
    2.  Check the corresponding `llm-*.err` file for timeout errors. If timeouts are frequent, consider increasing the time allocation for LLM jobs in `src/cfg/constants.py`.
