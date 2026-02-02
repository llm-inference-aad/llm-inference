# 00 - Introduction & Onboarding

This document provides a high-level overview of the LLMGE project for new team members.

## 1. Project Goal

The primary objective is to **optimize LLM inference** by applying genetic evolution (LLMGE) to network architectures. We aim to establish a performance baseline with the existing setup and then measure improvements from techniques like RAG and speculative decoding.

## 2. Core Assumptions

The evolutionary process is based on the following principles:

- **Genetic Algorithm (GA) manipulates network architecture**: The DEAP-driven loop mutates ExquisiteNetV2 variants, which are then evaluated on the CIFAR10 dataset.
- **"Code blocks" are the genes**: The network is partitioned into `# --OPTION--` segments. The GA selects and rewrites these blocks using an LLM.
- **Fitness drives evolution**: Fitness is a multi-objective tuple, typically `(test_accuracy, parameter_count)`. Invalid individuals are heavily penalized.
- **Selection favors high-performers**: SPEA2/NSGA-II selection biases toward Pareto-strong parents for crossover and mutation.
- **Mutation maintains diversity**: A high mutation probability combined with LLM-based rewriting ensures population diversity.

## 3. High-Level Architecture

The system consists of several key components:

| Component | Location | Purpose |
| --- | --- | --- |
| **GA Driver** | `run_improved.py` | Controls the main evolution loop, selection, variation, and fitness evaluation. |
| **LLM Operators** | `src/llm_mutation.py`, `src/llm_crossover.py` | Encapsulate prompt construction, LLM interaction, and code validation. |
| **Evaluation** | `sota/ExquisiteNetV2/train.py` | Trains a candidate network and records its performance metrics. |
| **LLM Server** | `server.py` | A local FastAPI server that hosts the Hugging Face model for inference. |
| **Configuration**| `src/cfg/constants.py` | Central hub for all hyperparameters, paths, and execution flags. |

## 4. Evolutionary Workflow

1.  **Initialization**: The system starts with a seed population based on the ExquisiteNet baseline.
2.  **Variation**: For each individual, mutation or crossover scripts are submitted to generate a new `network_{gene}.py` via LLM prompts.
3.  **Evaluation**: Each generated network is trained, and its fitness metrics are saved to a results file.
4.  **Selection**: The next generation's parents are selected using SPEA2/NSGA-II.
5.  **Loop**: The process repeats, updating fitness, tracking ancestry, and checkpointing state.
