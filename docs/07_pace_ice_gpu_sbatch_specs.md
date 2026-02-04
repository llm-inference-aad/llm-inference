# PACE ICE Cluster: GPU Request Guide

Reference documentation for requesting GPU resources on the ICE cluster via Slurm.

## Requesting GPUs
Two approaches:
1. `GRES` (Per Node): `--gres=gpu:<type>:<number>`
2. `GPUS` (Per Job): `-G` or `--gpus=<type>:<total>`

**Minimum**: 1 GPU per node.

### A100 Nodes (Nvidia A100)
- **Specs**: 2 GPUs per node (AMD CPUs)
- **Memory**: 40GB or 80GB variants
- **Syntax**:
    - Any A100: `--gres=gpu:A100:1`
    - 40GB Specific: `--gres=gpu:1 -C A100-40GB`
    - 80GB Specific: `--gres=gpu:1 -C A100-80GB`

### H100 Nodes (Nvidia H100)
- **Specs**: 8 GPUs per node
- **Syntax**:
    - Generic: `--gres=gpu:H100:1`
    - With Constraint: `--gres=gpu:1 -C H100`
    - First Available (H100 or H200): `-C HX00`

### H200 Nodes (Nvidia H200)
- **Specs**: 8 GPUs per node
- **Syntax**:
    - Generic: `--gres=gpu:H200:1`
    - With Constraint: `--gres=gpu:1 -C H200`

### V100 Nodes (Nvidia Tesla V100)
- **Specs**: Max 4 per node
- **Syntax**:
    - Any V100: `--gres=gpu:V100:1`
    - 16GB: `--gres=gpu:1 -C V100-16GB`
    - 32GB: `--gres=gpu:1 -C V100-32GB`

### Other GPUs
- **A40**: Max 2 per node (`--gres=gpu:A40:1`)
- **RTX 6000**: Max 4 per node (`--gres=gpu:RTX_6000:1`)
- **L40S**: Max 8 per node (`--gres=gpu:L40S:1`)
- **MI210 (AMD)**: Max 2 per node (`--gres=gpu:MI210:1`)

### Advanced Constraints (AND / OR)
Node features can be combined to create flexible requests using specific operators.

- **AND (`&`)**: All features must be present.
    - Example: `-C "intel&gpu"`
- **OR (`|`)**: At least one feature must be present.
    - Example: `-C "intel|amd"` (Specific features)
    - Example: `-C "[rack1|rack2]"` (Matching OR - all nodes same)
- **Parentheses**: Grouping operations.
    - Example: `-C "usage&(A100-80GB|H100)"` requires "usage" AND either "A100-80GB" or "H100".
    - **Note**: Without parentheses, parsing is left-to-right (`foo&bar|baz` = (foo AND bar) OR baz).

### Precise GPU Allocation
To avoid `BadConstraints` errors when mixing generic requests with specific nodes:

- **Use `--gpus-per-node`**: Specifies exact GPU count per node.
    - Syntax: `--gpus-per-node=<type>:<number>`
    - Example: `#SBATCH --gpus-per-node=a100:2`
- **Use `--gpus-per-socket` / `--gpus-per-task`** for finer control.

**Recommended for 2-GPU A100 Jobs**:
```bash
#SBATCH -C "A100-80GB"
#SBATCH --gpus-per-node=2  # OR --gpus-per-node=a100:2
#SBATCH -p ice-gpu
```
