Container instructions for running server inside Apptainer/Singularity/Docker

Quick options:

- If you have Apptainer on the cluster, build or pull a vllm-enabled image and place `vllm.sif` in the repository root.
  - Example (pull from Docker Hub):

    apptainer pull vllm.sif docker://vllm/vllm:latest

  - Or build from a local Docker image: scp the image or build a Singularity definition and run `apptainer build`.

- If you don't have Apptainer, you can build the image on a workstation with Docker and then convert or copy the `.sif` file to the cluster.

- Once `vllm.sif` is placed in the repository root, submit the new server with:

    sbatch server_container.sh

- After server submission, run the benchmark helper to submit dependent jobs (the helper will typically detect the server registry or you can rely on Slurm dependency flags):

    bash submit_rag_100_jobs.sh

Notes & troubleshooting:
- If the cluster disallows Docker or lacks Apptainer, ask your sysadmin to provide an Apptainer runtime or a prebuilt `.sif` in a shared location.
- The wrapper will fall back to a conda env named `llm-vllm` or to `./.venv` if containers are not available. For best results, supply a `vllm.sif`.
- If you want me to build a Singularity def and try building on this node, tell me and I'll produce a `Singularity.def` file; building may take time and require Apptainer.
