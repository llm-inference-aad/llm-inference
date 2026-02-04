# GPU & Cluster Monitoring Guide

This guide provides commands to monitor GPU utilization, memory usage, and potential CPU offloading on the PACE cluster.

## 1. Monitoring Active Jobs (`nvidia-smi`)

To see real-time GPU stats for your running job, you need to `ssh` into the compute node where your job is running.

### Step 1: Find your Node
Run:
```bash
squeue -u $USER
```
Look for the **NODELIST** column (e.g., `atl1-1-01-005-13-0`).

### Step 2: SSH into the Node
```bash
ssh atl1-1-01-005-13-0
```

### Step 3: Run Nvidia SMI
Once logged in, run:
```bash
watch -n 1 nvidia-smi
```
- **Volatile GPU-Util**: GPU active processing % (should be >0% during inference).
- **Memory-Usage**: VRAM usage. If this hits the cap (e.g., 80GB), your model might crash or offload.
- **Pwr:Usage/Cap**: Power draw.

## 2. Checking for CPU Offloading

If GPU memory is full, PyTorch might offload tensors to System RAM (CPU). This causes **PCIe Bus utilization** to spike and GPU utilization to drop.

### Monitor PCIe Bandwidth (`nvidia-smi dmon`)
Run this on the node to verify data transfer rates:
```bash
nvidia-smi dmon -s pucvmet
```
- **rxpci/txpci**: Data received/transmitted over PCIe. High constant values during inference suggest offloading/thrashing.
- **sm**: Streaming Multiprocessor utilization (GPU core usage).

### Monitor System RAM (`htop`)
To see if system RAM is filling up (indicating offloaded weights):
```bash
htop
```
Check the **MEM** bar.

## 3. Slurm Job Efficiency (`seff`)

After a job finishes (or while running), check its efficiency summary:
```bash
seff <JOB_ID>
```
Example: `seff 4046292`
- **Memory Utilized**: Peak system memory used.
- **CPU Efficiency**: How much CPU was used vs requested.

## 4. Cluster Status (`sinfo`)

To check the state of the partition or verify node features:
```bash
sinfo -p ice-gpu -o "%20N %10c %10m %10f %10G"
```
Or to see free GPUs:
```bash
sinfo -p ice-gpu -t idle
```
