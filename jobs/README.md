# CausalHES GPU Cluster Job Scripts

This directory contains job scripts for running CausalHES experiments on the GPU cluster using LSF (Load Sharing Facility).

## Available Job Scripts

### 1. `process_irish_data.sh`
**Purpose**: Process the raw Irish household energy dataset
**Resources**: 1 GPU, 4 cores, 8GB RAM, 1 hour
**Prerequisites**: Raw Irish data files in `data/Irish/`

```bash
bsub < jobs/process_irish_data.sh
```

### 2. `run_irish_experiments.sh`
**Purpose**: Run complete CausalHES experiments on Irish dataset
**Resources**: 1 GPU, 8 cores, 16GB RAM, 6 hours
**Prerequisites**: Processed Irish data (run `process_irish_data.sh` first)

```bash
bsub < jobs/run_irish_experiments.sh
```

### 3. `run_synthetic_experiments.sh`
**Purpose**: Run complete CausalHES experiments on synthetic dataset
**Resources**: 1 GPU, 8 cores, 12GB RAM, 4 hours
**Prerequisites**: None (generates synthetic data automatically)

```bash
bsub < jobs/run_synthetic_experiments.sh
```

### 4. `run_all_experiments.sh`
**Purpose**: Run complete experimental suite (both datasets)
**Resources**: 1 GPU, 8 cores, 20GB RAM, 8 hours
**Prerequisites**: None (handles all data generation/processing)

```bash
bsub < jobs/run_all_experiments.sh
```

## Recommended Workflow

### Option 1: Run Everything at Once
```bash
# Submit the comprehensive job that handles everything
bsub < jobs/run_all_experiments.sh
```

### Option 2: Step-by-Step Execution
```bash
# Step 1: Process Irish data (if you have it)
bsub < jobs/process_irish_data.sh

# Step 2: Run synthetic experiments
bsub < jobs/run_synthetic_experiments.sh

# Step 3: Run Irish experiments (after Step 1 completes)
bsub < jobs/run_irish_experiments.sh
```

## Job Monitoring

### Check job status:
```bash
bjobs                    # Show all your jobs
bjobs -l <job_id>       # Detailed info for specific job
```

### View job output:
```bash
# Real-time monitoring
tail -f jobs/output_*_<job_id>.out

# View errors
tail -f jobs/error_*_<job_id>.err
```

### Cancel a job:
```bash
bkill <job_id>
```

## Expected Outputs

### After `process_irish_data.sh`:
```
data/processed_irish/
├── irish_dataset_processed.npz
├── household_characteristics.csv
├── profile_metadata.csv
└── processing_summary.json
```

### After `run_synthetic_experiments.sh`:
```
experiments/results/causal_hes/
├── experiment_results.json
├── clustering_metrics.csv
├── source_separation_analysis.json
└── training_history.json
```

### After `run_irish_experiments.sh`:
```
experiments/results/irish_dataset/
├── irish_dataset_results.json
├── irish_dataset_summary.csv
└── comprehensive_analysis.json
```

## Resource Requirements

| Job Script | GPU | Cores | RAM | Time | Purpose |
|------------|-----|-------|-----|------|---------|
| `process_irish_data.sh` | 1 | 4 | 8GB | 1h | Data processing |
| `run_irish_experiments.sh` | 1 | 8 | 16GB | 6h | Irish experiments |
| `run_synthetic_experiments.sh` | 1 | 8 | 12GB | 4h | Synthetic experiments |
| `run_all_experiments.sh` | 1 | 8 | 20GB | 8h | Complete suite |

## Troubleshooting

### Common Issues:

1. **Irish data not found**:
   - Ensure `data/Irish/ElectricityConsumption.csv` exists
   - Ensure `data/Irish/household_characteristics.csv` exists

2. **GPU memory issues**:
   - Jobs include memory management settings
   - Reduce batch size in config files if needed

3. **Job fails to start**:
   - Check queue availability: `bqueues`
   - Verify resource requirements are reasonable

4. **Python module errors**:
   - Ensure all dependencies are installed
   - Check if virtual environment activation is needed

### Debugging:
```bash
# Check job details
bjobs -l <job_id>

# View full output
cat jobs/output_*_<job_id>.out

# View errors
cat jobs/error_*_<job_id>.err

# Check GPU usage during job
ssh <compute_node>
nvidia-smi
```

## Notes

- All jobs use the `gpua100` queue for A100 GPUs
- Jobs include email notifications (`-B -N` flags)
- Output files are saved in the `jobs/` directory
- Jobs automatically load required modules (CUDA 12.8.1, Python 3.11.7)
- Memory and time limits are set conservatively but can be adjusted

## Customization

To modify resource requirements, edit the `#BSUB` directives:
- `-W`: Wall time limit (format: hours:minutes)
- `-R "rusage[mem=XGB]"`: Memory requirement
- `-n`: Number of cores
- `-gpu "num=X"`: Number of GPUs

Example:
```bash
#BSUB -W 10:00          # 10 hours
#BSUB -R "rusage[mem=32GB]"  # 32GB RAM
#BSUB -n 16             # 16 cores
```
