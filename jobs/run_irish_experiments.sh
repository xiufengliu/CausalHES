#!/bin/bash
#BSUB -q gpua100
#BSUB -J irish_experiments
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"
#BSUB -W 6:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o jobs/output_irish_exp_%J.out
#BSUB -e jobs/error_irish_exp_%J.err
#BSUB -B
#BSUB -N
# Load required modules
module load cuda/12.8.1
module load python3/3.13.2

# Optional: Activate virtual environment if you have one
# source ~/venv_ds/bin/activate

# Navigate to your project directory
cd /zhome/bb/9/101964/xiuli/CausalHES

# Job metadata
echo "Starting CausalHES Irish Dataset Experiments..."
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi

# Check for processed Irish data
if [ ! -d "data/processed_irish" ]; then
    echo "ERROR: Directory 'data/processed_irish' not found."
    echo "Please preprocess the Irish data first."
    exit 1
fi

if [ ! -f "data/processed_irish/irish_dataset_processed.npz" ]; then
    echo "ERROR: Dataset file not found!"
    echo "Please run the data processing job first."
    exit 1
fi

echo "Processed data found. Running experiments..."

# Create results directory
mkdir -p experiments/results/irish_dataset

# Set PyTorch GPU memory handling preferences
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run experiment
python3 experiments/run_irish_dataset_experiments.py

# Check job result
if [ $? -eq 0 ]; then
    echo "Irish dataset experiments completed successfully!"
    echo "Results in: experiments/results/irish_dataset/"
    ls -la experiments/results/irish_dataset/

    if [ -f "experiments/results/irish_dataset/irish_dataset_summary.csv" ]; then
        echo ""
        echo "=== EXPERIMENT RESULTS SUMMARY ==="
        cat experiments/results/irish_dataset/irish_dataset_summary.csv
        echo "=================================="
    fi
else
    echo "ERROR: Experiment execution failed!"
    exit 1
fi

echo "Job completed at: $(date)"
