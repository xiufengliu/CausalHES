#!/bin/bash

# Navigate to your project directory
cd /zhome/bb/9/101964/xiuli/CausalHES

echo "Starting CausalHES Irish Dataset Experiments (Direct Run)..."
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi

# Check for processed Irish data
if [ ! -d "data/processed_irish" ]; then
    echo "Processing Irish dataset..."
    python3 process_irish_dataset.py
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to process Irish dataset!"
        exit 1
    fi
fi

echo "Processed data ready. Running experiments..."

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
