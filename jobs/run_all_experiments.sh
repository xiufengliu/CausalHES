#!/bin/bash
#BSUB -q gpua100
#BSUB -J causal_hes_all
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -o jobs/output_all_exp_%J.out
#BSUB -e jobs/error_all_exp_%J.err
#BSUB -B
#BSUB -N

# Load required modules
module load cuda/12.8.1
module load python3/3.13.2

# Change to project directory
cd /zhome/bb/9/101964/xiuli/CausalHES

# Activate virtual environment if you have one
# source venv/bin/activate

echo "Starting Complete CausalHES Experimental Suite..."
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create all necessary directories
mkdir -p experiments/results/causal_hes
mkdir -p experiments/results/irish_dataset
mkdir -p data/processed_irish

echo "=== STEP 1: GENERATING SYNTHETIC DATASET ==="
if [ ! -d "data/pecan_street_style" ]; then
    echo "Generating Pecan Street-style synthetic dataset..."
    python3 generate_pecan_street_style.py
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate synthetic dataset!"
        exit 1
    fi
    echo "Synthetic dataset generated successfully!"
else
    echo "Synthetic dataset already exists."
fi

echo ""
echo "=== STEP 2: PROCESSING IRISH DATASET ==="
if [ ! -f "data/processed_irish/irish_dataset_processed.npz" ]; then
    # Check if raw Irish data exists
    if [ ! -d "data/Irish" ] || [ ! -f "data/Irish/ElectricityConsumption.csv" ]; then
        echo "WARNING: Irish raw data not found. Skipping Irish experiments."
        echo "To include Irish experiments, ensure data/Irish/ contains:"
        echo "  - ElectricityConsumption.csv"
        echo "  - household_characteristics.csv"
        SKIP_IRISH=true
    else
        echo "Processing Irish dataset..."
        python3 process_irish_dataset.py
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to process Irish dataset!"
            SKIP_IRISH=true
        else
            echo "Irish dataset processed successfully!"
            SKIP_IRISH=false
        fi
    fi
else
    echo "Irish dataset already processed."
    SKIP_IRISH=false
fi

echo ""
echo "=== STEP 3: RUNNING SYNTHETIC DATASET EXPERIMENTS ==="
echo "Running comprehensive synthetic dataset experiments..."
python3 experiments/run_causal_hes_experiments.py

if [ $? -eq 0 ]; then
    echo "Synthetic dataset experiments completed successfully!"
else
    echo "ERROR: Synthetic dataset experiments failed!"
    exit 1
fi

if [ "$SKIP_IRISH" != "true" ]; then
    echo ""
    echo "=== STEP 4: RUNNING IRISH DATASET EXPERIMENTS ==="
    echo "Running comprehensive Irish dataset experiments..."
    python3 experiments/run_irish_dataset_experiments.py
    
    if [ $? -eq 0 ]; then
        echo "Irish dataset experiments completed successfully!"
    else
        echo "ERROR: Irish dataset experiments failed!"
        exit 1
    fi
else
    echo ""
    echo "=== STEP 4: SKIPPED IRISH DATASET EXPERIMENTS ==="
    echo "Irish dataset experiments skipped due to missing data."
fi

echo ""
echo "=== EXPERIMENTAL SUITE COMPLETED ==="
echo "Results summary:"
echo "Synthetic dataset results: experiments/results/causal_hes/"
ls -la experiments/results/causal_hes/ 2>/dev/null || echo "  No synthetic results found"

if [ "$SKIP_IRISH" != "true" ]; then
    echo "Irish dataset results: experiments/results/irish_dataset/"
    ls -la experiments/results/irish_dataset/ 2>/dev/null || echo "  No Irish results found"
fi

echo ""
echo "Job completed at: $(date)"
echo "All CausalHES experiments finished successfully!"
