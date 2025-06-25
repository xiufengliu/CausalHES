#!/bin/sh
#BSUB -q gpuv100                        # GPU queue name
#BSUB -J enhanced_baseline_experiments  # Job name
#BSUB -n 8                              # Number of CPU cores (more for data processing)
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU
#BSUB -R "rusage[mem=8GB]"              # Memory per CPU core (increased for large dataset)
#BSUB -R "span[hosts=1]"                # Single host (SMP job)
#BSUB -W 4:00                           # Max wall time (4 hours for comprehensive experiments)
#BSUB -o enhanced_baseline_%J.out       # Output file
#BSUB -e enhanced_baseline_%J.err       # Error file

# Load required modules
module load cuda/12.8.1
module load python3/3.13.2
module load pandas/2.2.3-python-3.13.2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd /zhome/bb/9/101964/xiuli/CausalHES

# Print job information
echo "======================================"
echo "Enhanced Baseline Experiments Started"
echo "======================================"
echo "Job ID: $LSB_JOBID"
echo "Host: $HOSTNAME"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "GPU Info:"
nvidia-smi
echo "======================================"

# Run the enhanced baseline experiments
python3 experiments/run_enhanced_baseline_experiments.py

echo "======================================"
echo "Enhanced Baseline Experiments Complete"
echo "Date: $(date)"
echo "======================================"
