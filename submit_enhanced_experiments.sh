#!/bin/sh
#BSUB -q hpc                            # CPU queue name (changed from gpuv100)
#BSUB -J enhanced_baseline_experiments  # Job name
#BSUB -n 16                             # Number of CPU cores (increased for CPU-only processing)
#BSUB -R "rusage[mem=4GB]"              # Memory per CPU core
#BSUB -R "span[hosts=1]"                # Single host (SMP job)
#BSUB -W 6:00                           # Max wall time (6 hours for CPU-based experiments)
#BSUB -o enhanced_baseline_%J.out       # Output file
#BSUB -e enhanced_baseline_%J.err       # Error file

# Load required modules (removed CUDA since not needed)
module load python3/3.13.2
module load pandas/2.2.3-python-3.13.2

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
echo "CPU Info:"
lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core)"
echo "Memory Info:"
free -h
echo "======================================"

# Run the enhanced baseline experiments
python3 experiments/run_enhanced_baseline_experiments.py

echo "======================================"
echo "Enhanced Baseline Experiments Complete"
echo "Date: $(date)"
echo "======================================"
