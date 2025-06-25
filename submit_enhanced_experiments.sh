#!/bin/sh
#BSUB -q hpc                            # CPU queue name (changed from gpuv100)
#BSUB -J enhanced_baseline_experiments  # Job name
#BSUB -n 24                             # Number of CPU cores (increased for 1000 households)
#BSUB -R "rusage[mem=8GB]"              # Memory per CPU core (increased for larger dataset)
#BSUB -R "span[hosts=1]"                # Single host (SMP job)
#BSUB -W 8:00                           # Max wall time (8 hours for 1000 households)
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

# Check if required data exists
if [ ! -d "data/Irish" ]; then
    echo "ERROR: Irish data directory not found!"
    echo "Please ensure data/Irish/ contains the required files."
    exit 1
fi

if [ ! -f "data/Irish/ElectricityConsumption.csv" ]; then
    echo "ERROR: ElectricityConsumption.csv not found!"
    exit 1
fi

# Run the enhanced baseline experiments
echo "Starting enhanced baseline experiments..."
python3 experiments/run_enhanced_baseline_experiments.py

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "======================================"
    echo "Enhanced Baseline Experiments Complete"
    echo "Date: $(date)"
    echo "======================================"

    # Show results summary if available
    if [ -f "experiments/results/enhanced_baselines/enhanced_baseline_summary.csv" ]; then
        echo ""
        echo "=== RESULTS SUMMARY ==="
        head -10 experiments/results/enhanced_baselines/enhanced_baseline_summary.csv
        echo "======================="
    fi
else
    echo "======================================"
    echo "Enhanced Baseline Experiments FAILED"
    echo "Date: $(date)"
    echo "======================================"
    exit 1
fi
