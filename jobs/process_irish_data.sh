#!/bin/bash
#BSUB -q gpua100
#BSUB -J irish_data_process
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"
#BSUB -W 1:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[ptile=4]"
#BSUB -o jobs/output_irish_process_%J.out
#BSUB -e jobs/error_irish_process_%J.err
#BSUB -B
#BSUB -N

# Load required modules
module load cuda/12.8.1
module load python3/3.13.2

cd /zhome/bb/9/101964/xiuli/CausalHES

echo "Starting Irish dataset processing..."
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

if [ ! -d "data/Irish" ]; then
    echo "ERROR: Irish data directory not found at data/Irish/"
    exit 1
fi

if [ ! -f "data/Irish/ElectricityConsumption.csv" ]; then
    echo "ERROR: ElectricityConsumption.csv not found!"
    exit 1
fi

if [ ! -f "data/Irish/household_characteristics.csv" ]; then
    echo "ERROR: household_characteristics.csv not found!"
    exit 1
fi

echo "Files found. Beginning processing..."
python3 process_irish_standalone.py

if [ $? -eq 0 ]; then
    echo "Processing completed successfully!"
    ls -la data/processed_irish/
else
    echo "Processing failed!"
    exit 1
fi

echo "Job finished at: $(date)"
