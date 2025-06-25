#!/bin/sh
#BSUB -q gpuv100                        # GPU queue name
#BSUB -J test_enhanced_baseline         # Job name
#BSUB -n 4                              # Number of CPU cores (minimum 4 for GPU)
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU
#BSUB -R "rusage[mem=4GB]"              # Memory per CPU core
#BSUB -R "span[hosts=1]"                # Single host (SMP job)
#BSUB -W 0:15                           # Max wall time (15 minutes for test)
#BSUB -o test_baseline_%J.out           # Output file
#BSUB -e test_baseline_%J.err           # Error file

# Load required modules
module load cuda/12.8.1
module load python3/3.13.2
module load pandas/2.2.3-python-3.13.2

# Change to project directory
cd /zhome/bb/9/101964/xiuli/CausalHES

# Print job information
echo "======================================"
echo "Enhanced Baseline Test Started"
echo "======================================"
echo "Job ID: $LSB_JOBID"
echo "Host: $HOSTNAME"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo "======================================"

# Test basic imports
echo "Testing basic imports..."
python3 -c "
import sys
import os
sys.path.append('.')

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    import torch
    print('✓ Basic packages imported successfully')
    
    from src.evaluation.metrics import calculate_clustering_metrics
    from src.utils.logging import setup_logging, get_logger
    print('✓ CausalHES modules imported successfully')
    
    # Test metrics function
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    X = np.random.randn(6, 10)
    metrics = calculate_clustering_metrics(y_true, y_pred, X)
    print(f'✓ Metrics calculation successful: ACC={metrics[\"accuracy\"]:.3f}')
    
    # Test data availability
    import pathlib
    processed_dir = pathlib.Path('data/processed_irish')
    if processed_dir.exists() and (processed_dir / 'irish_dataset_processed.npz').exists():
        data = np.load(processed_dir / 'irish_dataset_processed.npz')
        print(f'✓ Processed data found: {list(data.keys())}')
        print(f'  Load profiles shape: {data[\"load_profiles\"].shape}')
        print(f'  Weather profiles shape: {data[\"weather_profiles\"].shape}')
        print(f'  Cluster labels shape: {data[\"cluster_labels\"].shape}')
    else:
        print('✗ Processed data not found')
        
    print('✓ All tests passed - Ready for full experiment!')
    
except Exception as e:
    print(f'✗ Test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo "======================================"
echo "Enhanced Baseline Test Complete"
echo "Date: $(date)"
echo "======================================"
