#!/usr/bin/env python3
"""
Process Irish Dataset for CausalHES Experiments.

This script processes the raw Irish household energy dataset and prepares it
for use with the CausalHES framework.

Usage:
    python process_irish_dataset.py

Author: CausalHES Team
Date: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data.irish_dataset_processor import IrishDatasetProcessor
from src.utils.logging import setup_logging, get_logger


def main():
    """Main function to process Irish dataset."""
    print("="*80)
    print("PROCESSING IRISH HOUSEHOLD ENERGY DATASET FOR CAUSALHES")
    print("="*80)
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("IrishDatasetProcessing")
    
    # Check if Irish data directory exists
    data_dir = Path("data/Irish")
    if not data_dir.exists():
        logger.error(f"Irish data directory not found: {data_dir}")
        logger.error("Please ensure the Irish dataset files are placed in data/Irish/")
        logger.error("Required files:")
        logger.error("  - ElectricityConsumption.csv")
        logger.error("  - household_characteristics.csv")
        return
    
    # Initialize processor
    logger.info("Initializing Irish dataset processor...")
    processor = IrishDatasetProcessor(
        data_dir=str(data_dir),
        random_state=42
    )
    
    # Process the dataset
    logger.info("Starting dataset processing...")
    try:
        processed_data = processor.process_irish_dataset(
            n_households=1000,
            n_clusters=4,
            normalize=True,
            save_processed=True,
            output_dir="data/processed_irish"
        )
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"Dataset Summary:")
        print(f"  Total households: {processed_data['metadata']['n_households']}")
        print(f"  Total daily profiles: {processed_data['metadata']['n_profiles']:,}")
        print(f"  Number of clusters: {processed_data['metadata']['n_clusters']}")
        print(f"  Date range: {processed_data['metadata']['date_range'][0]} to {processed_data['metadata']['date_range'][1]}")
        print(f"  Normalization: {processed_data['metadata']['normalization']}")
        
        print(f"\nData Shapes:")
        print(f"  Load profiles: {processed_data['load_profiles'].shape}")
        print(f"  Weather profiles: {processed_data['weather_profiles'].shape}")
        print(f"  Cluster labels: {processed_data['cluster_labels'].shape}")
        
        print(f"\nCluster Distribution:")
        cluster_counts = np.bincount(processed_data['cluster_labels'])
        for i, count in enumerate(cluster_counts):
            percentage = (count / len(processed_data['cluster_labels'])) * 100
            print(f"  Cluster {i}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\nProcessed data saved to: data/processed_irish/")
        print(f"Files created:")
        print(f"  - irish_dataset_processed.npz")
        print(f"  - household_characteristics.csv")
        print(f"  - profile_metadata.csv")
        print(f"  - processing_summary.json")
        
        print("\n" + "="*60)
        print("READY FOR CAUSALHES EXPERIMENTS!")
        print("="*60)
        print("You can now run:")
        print("  python experiments/run_irish_dataset_experiments.py")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"\nERROR: {e}")
        print("\nPlease check the log for more details.")
        return
    
    logger.info("Irish dataset processing completed successfully!")


if __name__ == "__main__":
    main()
