#!/usr/bin/env python3
"""
Test the optimized Irish dataset processing with a small subset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import time

from src.data.irish_dataset_processor import IrishDatasetProcessor
from src.utils.logging import setup_logging, get_logger


def test_optimized_processing():
    """Test the optimized processing with a small number of households."""
    print("="*80)
    print("TESTING OPTIMIZED IRISH DATASET PROCESSING")
    print("="*80)
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("TestOptimizedProcessing")
    
    # Check if Irish data directory exists
    data_dir = Path("data/Irish")
    if not data_dir.exists():
        logger.error(f"Irish data directory not found: {data_dir}")
        return False
    
    # Initialize processor
    logger.info("Initializing Irish dataset processor...")
    processor = IrishDatasetProcessor(
        data_dir=str(data_dir),
        random_state=42
    )
    
    # Test with small number of households first
    test_households = [50, 100]
    
    for n_households in test_households:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING WITH {n_households} HOUSEHOLDS")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Process the dataset
            processed_data = processor.process_irish_dataset(
                n_households=n_households,
                n_clusters=4,
                normalize=True,
                save_processed=False,  # Don't save during testing
                output_dir="data/processed_irish_test"
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Print results
            print(f"\n{'='*60}")
            print(f"PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"{'='*60}")
            
            print(f"Dataset Summary:")
            print(f"  Total households: {processed_data['metadata']['n_households']}")
            print(f"  Total daily profiles: {processed_data['metadata']['n_profiles']:,}")
            print(f"  Number of clusters: {processed_data['metadata']['n_clusters']}")
            print(f"  Date range: {processed_data['metadata']['date_range'][0]} to {processed_data['metadata']['date_range'][1]}")
            
            print(f"\nData Shapes:")
            print(f"  Load profiles: {processed_data['load_profiles'].shape}")
            print(f"  Weather profiles: {processed_data['weather_profiles'].shape}")
            print(f"  Cluster labels: {processed_data['cluster_labels'].shape}")
            
            print(f"\nCluster Distribution:")
            cluster_counts = np.bincount(processed_data['cluster_labels'])
            for i, count in enumerate(cluster_counts):
                percentage = (count / len(processed_data['cluster_labels'])) * 100
                print(f"  Cluster {i}: {count:,} samples ({percentage:.1f}%)")
            
            # Estimate time for 1000 households
            time_per_household = processing_time / n_households
            estimated_time_1000 = time_per_household * 1000
            print(f"\nEstimated time for 1000 households: {estimated_time_1000:.2f} seconds ({estimated_time_1000/60:.1f} minutes)")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            print(f"\nERROR: {e}")
            return False
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED! Ready for full 1000 household processing.")
    print(f"{'='*80}")
    
    return True


if __name__ == "__main__":
    success = test_optimized_processing()
    if success:
        print("\nYou can now run the full experiment with:")
        print("  bsub < submit_enhanced_experiments.sh")
    else:
        print("\nPlease fix the issues before running the full experiment.")
