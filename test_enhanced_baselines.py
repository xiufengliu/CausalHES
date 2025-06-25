#!/usr/bin/env python3
"""
Test script for enhanced baseline experiments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        # Test sklearn
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        print("✓ Sklearn imports successful")
        
        # Test torch
        import torch
        print("✓ PyTorch imports successful")
        
        # Test project utils
        from src.utils.config import load_config
        from src.utils.logging import setup_logging, get_logger
        print("✓ Utils imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_processing():
    """Test data processing."""
    print("\nTesting data processing...")
    
    try:
        from src.data.irish_dataset_processor import IrishDatasetProcessor
        print("✓ Data processor import successful")
        
        # Check if processed data exists
        processed_dir = Path("data/processed_irish")
        if processed_dir.exists() and (processed_dir / "irish_dataset_processed.npz").exists():
            print("✓ Processed data found")
            
            # Load processed data
            data = np.load(processed_dir / "irish_dataset_processed.npz")
            print(f"✓ Data loaded: {list(data.keys())}")
            return True
        else:
            print("✗ Processed data not found")
            return False
            
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    try:
        from src.evaluation.metrics import calculate_clustering_metrics
        print("✓ Metrics import successful")
        
        # Test with dummy data
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        X = np.random.randn(6, 10)
        
        metrics = calculate_clustering_metrics(y_true, y_pred, X)
        print(f"✓ Metrics calculation successful: ACC={metrics['accuracy']:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

def run_simple_baseline():
    """Run a simple baseline experiment."""
    print("\nRunning simple baseline experiment...")
    
    try:
        # Load processed data
        processed_dir = Path("data/processed_irish")
        data = np.load(processed_dir / "irish_dataset_processed.npz")
        
        load_profiles = data['load_profiles'][:100]  # Use first 100 samples
        cluster_labels = data['cluster_labels'][:100]
        
        print(f"Data shape: {load_profiles.shape}")
        print(f"Labels shape: {cluster_labels.shape}")
        
        # Run K-means
        from sklearn.cluster import KMeans
        from src.evaluation.metrics import calculate_clustering_metrics
        
        X_flat = load_profiles.reshape(len(load_profiles), -1)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        y_pred = kmeans.fit_predict(X_flat)
        
        metrics = calculate_clustering_metrics(cluster_labels, y_pred, X_flat)
        
        print(f"✓ K-means baseline completed:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  NMI: {metrics['nmi']:.4f}")
        print(f"  ARI: {metrics['ari']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple baseline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("="*60)
    print("ENHANCED BASELINE EXPERIMENTS - TESTING")
    print("="*60)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_basic_imports()
    all_tests_passed &= test_data_processing()
    all_tests_passed &= test_evaluation_metrics()
    all_tests_passed &= run_simple_baseline()
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED - Ready to run enhanced baseline experiments!")
        print("\nTo run the full experiments, use:")
        print("python experiments/run_enhanced_baseline_experiments.py")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before running experiments")
    print("="*60)

if __name__ == "__main__":
    main()
