#!/usr/bin/env python3
"""
Enhanced Baseline Experiments - Safe Import Version
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Test basic imports first
def test_basic_imports():
    print("Testing basic imports...")
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        print("✓ Sklearn imports successful")
        
        from src.utils.config import load_config
        from src.utils.logging import setup_logging, get_logger
        print("✓ Utils imports successful")
        
        from src.evaluation.metrics import calculate_clustering_metrics
        print("✓ Metrics import successful")
        
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_data_loading():
    print("Testing data loading...")
    try:
        # Load processed data directly
        processed_dir = Path("data/processed_irish")
        data = np.load(processed_dir / "irish_dataset_processed.npz")
        
        load_profiles = data['load_profiles'][:100]  # Small subset
        cluster_labels = data['cluster_labels'][:100]
        
        print(f"✓ Data loaded: {load_profiles.shape}")
        return load_profiles, cluster_labels, True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None, None, False

def test_clustering_pipeline():
    print("Testing clustering pipeline...")
    
    # Get data
    load_profiles, cluster_labels, success = test_data_loading()
    if not success:
        return False
    
    try:
        from sklearn.cluster import KMeans
        from src.evaluation.metrics import calculate_clustering_metrics
        
        # Simple K-means test
        X_flat = load_profiles.reshape(len(load_profiles), -1)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        y_pred = kmeans.fit_predict(X_flat)
        
        # Test metrics
        metrics = calculate_clustering_metrics(cluster_labels, y_pred, X_flat)
        print(f"✓ Clustering test successful: ACC={metrics['accuracy']:.3f}")
        return True
    except Exception as e:
        print(f"✗ Clustering pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def safe_import_clustering_modules():
    """Safely import clustering modules one by one."""
    results = {}
    
    # Test traditional clustering
    try:
        from src.clustering.traditional import SAXKMeans, TwoStageKMeans
        results['traditional'] = True
        print("✓ Traditional clustering imports successful")
    except Exception as e:
        print(f"✗ Traditional clustering failed: {e}")
        results['traditional'] = False
    
    # Test weather normalization
    try:
        from src.clustering.weather_normalization import LinearWeatherNormalization, CDDHDDWeatherNormalization
        results['weather_normalization'] = True
        print("✓ Weather normalization imports successful")
    except Exception as e:
        print(f"✗ Weather normalization failed: {e}")
        results['weather_normalization'] = False
    
    # Test VAE baselines
    try:
        from src.clustering.vae_baselines import BetaVAEClustering, FactorVAEClustering, MultiModalVAEClustering
        results['vae_baselines'] = True
        print("✓ VAE baselines imports successful")
    except Exception as e:
        print(f"✗ VAE baselines failed: {e}")
        results['vae_baselines'] = False
    
    # Test multimodal baselines
    try:
        from src.clustering.multimodal_baselines import AttentionFusedDEC, ContrastiveMVClustering, GraphMultiModalClustering
        results['multimodal_baselines'] = True
        print("✓ Multimodal baselines imports successful")
    except Exception as e:
        print(f"✗ Multimodal baselines failed: {e}")
        results['multimodal_baselines'] = False
    
    # Test causal baselines
    try:
        from src.clustering.causal_baselines import DoublyRobustClustering, InstrumentalVariableClustering, DomainAdaptationClustering
        results['causal_baselines'] = True
        print("✓ Causal baselines imports successful")
    except Exception as e:
        print(f"✗ Causal baselines failed: {e}")
        results['causal_baselines'] = False
    
    return results

def main():
    print("="*60)
    print("ENHANCED BASELINE EXPERIMENTS - DIAGNOSTIC")
    print("="*60)
    
    # Step 1: Test basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed. Cannot proceed.")
        return
    
    # Step 2: Test clustering pipeline
    if not test_clustering_pipeline():
        print("❌ Clustering pipeline failed. Cannot proceed.")
        return
    
    # Step 3: Test all clustering modules
    import_results = safe_import_clustering_modules()
    
    print("\n" + "="*60)
    print("IMPORT RESULTS SUMMARY:")
    print("="*60)
    for module, success in import_results.items():
        status = "✓" if success else "✗"
        print(f"{status} {module}: {'SUCCESS' if success else 'FAILED'}")
    
    successful_modules = sum(import_results.values())
    total_modules = len(import_results)
    
    print(f"\nOverall: {successful_modules}/{total_modules} modules imported successfully")
    
    if successful_modules >= 3:  # At least 3 modules working
        print("\n✅ Ready to run enhanced baseline experiments!")
        print("You can now run: python experiments/run_enhanced_baseline_experiments.py")
    else:
        print("\n❌ Too many import failures. Need to fix issues first.")
    
    print("="*60)

if __name__ == "__main__":
    main()
