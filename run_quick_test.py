#!/usr/bin/env python3
"""
Quick test script to verify the enhanced baselines setup.
This runs a minimal version for testing before the full experiment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Core imports
        from src.clustering.weather_normalization import LinearWeatherNormalization
        from src.clustering.vae_baselines import BetaVAEClustering
        from src.clustering.multimodal_baselines import AttentionFusedDEC
        from src.clustering.causal_baselines import DoublyRobustClustering
        from src.clustering.traditional import SAXKMeans
        from src.evaluation.visualization import ClusterVisualizer
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation for baselines."""
    print("Testing data generation...")
    
    try:
        # Generate synthetic data
        n_samples = 50
        n_timesteps = 24
        n_clusters = 3
        
        # Load data
        X_load = np.random.randn(n_samples, n_timesteps, 1)
        X_weather = np.random.randn(n_samples, n_timesteps, 1)
        y_true = np.random.randint(0, n_clusters, n_samples)
        
        print(f"✓ Generated data: {X_load.shape}, {X_weather.shape}, {y_true.shape}")
        return X_load, X_weather, y_true
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        return None, None, None

def test_baseline_method(name, method_class, X_load, X_weather, y_true, **kwargs):
    """Test a single baseline method."""
    print(f"Testing {name}...")
    
    try:
        # Initialize method
        method = method_class(**kwargs)
        
        # Test fit and predict
        if hasattr(method, 'fit') and 'X_secondary' in method.fit.__code__.co_varnames:
            # Multi-modal method
            method.fit(X_load, X_secondary=X_weather)
            y_pred = method.predict(X_load, X_secondary=X_weather)
        else:
            # Single-modal method
            method.fit(X_load)
            y_pred = method.predict(X_load)
            
        print(f"✓ {name} completed - Predictions: {y_pred.shape}")
        return True
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        return False

def test_table_generation():
    """Test LaTeX table generation."""
    print("Testing LaTeX table generation...")
    
    try:
        import pandas as pd
        
        # Create test data
        test_data = {
            'Method': ['Method A', 'Method B', 'CausalHES'],
            'ACC': ['0.75$^\\dagger$', '0.82$^\\dagger$', '\\textbf{0.88}'],
            'NMI': ['0.65$^\\dagger$', '0.74$^\\dagger$', '\\textbf{0.81}'],
            'ARI': ['0.60$^\\dagger$', '0.70$^\\dagger$', '\\textbf{0.79}']
        }
        
        df = pd.DataFrame(test_data)
        
        # Generate simple LaTeX table
        latex_table = r"""\begin{table}[t]
\caption{Test Table}
\begin{tabular}{lccc}
\toprule
Method & ACC & NMI & ARI \\
\midrule
"""
        
        for _, row in df.iterrows():
            latex_table += f"{row['Method']} & {row['ACC']} & {row['NMI']} & {row['ARI']} \\\\\n"
            
        latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        # Save test table
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)
        
        with open(test_dir / "test_table.tex", 'w') as f:
            f.write(latex_table)
        
        # Verify file was created
        if (test_dir / "test_table.tex").exists():
            print("✓ LaTeX table generation successful")
            return True
        else:
            print("✗ LaTeX table file not created")
            return False
            
    except Exception as e:
        print(f"✗ LaTeX table generation failed: {e}")
        return False

def run_quick_tests():
    """Run all quick tests."""
    print("="*60)
    print("CAUSAL HES ENHANCED BASELINES - QUICK TEST")
    print("="*60)
    
    # Test 1: Imports
    imports_ok = test_imports()
    
    # Test 2: Data generation
    X_load, X_weather, y_true = test_data_generation()
    if X_load is None:
        print("Cannot proceed with method tests - data generation failed")
        return False
        
    # Test 3: Baseline methods
    n_clusters = 3
    
    methods_to_test = [
        ("Linear Weather Normalization", "LinearWeatherNormalization", {
            'n_clusters': n_clusters,
            'base_clusterer': None,  # Will be handled in test
            'regression_type': 'linear'
        }),
        ("SAX K-means", "SAXKMeans", {
            'n_clusters': n_clusters,
            'word_size': 12,
            'alphabet_size': 8
        }),
        ("β-VAE Clustering", "BetaVAEClustering", {
            'n_clusters': n_clusters,
            'embedding_dim': 16,
            'beta': 2.0
        })
    ]
    
    method_results = []
    
    for name, class_name, kwargs in methods_to_test:
        try:
            if class_name == "LinearWeatherNormalization":
                from src.clustering.weather_normalization import LinearWeatherNormalization
                from sklearn.cluster import KMeans
                kwargs['base_clusterer'] = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                method_class = LinearWeatherNormalization
            elif class_name == "SAXKMeans":
                from src.clustering.traditional import SAXKMeans
                method_class = SAXKMeans
            elif class_name == "BetaVAEClustering":
                from src.clustering.vae_baselines import BetaVAEClustering
                method_class = BetaVAEClustering
                
            result = test_baseline_method(name, method_class, X_load, X_weather, y_true, **kwargs)
            method_results.append((name, result))
            
        except ImportError as e:
            print(f"✗ {name} - Import failed: {e}")
            method_results.append((name, False))
    
    # Test 4: LaTeX table generation
    table_ok = test_table_generation()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Data generation: {'✓ PASS' if X_load is not None else '✗ FAIL'}")
    
    print("Baseline methods:")
    for name, result in method_results:
        print(f"  {name}: {'✓ PASS' if result else '✗ FAIL'}")
        
    print(f"LaTeX table generation: {'✓ PASS' if table_ok else '✗ FAIL'}")
    
    # Overall result
    all_passed = (imports_ok and X_load is not None and 
                 all(result for _, result in method_results) and table_ok)
    
    print("\n" + "-"*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready to run full experiments!")
        print("\nNext step: Run the full experiments with:")
        print("python experiments/run_enhanced_baseline_experiments.py")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
        print("\nPlease fix the issues before running full experiments.")
    print("-"*60)
    
    return all_passed

if __name__ == "__main__":
    run_quick_tests()