#!/usr/bin/env python3
"""
Verification script for PyTorch conversion.

This script verifies that all converted components work correctly.
Run this after installing PyTorch dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def verify_imports():
    """Verify all imports work correctly."""
    print("🔍 Verifying imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
    except ImportError:
        print("❌ PyTorch not available. Please install: pip install torch")
        return False
    
    try:
        from src.models.clustering_layers import ClusteringLayer, BayesianClusteringLayer, AdaptiveClusteringLayer
        print("✅ Clustering layers imported successfully")
    except ImportError as e:
        print(f"❌ Clustering layers import failed: {e}")
        return False
    
    try:
        from src.losses import ReconstructionLoss, CSSAELoss
        print("✅ Loss functions imported successfully")
    except ImportError as e:
        print(f"❌ Loss functions import failed: {e}")
        return False
    
    try:
        from src.clustering.deep_clustering import DeepEmbeddedClustering
        print("✅ Deep clustering imported successfully")
    except ImportError as e:
        print(f"❌ Deep clustering import failed: {e}")
        return False
    
    return True


def verify_functionality():
    """Verify basic functionality of converted components."""
    print("\n🧪 Verifying functionality...")
    
    import torch
    from src.models.clustering_layers import ClusteringLayer
    from src.losses import ReconstructionLoss
    
    # Test clustering layer
    try:
        layer = ClusteringLayer(n_clusters=5, embedding_dim=32)
        x = torch.randn(10, 32)
        output = layer(x)
        
        assert output.shape == (10, 5), f"Expected shape (10, 5), got {output.shape}"
        assert torch.allclose(torch.sum(output, dim=1), torch.ones(10), atol=1e-6), "Probabilities don't sum to 1"
        
        print(f"✅ ClusteringLayer: Input {x.shape} → Output {output.shape}")
    except Exception as e:
        print(f"❌ ClusteringLayer test failed: {e}")
        return False
    
    # Test loss function
    try:
        loss_fn = ReconstructionLoss()
        y_true = torch.randn(10, 24, 1)
        y_pred = torch.randn(10, 24, 1)
        loss = loss_fn(y_true, y_pred)
        
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.item() >= 0, "MSE loss should be non-negative"
        
        print(f"✅ ReconstructionLoss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ ReconstructionLoss test failed: {e}")
        return False
    
    return True


def verify_device_support():
    """Verify CUDA support if available."""
    print("\n🖥️  Verifying device support...")
    
    import torch
    
    print(f"✅ CPU support: Available")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA support: Available ({torch.cuda.device_count()} devices)")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name()}")
    else:
        print("ℹ️  CUDA support: Not available (CPU only)")
    
    return True


def verify_data_generation():
    """Verify data generation scripts work."""
    print("\n📊 Verifying data generation...")
    
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        print("✅ NumPy and Pandas available")
        
        # Test that data generation files exist
        pecan_file = project_root / "generate_pecan_street_style.py"
        realistic_file = project_root / "generate_realistic_timeseries.py"
        
        if pecan_file.exists():
            print("✅ Pecan Street generator file exists")
        else:
            print("❌ Pecan Street generator file missing")
            
        if realistic_file.exists():
            print("✅ Realistic timeseries generator file exists")
        else:
            print("❌ Realistic timeseries generator file missing")
            
    except ImportError as e:
        print(f"❌ Data generation verification failed: {e}")
        return False
    
    return True


def main():
    """Main verification function."""
    print("🔄 CausalHES PyTorch Conversion Verification")
    print("=" * 50)
    
    success = True
    
    # Run all verification steps
    success &= verify_imports()
    success &= verify_functionality()
    success &= verify_device_support()
    success &= verify_data_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All verifications passed! PyTorch conversion successful.")
        print("\n📋 Next steps:")
        print("   1. Complete CSSAE model implementation")
        print("   2. Finish training infrastructure")
        print("   3. Run comprehensive tests: python run_tests.py")
        print("   4. Generate data: python generate_pecan_street_style.py")
        print("   5. Run demo: python examples/causal_hes_demo.py")
    else:
        print("❌ Some verifications failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
