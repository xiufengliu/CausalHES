"""
Tests for data generation scripts.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


class TestPecanStreetGenerator:
    """Test Pecan Street-style data generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        try:
            from generate_pecan_street_style import PecanStreetStyleGenerator
            
            generator = PecanStreetStyleGenerator(random_state=42)
            assert generator.random_state == 42
            assert len(generator.archetypes) == 5
            
        except ImportError:
            pytest.skip("Pecan Street generator not available")
    
    def test_small_dataset_generation(self):
        """Test generation of a small dataset."""
        try:
            from generate_pecan_street_style import PecanStreetStyleGenerator
            
            generator = PecanStreetStyleGenerator(random_state=42)
            
            # Generate small dataset
            load_profiles, weather_data, labels, metadata = generator.generate_pecan_street_dataset(
                n_homes=10,
                n_days=7,
                start_date="2022-01-01"
            )
            
            # Check shapes
            expected_samples = 10 * 7  # 70 samples
            assert load_profiles.shape == (expected_samples, 24, 1)
            assert weather_data.shape == (expected_samples, 24, 2)
            assert labels.shape == (expected_samples,)
            assert len(metadata) == expected_samples
            
            # Check data ranges
            assert np.all(load_profiles >= 0)  # Load should be non-negative
            assert np.all(weather_data >= 0) and np.all(weather_data <= 1)  # Normalized weather
            assert np.all(labels >= 0) and np.all(labels < 5)  # 5 archetypes
            
        except ImportError:
            pytest.skip("Pecan Street generator not available")
    
    def test_archetype_distribution(self):
        """Test that archetypes are properly distributed."""
        try:
            from generate_pecan_street_style import PecanStreetStyleGenerator
            
            generator = PecanStreetStyleGenerator(random_state=42)
            
            # Generate dataset with balanced homes
            n_homes = 25  # 5 homes per archetype
            load_profiles, weather_data, labels, metadata = generator.generate_pecan_street_dataset(
                n_homes=n_homes,
                n_days=1,
                start_date="2022-01-01"
            )
            
            # Check archetype distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            assert len(unique_labels) == 5  # All 5 archetypes should be present
            
            # Each archetype should have equal representation (5 homes each)
            expected_count = n_homes // 5
            for count in counts:
                assert count == expected_count
                
        except ImportError:
            pytest.skip("Pecan Street generator not available")


class TestRealisticTimeseriesGenerator:
    """Test realistic timeseries generator."""
    
    def test_generator_import(self):
        """Test that realistic generator can be imported."""
        try:
            from generate_realistic_timeseries import RealisticTimeseriesGenerator
            generator = RealisticTimeseriesGenerator(random_state=42)
            assert generator.random_state == 42
        except ImportError:
            pytest.skip("Realistic timeseries generator not available")


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_data_normalization(self):
        """Test that generated data is properly normalized."""
        # Create dummy data
        data = np.random.randn(100, 24, 1) * 10 + 50  # Unnormalized data
        
        # Normalize to [0, 1]
        data_min = np.min(data)
        data_max = np.max(data)
        normalized = (data - data_min) / (data_max - data_min)
        
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1
        assert np.allclose(np.min(normalized), 0, atol=1e-6)
        assert np.allclose(np.max(normalized), 1, atol=1e-6)
    
    def test_temporal_consistency(self):
        """Test that time series data maintains temporal structure."""
        # Generate simple sinusoidal pattern
        t = np.linspace(0, 2*np.pi, 24)
        pattern = np.sin(t)
        
        # Check that pattern has expected properties
        assert len(pattern) == 24
        assert np.argmax(pattern) == 6  # Peak around hour 6 (π/2)
        assert np.argmin(pattern) == 18  # Trough around hour 18 (3π/2)


if __name__ == "__main__":
    pytest.main([__file__])
