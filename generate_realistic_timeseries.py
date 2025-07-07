"""
Realistic timeseries generator for CausalHES framework.

This module generates realistic household energy consumption timeseries
with various patterns and characteristics.
"""

import numpy as np
from typing import Tuple, Dict, Any


class RealisticTimeseriesGenerator:
    """
    Generator for realistic household energy consumption timeseries.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the generator.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_realistic_timeseries(self, n_samples: int = 100, 
                                    n_timesteps: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic household energy timeseries.
        
        Args:
            n_samples: Number of timeseries to generate
            n_timesteps: Number of timesteps per series
            
        Returns:
            Tuple of (timeseries_data, labels)
        """
        timeseries = []
        labels = []
        
        for i in range(n_samples):
            # Generate different patterns
            pattern_type = i % 4
            
            if pattern_type == 0:
                ts = self._generate_morning_pattern(n_timesteps)
            elif pattern_type == 1:
                ts = self._generate_evening_pattern(n_timesteps)
            elif pattern_type == 2:
                ts = self._generate_flat_pattern(n_timesteps)
            else:
                ts = self._generate_night_pattern(n_timesteps)
                
            timeseries.append(ts)
            labels.append(pattern_type)
            
        return np.array(timeseries), np.array(labels)
        
    def _generate_morning_pattern(self, n_timesteps: int) -> np.ndarray:
        """Generate morning peak pattern."""
        t = np.arange(n_timesteps)
        pattern = 0.3 + 0.7 * np.exp(-((t - 7)**2) / 8)
        return pattern + 0.1 * np.random.randn(n_timesteps)
        
    def _generate_evening_pattern(self, n_timesteps: int) -> np.ndarray:
        """Generate evening peak pattern."""
        t = np.arange(n_timesteps)
        pattern = 0.3 + 0.7 * np.exp(-((t - 19)**2) / 8)
        return pattern + 0.1 * np.random.randn(n_timesteps)
        
    def _generate_flat_pattern(self, n_timesteps: int) -> np.ndarray:
        """Generate flat consumption pattern."""
        return 0.5 + 0.1 * np.random.randn(n_timesteps)
        
    def _generate_night_pattern(self, n_timesteps: int) -> np.ndarray:
        """Generate night shift pattern."""
        t = np.arange(n_timesteps)
        pattern = 0.3 + 0.6 * np.exp(-((t - 2)**2) / 12)
        return pattern + 0.1 * np.random.randn(n_timesteps)


if __name__ == "__main__":
    generator = RealisticTimeseriesGenerator(random_state=42)
    data, labels = generator.generate_realistic_timeseries(n_samples=100)
    print(f"Generated {data.shape[0]} timeseries with {data.shape[1]} timesteps each")
    print(f"Labels: {np.unique(labels)}")