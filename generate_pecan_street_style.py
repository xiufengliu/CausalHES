"""
Pecan Street-style data generator for testing CausalHES framework.

This module generates synthetic household energy data that mimics the structure
and characteristics of the Pecan Street dataset.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from datetime import datetime, timedelta


class PecanStreetStyleGenerator:
    """
    Generator for Pecan Street-style household energy data.
    
    Creates synthetic data with realistic household energy consumption patterns
    for testing and development of the CausalHES framework.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the generator.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define household archetypes
        self.archetypes = {
            0: "high_morning_peak",
            1: "high_evening_peak", 
            2: "flat_consumption",
            3: "night_shift",
            4: "weekend_pattern"
        }
        
    def generate_pecan_street_dataset(self, n_homes: int = 100, n_days: int = 30,
                                    start_date: str = "2022-01-01") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate a complete Pecan Street-style dataset.
        
        Args:
            n_homes: Number of homes to generate
            n_days: Number of days of data per home
            start_date: Start date for the dataset
            
        Returns:
            Tuple of (load_profiles, weather_data, labels, metadata)
        """
        total_samples = n_homes * n_days
        
        # Generate load profiles
        load_profiles = []
        weather_data = []
        labels = []
        metadata = []
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        for home_id in range(n_homes):
            archetype = home_id % len(self.archetypes)
            
            for day in range(n_days):
                current_date = start_dt + timedelta(days=day)
                
                # Generate load profile for this home and day
                load_profile = self._generate_load_profile(archetype, current_date)
                
                # Generate corresponding weather data
                weather_profile = self._generate_weather_profile(current_date)
                
                load_profiles.append(load_profile.reshape(24, 1))
                weather_data.append(weather_profile)
                labels.append(archetype)
                
                metadata.append({
                    'home_id': home_id,
                    'date': current_date.strftime("%Y-%m-%d"),
                    'archetype': self.archetypes[archetype],
                    'day_of_week': current_date.weekday(),
                    'is_weekend': current_date.weekday() >= 5
                })
        
        load_profiles = np.array(load_profiles)
        weather_data = np.array(weather_data)
        labels = np.array(labels)
        
        return load_profiles, weather_data, labels, metadata
        
    def _generate_load_profile(self, archetype: int, date: datetime) -> np.ndarray:
        """Generate a 24-hour load profile for a specific archetype and date."""
        hours = np.arange(24)
        base_load = 0.2 + 0.1 * np.random.random()  # Random base load
        
        if archetype == 0:  # High morning peak
            morning_peak = 0.8 * np.exp(-((hours - 7)**2) / 8)
            evening_peak = 0.4 * np.exp(-((hours - 19)**2) / 12)
            profile = base_load + morning_peak + evening_peak
            
        elif archetype == 1:  # High evening peak
            morning_peak = 0.3 * np.exp(-((hours - 7)**2) / 8)
            evening_peak = 0.9 * np.exp(-((hours - 19)**2) / 8)
            profile = base_load + morning_peak + evening_peak
            
        elif archetype == 2:  # Flat consumption
            profile = base_load + 0.1 * np.random.randn(24)
            
        elif archetype == 3:  # Night shift
            night_peak = 0.7 * np.exp(-((hours - 2)**2) / 12)
            afternoon_peak = 0.5 * np.exp(-((hours - 14)**2) / 8)
            profile = base_load + night_peak + afternoon_peak
            
        else:  # Weekend pattern
            late_morning = 0.5 * np.exp(-((hours - 10)**2) / 12)
            evening = 0.6 * np.exp(-((hours - 20)**2) / 16)
            profile = base_load + late_morning + evening
            
        # Add weekend effects
        if date.weekday() >= 5:  # Weekend
            profile *= 1.2  # Higher consumption on weekends
            
        # Add noise and ensure non-negative
        noise = 0.05 * np.random.randn(24)
        profile = np.maximum(profile + noise, 0.01)
        
        # Normalize to [0, 1] range
        profile = (profile - profile.min()) / (profile.max() - profile.min())
        
        return profile
        
    def _generate_weather_profile(self, date: datetime) -> np.ndarray:
        """Generate 24-hour weather profile (temperature, humidity)."""
        hours = np.arange(24)
        
        # Seasonal temperature variation
        day_of_year = date.timetuple().tm_yday
        seasonal_temp = 0.5 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Daily temperature cycle
        daily_cycle = 0.2 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
        
        # Temperature profile
        temperature = seasonal_temp + daily_cycle + 0.05 * np.random.randn(24)
        temperature = np.clip(temperature, 0, 1)
        
        # Humidity (inversely correlated with temperature)
        humidity = 0.8 - 0.4 * temperature + 0.1 * np.random.randn(24)
        humidity = np.clip(humidity, 0, 1)
        
        return np.stack([temperature, humidity], axis=-1)


if __name__ == "__main__":
    # Example usage
    generator = PecanStreetStyleGenerator(random_state=42)
    
    load_profiles, weather_data, labels, metadata = generator.generate_pecan_street_dataset(
        n_homes=10, n_days=7, start_date="2022-01-01"
    )
    
    print(f"Generated dataset:")
    print(f"  Load profiles shape: {load_profiles.shape}")
    print(f"  Weather data shape: {weather_data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Metadata entries: {len(metadata)}")
    print(f"  Archetypes: {np.unique(labels)}")