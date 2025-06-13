#!/usr/bin/env python3
"""
Generate Pecan Street-style household energy consumption dataset.

This script generates synthetic data that matches the format described in the paper:
- 500 homes in Austin, Texas
- One-year period of daily 24-hour load profiles
- 5 consumption archetypes: Low Usage, Morning Peakers, Afternoon Peakers, Evening Peakers, Night Owls
- Weather data: temperature and humidity
- Format: X^(p) ∈ R^(N_samples × 24 × 1), X^(s) ∈ R^(N_samples × 24 × 2)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class PecanStreetStyleGenerator:
    """
    Generator for Pecan Street-style household energy consumption data.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the generator."""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define the 5 consumption archetypes from the paper
        self.archetypes = {
            0: {
                'name': 'Low Usage',
                'description': 'Minimal consumption throughout the day',
                'base_consumption': 0.15,
                'peak_hours': [],
                'peak_strength': 0.2,
                'variability': 0.1
            },
            1: {
                'name': 'Morning Peakers',
                'description': 'High consumption in morning hours',
                'base_consumption': 0.3,
                'peak_hours': [6, 7, 8, 9],
                'peak_strength': 0.6,
                'variability': 0.2
            },
            2: {
                'name': 'Afternoon Peakers', 
                'description': 'High consumption in afternoon hours',
                'base_consumption': 0.35,
                'peak_hours': [12, 13, 14, 15, 16],
                'peak_strength': 0.7,
                'variability': 0.25
            },
            3: {
                'name': 'Evening Peakers',
                'description': 'High consumption in evening hours',
                'base_consumption': 0.4,
                'peak_hours': [17, 18, 19, 20, 21],
                'peak_strength': 0.8,
                'variability': 0.3
            },
            4: {
                'name': 'Night Owls',
                'description': 'High consumption in late night/early morning',
                'base_consumption': 0.25,
                'peak_hours': [22, 23, 0, 1, 2],
                'peak_strength': 0.5,
                'variability': 0.2
            }
        }
        
    def generate_pecan_street_dataset(self,
                                    n_homes: int = 500,
                                    n_days: int = 365,
                                    start_date: str = "2022-01-01") -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate Pecan Street-style dataset.
        
        Args:
            n_homes: Number of homes (500 in paper)
            n_days: Number of days (365 for one year)
            start_date: Start date for the dataset
            
        Returns:
            Tuple of (load_profiles, weather_data, labels, metadata)
        """
        print(f"Generating Pecan Street-style dataset...")
        print(f"Homes: {n_homes}, Days: {n_days}")
        
        # Assign homes to archetypes
        home_archetypes = self._assign_home_archetypes(n_homes)
        
        # Generate date range
        start = pd.to_datetime(start_date)
        dates = pd.date_range(start=start, periods=n_days, freq='D')
        
        # Generate weather data for Austin, Texas
        print("Generating Austin weather patterns...")
        weather_data = self._generate_austin_weather(dates)
        
        # Generate load profiles
        print("Generating household load profiles...")
        load_profiles = []
        sample_metadata = []
        sample_labels = []
        
        sample_id = 0
        for home_id in range(n_homes):
            if home_id % 100 == 0:
                print(f"  Processing home {home_id}/{n_homes}")
                
            archetype = home_archetypes[home_id]
            home_characteristics = self._generate_home_characteristics(home_id, archetype)
            
            for day_idx, date in enumerate(dates):
                # Generate 24-hour load profile for this home on this day
                daily_profile = self._generate_daily_load_profile(
                    home_characteristics, 
                    weather_data[day_idx],
                    date
                )
                
                load_profiles.append(daily_profile)
                sample_labels.append(archetype)
                sample_metadata.append({
                    'sample_id': sample_id,
                    'home_id': home_id,
                    'date': date,
                    'archetype': archetype,
                    'archetype_name': self.archetypes[archetype]['name'],
                    'day_of_year': date.dayofyear,
                    'month': date.month,
                    'season': self._get_season(date.month),
                    'is_weekend': date.weekday() >= 5
                })
                sample_id += 1
                
        # Convert to numpy arrays
        load_profiles = np.array(load_profiles)  # Shape: (N_samples, 24, 1)
        weather_data_expanded = np.tile(weather_data, (n_homes, 1, 1))  # Shape: (N_samples, 24, 2)
        labels = np.array(sample_labels)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(sample_metadata)
        
        print(f"Generated dataset shapes:")
        print(f"  Load profiles: {load_profiles.shape}")
        print(f"  Weather data: {weather_data_expanded.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Metadata: {metadata_df.shape}")
        
        return load_profiles, weather_data_expanded, labels, metadata_df
        
    def _assign_home_archetypes(self, n_homes: int) -> np.ndarray:
        """Assign homes to the 5 consumption archetypes."""
        # Create balanced distribution across archetypes
        homes_per_archetype = n_homes // 5
        archetypes = []
        
        for archetype_id in range(5):
            n_homes_this_type = homes_per_archetype
            if archetype_id == 4:  # Last archetype gets remaining homes
                n_homes_this_type = n_homes - archetype_id * homes_per_archetype
            archetypes.extend([archetype_id] * n_homes_this_type)
            
        return np.array(archetypes)
        
    def _generate_home_characteristics(self, home_id: int, archetype: int) -> Dict:
        """Generate individual characteristics for a home."""
        base_archetype = self.archetypes[archetype]
        
        # Add individual variations to archetype parameters
        characteristics = {
            'home_id': home_id,
            'archetype': archetype,
            'base_consumption': base_archetype['base_consumption'] + np.random.normal(0, 0.05),
            'peak_hours': base_archetype['peak_hours'].copy(),
            'peak_strength': base_archetype['peak_strength'] + np.random.normal(0, 0.1),
            'variability': base_archetype['variability'] + np.random.normal(0, 0.02),
            'weather_sensitivity': np.random.uniform(0.1, 0.5),
            'seasonal_factor': np.random.uniform(0.8, 1.2),
            'weekend_factor': np.random.uniform(0.9, 1.1)
        }
        
        # Ensure positive values
        characteristics['base_consumption'] = max(0.05, characteristics['base_consumption'])
        characteristics['peak_strength'] = max(0.1, characteristics['peak_strength'])
        characteristics['variability'] = max(0.05, characteristics['variability'])
        
        return characteristics
        
    def _generate_austin_weather(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic weather patterns for Austin, Texas."""
        n_days = len(dates)
        weather_data = np.zeros((n_days, 24, 2))  # (days, hours, [temp, humidity])
        
        for day_idx, date in enumerate(dates):
            # Austin climate characteristics
            day_of_year = date.dayofyear
            
            # Annual temperature cycle (Austin: hot summers, mild winters)
            annual_temp_base = 70 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # °F
            
            # Daily temperature cycle
            hours = np.arange(24)
            daily_temp_variation = 15 * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at 2 PM
            
            # Temperature for each hour
            temperature_f = annual_temp_base + daily_temp_variation + np.random.normal(0, 3, 24)
            
            # Convert to Celsius and normalize
            temperature_c = (temperature_f - 32) * 5/9
            temp_normalized = (temperature_c - 0) / 40  # Assume 0-40°C range
            temp_normalized = np.clip(temp_normalized, 0, 1)
            
            # Humidity (Austin is humid, inverse correlation with temperature)
            base_humidity = 70 - 0.5 * (temperature_f - 70)  # Base humidity
            daily_humidity_variation = 20 * np.sin(2 * np.pi * (hours - 18) / 24)  # Peak at night
            
            humidity = base_humidity + daily_humidity_variation + np.random.normal(0, 5, 24)
            humidity = np.clip(humidity, 20, 95)  # Realistic humidity range
            humidity_normalized = (humidity - 20) / 75  # Normalize to [0,1]
            
            weather_data[day_idx, :, 0] = temp_normalized
            weather_data[day_idx, :, 1] = humidity_normalized
            
        return weather_data
        
    def _generate_daily_load_profile(self, 
                                   home_characteristics: Dict,
                                   weather_day: np.ndarray,
                                   date: pd.Timestamp) -> np.ndarray:
        """Generate 24-hour load profile for a specific home and day."""
        # Base consumption pattern
        base_load = np.full(24, home_characteristics['base_consumption'])
        
        # Add archetype-specific peaks
        for peak_hour in home_characteristics['peak_hours']:
            peak_strength = home_characteristics['peak_strength']
            # Add some randomness to peak strength
            actual_strength = peak_strength * np.random.uniform(0.8, 1.2)
            base_load[peak_hour] += actual_strength
            
            # Add some spread around the peak
            for offset in [-1, 1]:
                neighbor_hour = (peak_hour + offset) % 24
                base_load[neighbor_hour] += actual_strength * 0.3
                
        # Weather influence (temperature effect)
        temperature = weather_day[:, 0]  # Normalized temperature
        humidity = weather_day[:, 1]     # Normalized humidity
        
        # Cooling load effect (higher temp -> higher consumption in summer)
        cooling_effect = home_characteristics['weather_sensitivity'] * temperature * 0.5
        
        # Heating load effect (lower temp -> higher consumption in winter)
        heating_effect = home_characteristics['weather_sensitivity'] * (1 - temperature) * 0.3
        
        # Seasonal adjustment
        month = date.month
        if month in [6, 7, 8]:  # Summer - more cooling
            weather_effect = cooling_effect
        elif month in [12, 1, 2]:  # Winter - more heating
            weather_effect = heating_effect
        else:  # Spring/Fall - moderate effect
            weather_effect = (cooling_effect + heating_effect) * 0.5
            
        base_load += weather_effect
        
        # Weekend effect
        if date.weekday() >= 5:  # Weekend
            base_load *= home_characteristics['weekend_factor']
            
        # Seasonal factor
        season_multiplier = self._get_seasonal_multiplier(date.month)
        base_load *= season_multiplier * home_characteristics['seasonal_factor']
        
        # Add noise
        noise = np.random.normal(0, home_characteristics['variability'], 24)
        base_load += noise
        
        # Ensure non-negative values
        base_load = np.maximum(base_load, 0.01)
        
        # Reshape to (24, 1) format
        return base_load.reshape(24, 1)
        
    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
            
    def _get_seasonal_multiplier(self, month: int) -> float:
        """Get seasonal consumption multiplier."""
        if month in [6, 7, 8]:  # Summer (high AC usage)
            return 1.3
        elif month in [12, 1, 2]:  # Winter (heating)
            return 1.1
        else:  # Spring/Fall
            return 0.9


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    """Generate Pecan Street-style dataset."""
    print("Starting Pecan Street-style data generation...")
    
    # Create output directory
    output_dir = Path("data/pecan_street_style")
    ensure_dir(output_dir)
    
    # Initialize generator
    generator = PecanStreetStyleGenerator(random_state=42)
    
    # Generation parameters (matching paper description)
    n_homes = 500
    n_days = 365  # One year
    start_date = "2022-01-01"
    
    print(f"Parameters:")
    print(f"  Homes: {n_homes}")
    print(f"  Days: {n_days} (one year)")
    print(f"  Start date: {start_date}")
    print(f"  Total samples: {n_homes * n_days:,}")
    
    # Generate dataset
    load_profiles, weather_data, labels, metadata_df = generator.generate_pecan_street_dataset(
        n_homes=n_homes,
        n_days=n_days,
        start_date=start_date
    )
    
    # Save in the format described in the paper
    print("Saving dataset in paper format...")
    
    # Save load profiles (primary modality)
    print("Saving load profiles (X^(p))...")
    np.save(output_dir / "load_profiles.npy", load_profiles)
    
    # Save weather data (secondary modality)  
    print("Saving weather data (X^(s))...")
    np.save(output_dir / "weather_data.npy", weather_data)
    
    # Save labels
    print("Saving labels...")
    np.save(output_dir / "labels.npy", labels)
    
    # Save metadata
    print("Saving metadata...")
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    
    # Create CSV versions for easy inspection
    print("Creating CSV versions...")
    
    # Flatten load profiles for CSV
    load_df = pd.DataFrame()
    load_df['sample_id'] = range(len(load_profiles))
    load_df['home_id'] = metadata_df['home_id']
    load_df['date'] = metadata_df['date']
    load_df['archetype'] = labels
    load_df['archetype_name'] = metadata_df['archetype_name']
    
    # Add hourly load data
    for hour in range(24):
        load_df[f'load_h{hour:02d}'] = load_profiles[:, hour, 0]
        
    load_df.to_csv(output_dir / "load_profiles.csv", index=False)
    
    # Flatten weather data for CSV
    weather_df = pd.DataFrame()
    weather_df['sample_id'] = range(len(weather_data))
    weather_df['date'] = metadata_df['date']
    
    # Add hourly weather data
    for hour in range(24):
        weather_df[f'temp_h{hour:02d}'] = weather_data[:, hour, 0]
        weather_df[f'humid_h{hour:02d}'] = weather_data[:, hour, 1]
        
    weather_df.to_csv(output_dir / "weather_data.csv", index=False)
    
    # Generate summary statistics
    print("Generating summary statistics...")
    
    # Archetype distribution
    archetype_counts = pd.Series(labels).value_counts().sort_index()
    archetype_names = [generator.archetypes[i]['name'] for i in range(5)]
    
    summary_stats = {
        'archetype': list(range(5)),
        'archetype_name': archetype_names,
        'count': archetype_counts.values,
        'percentage': (archetype_counts.values / len(labels) * 100).round(2),
        'mean_consumption': [load_profiles[labels == i].mean() for i in range(5)],
        'std_consumption': [load_profiles[labels == i].std() for i in range(5)]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "archetype_summary.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("PECAN STREET-STYLE DATASET GENERATION SUMMARY")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Dataset format matches paper description:")
    print(f"  X^(p) (load profiles): {load_profiles.shape}")
    print(f"  X^(s) (weather data): {weather_data.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Total samples: {len(load_profiles):,}")
    print(f"  Homes: {n_homes}")
    print(f"  Days per home: {n_days}")
    
    print("\nArchetype distribution:")
    for i, name in enumerate(archetype_names):
        count = archetype_counts[i]
        pct = count / len(labels) * 100
        print(f"  {i}: {name:<20} {count:>6} samples ({pct:5.1f}%)")
        
    print(f"\nFiles created:")
    print(f"  1. load_profiles.npy ({load_profiles.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  2. weather_data.npy ({weather_data.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  3. labels.npy")
    print(f"  4. metadata.csv")
    print(f"  5. load_profiles.csv")
    print(f"  6. weather_data.csv")
    print(f"  7. archetype_summary.csv")
    print("="*60)


if __name__ == "__main__":
    main()
