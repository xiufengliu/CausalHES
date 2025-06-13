#!/usr/bin/env python3
"""
Generate realistic long-term household energy consumption time series data.

This script generates multi-year household energy consumption data with:
- Seasonal variations
- Weekly patterns (weekday vs weekend)
- Long-term trends
- Weather correlations
- Individual household characteristics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class RealisticTimeSeriesGenerator:
    """
    Generator for realistic long-term household energy consumption time series.
    """
    
    def __init__(self, random_state: Optional[int] = 42):
        """Initialize the generator."""
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
        # Seasonal parameters
        self.seasonal_params = {
            'summer_factor': 1.3,  # Higher consumption in summer (AC)
            'winter_factor': 1.2,  # Higher consumption in winter (heating)
            'spring_fall_factor': 0.9,  # Lower consumption in mild seasons
        }
        
        # Weekly patterns
        self.weekly_params = {
            'weekend_factor': 1.1,  # Slightly higher weekend consumption
            'weekday_factor': 1.0,
        }
        
    def generate_long_term_data(self,
                              n_households: int = 100,
                              n_years: int = 2,
                              n_clusters: int = 5,
                              start_date: str = "2022-01-01") -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Generate long-term household energy consumption data.
        
        Args:
            n_households: Number of households
            n_years: Number of years to generate
            n_clusters: Number of consumption clusters
            start_date: Start date for time series
            
        Returns:
            Tuple of (consumption_df, weather_df, household_clusters)
        """
        print(f"Generating {n_years} years of data for {n_households} households...")
        
        # Create date range
        start = pd.to_datetime(start_date)
        end = start + pd.DateOffset(years=n_years)
        date_range = pd.date_range(start=start, end=end, freq='H')[:-1]  # Exclude last hour
        n_timesteps = len(date_range)
        
        print(f"Time range: {start} to {end} ({n_timesteps:,} hours)")
        
        # Generate household characteristics
        household_clusters = self._assign_household_clusters(n_households, n_clusters)
        household_chars = self._generate_household_characteristics(n_households, household_clusters)
        
        # Generate weather data
        print("Generating weather patterns...")
        weather_df = self._generate_weather_timeseries(date_range)
        
        # Generate consumption data
        print("Generating consumption patterns...")
        consumption_data = []
        
        for household_id in range(n_households):
            if household_id % 20 == 0:
                print(f"  Processing household {household_id}/{n_households}")
                
            household_consumption = self._generate_household_timeseries(
                household_id, 
                household_chars[household_id],
                date_range,
                weather_df
            )
            consumption_data.append(household_consumption)
            
        # Create consumption DataFrame
        consumption_df = pd.DataFrame({
            'datetime': np.tile(date_range, n_households),
            'household_id': np.repeat(range(n_households), n_timesteps),
            'consumption': np.concatenate(consumption_data),
            'cluster': np.repeat(household_clusters, n_timesteps)
        })
        
        # Add time features
        consumption_df = self._add_time_features(consumption_df)
        
        print(f"Generated consumption data shape: {consumption_df.shape}")
        print(f"Generated weather data shape: {weather_df.shape}")
        
        return consumption_df, weather_df, household_clusters
        
    def _assign_household_clusters(self, n_households: int, n_clusters: int) -> np.ndarray:
        """Assign households to clusters."""
        # Create balanced clusters
        households_per_cluster = n_households // n_clusters
        clusters = []
        
        for cluster_id in range(n_clusters):
            n_in_cluster = households_per_cluster
            if cluster_id == n_clusters - 1:  # Last cluster gets remaining
                n_in_cluster = n_households - cluster_id * households_per_cluster
            clusters.extend([cluster_id] * n_in_cluster)
            
        return np.array(clusters)
        
    def _generate_household_characteristics(self, n_households: int, clusters: np.ndarray) -> list:
        """Generate characteristics for each household."""
        characteristics = []
        
        cluster_profiles = {
            0: {'type': 'low_consumption', 'base_load': 0.3, 'variability': 0.2, 'weather_sensitivity': 0.3},
            1: {'type': 'medium_residential', 'base_load': 0.5, 'variability': 0.3, 'weather_sensitivity': 0.5},
            2: {'type': 'high_residential', 'base_load': 0.7, 'variability': 0.4, 'weather_sensitivity': 0.7},
            3: {'type': 'commercial', 'base_load': 0.8, 'variability': 0.2, 'weather_sensitivity': 0.4},
            4: {'type': 'industrial', 'base_load': 0.9, 'variability': 0.1, 'weather_sensitivity': 0.2},
        }
        
        for household_id in range(n_households):
            cluster_id = clusters[household_id]
            base_profile = cluster_profiles.get(cluster_id, cluster_profiles[0])
            
            # Add individual variations
            char = {
                'cluster': cluster_id,
                'type': base_profile['type'],
                'base_load': base_profile['base_load'] + np.random.normal(0, 0.1),
                'variability': base_profile['variability'] + np.random.normal(0, 0.05),
                'weather_sensitivity': base_profile['weather_sensitivity'] + np.random.normal(0, 0.1),
                'seasonal_preference': np.random.choice(['summer_high', 'winter_high', 'balanced']),
                'weekend_factor': 1.0 + np.random.normal(0, 0.1),
                'trend_slope': np.random.normal(0, 0.001),  # Long-term trend
            }
            
            # Ensure positive values
            char['base_load'] = max(0.1, char['base_load'])
            char['variability'] = max(0.05, char['variability'])
            char['weather_sensitivity'] = max(0.0, min(1.0, char['weather_sensitivity']))
            
            characteristics.append(char)
            
        return characteristics
        
    def _generate_weather_timeseries(self, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic weather time series."""
        n_timesteps = len(date_range)
        
        # Create base seasonal temperature pattern
        day_of_year = date_range.dayofyear
        hour_of_day = date_range.hour
        
        # Annual temperature cycle (sine wave)
        annual_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
        
        # Daily temperature cycle
        daily_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak in afternoon
        
        # Combine and add noise
        temperature = annual_temp + daily_temp + np.random.normal(0, 2, n_timesteps)
        
        # Generate humidity (inverse correlation with temperature)
        base_humidity = 70 - 0.8 * (temperature - np.mean(temperature))
        daily_humidity = 15 * np.sin(2 * np.pi * (hour_of_day - 18) / 24)  # Peak at night
        humidity = base_humidity + daily_humidity + np.random.normal(0, 5, n_timesteps)
        humidity = np.clip(humidity, 10, 95)
        
        # Normalize to [0, 1]
        temp_normalized = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature))
        humid_normalized = (humidity - 10) / (95 - 10)
        
        weather_df = pd.DataFrame({
            'datetime': date_range,
            'temperature': temp_normalized,
            'humidity': humid_normalized,
            'temperature_raw': temperature,
            'humidity_raw': humidity
        })
        
        return weather_df
        
    def _generate_household_timeseries(self, 
                                     household_id: int,
                                     characteristics: dict,
                                     date_range: pd.DatetimeIndex,
                                     weather_df: pd.DataFrame) -> np.ndarray:
        """Generate time series for a single household."""
        n_timesteps = len(date_range)
        
        # Base consumption pattern
        base_consumption = np.full(n_timesteps, characteristics['base_load'])
        
        # Add seasonal variations
        seasonal_factor = self._get_seasonal_factors(date_range, characteristics)
        base_consumption *= seasonal_factor
        
        # Add weekly patterns
        weekly_factor = self._get_weekly_factors(date_range, characteristics)
        base_consumption *= weekly_factor
        
        # Add daily patterns
        daily_factor = self._get_daily_factors(date_range, characteristics)
        base_consumption *= daily_factor
        
        # Add weather influence
        weather_factor = self._get_weather_factors(weather_df, characteristics)
        base_consumption *= weather_factor
        
        # Add long-term trend
        trend_factor = self._get_trend_factors(date_range, characteristics)
        base_consumption *= trend_factor
        
        # Add noise
        noise = np.random.normal(1, characteristics['variability'], n_timesteps)
        base_consumption *= noise
        
        # Ensure non-negative values
        base_consumption = np.maximum(base_consumption, 0.01)
        
        return base_consumption
        
    def _get_seasonal_factors(self, date_range: pd.DatetimeIndex, char: dict) -> np.ndarray:
        """Get seasonal variation factors."""
        month = date_range.month
        factors = np.ones(len(date_range))
        
        # Summer months (June, July, August)
        summer_mask = np.isin(month, [6, 7, 8])
        # Winter months (December, January, February)
        winter_mask = np.isin(month, [12, 1, 2])
        # Spring/Fall months
        spring_fall_mask = ~(summer_mask | winter_mask)
        
        if char['seasonal_preference'] == 'summer_high':
            factors[summer_mask] = self.seasonal_params['summer_factor']
            factors[winter_mask] = self.seasonal_params['spring_fall_factor']
        elif char['seasonal_preference'] == 'winter_high':
            factors[winter_mask] = self.seasonal_params['winter_factor']
            factors[summer_mask] = self.seasonal_params['spring_fall_factor']
        else:  # balanced
            factors[summer_mask] = 1.1
            factors[winter_mask] = 1.1
            
        factors[spring_fall_mask] = self.seasonal_params['spring_fall_factor']
        
        return factors
        
    def _get_weekly_factors(self, date_range: pd.DatetimeIndex, char: dict) -> np.ndarray:
        """Get weekly pattern factors."""
        is_weekend = date_range.weekday >= 5  # Saturday=5, Sunday=6
        factors = np.ones(len(date_range))
        factors[is_weekend] = char['weekend_factor']
        return factors
        
    def _get_daily_factors(self, date_range: pd.DatetimeIndex, char: dict) -> np.ndarray:
        """Get daily pattern factors."""
        hour = date_range.hour
        factors = np.ones(len(date_range))
        
        if char['type'] == 'residential' or 'residential' in char['type']:
            # Morning peak (7-9 AM)
            morning_peak = ((hour >= 7) & (hour <= 9))
            factors[morning_peak] = 1.3
            
            # Evening peak (6-8 PM)
            evening_peak = ((hour >= 18) & (hour <= 20))
            factors[evening_peak] = 1.4
            
            # Night low (11 PM - 5 AM)
            night_low = ((hour >= 23) | (hour <= 5))
            factors[night_low] = 0.7
            
        elif char['type'] == 'commercial':
            # Business hours (9 AM - 5 PM)
            business_hours = ((hour >= 9) & (hour <= 17))
            factors[business_hours] = 1.2
            factors[~business_hours] = 0.6
            
        elif char['type'] == 'industrial':
            # More constant load with slight variations
            factors += 0.1 * np.sin(2 * np.pi * hour / 24)
            
        return factors
        
    def _get_weather_factors(self, weather_df: pd.DataFrame, char: dict) -> np.ndarray:
        """Get weather influence factors."""
        temp = weather_df['temperature'].values
        humidity = weather_df['humidity'].values
        
        # Temperature effect (higher temp -> higher cooling load)
        temp_effect = char['weather_sensitivity'] * (temp - 0.5) * 0.5
        
        # Humidity effect (higher humidity -> slightly higher load)
        humidity_effect = char['weather_sensitivity'] * (humidity - 0.5) * 0.2
        
        weather_factor = 1.0 + temp_effect + humidity_effect
        
        return weather_factor
        
    def _get_trend_factors(self, date_range: pd.DatetimeIndex, char: dict) -> np.ndarray:
        """Get long-term trend factors."""
        # Linear trend over time
        time_index = np.arange(len(date_range))
        trend_factor = 1.0 + char['trend_slope'] * time_index / len(date_range)
        return trend_factor
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataframe."""
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['dayofyear'] = df['datetime'].dt.dayofyear
        df['is_weekend'] = df['dayofweek'] >= 5
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        return df


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    """Generate realistic long-term time series data."""
    print("Starting realistic time series data generation...")
    
    # Create output directory
    output_dir = Path("data/realistic_timeseries")
    ensure_dir(output_dir)
    
    # Initialize generator
    generator = RealisticTimeSeriesGenerator(random_state=42)
    
    # Generation parameters
    n_households = 50  # Smaller number for manageable file sizes
    n_years = 2
    n_clusters = 5
    start_date = "2022-01-01"
    
    print(f"Parameters:")
    print(f"  Households: {n_households}")
    print(f"  Years: {n_years}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Start date: {start_date}")
    
    # Generate data
    consumption_df, weather_df, household_clusters = generator.generate_long_term_data(
        n_households=n_households,
        n_years=n_years,
        n_clusters=n_clusters,
        start_date=start_date
    )
    
    # Save consumption data
    print("Saving consumption data...")
    consumption_path = output_dir / "household_consumption_timeseries.csv"
    consumption_df.to_csv(consumption_path, index=False)
    print(f"Consumption data saved to: {consumption_path}")
    print(f"Size: {consumption_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save weather data
    print("Saving weather data...")
    weather_path = output_dir / "weather_timeseries.csv"
    weather_df.to_csv(weather_path, index=False)
    print(f"Weather data saved to: {weather_path}")
    print(f"Size: {weather_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Create household metadata
    print("Creating household metadata...")
    household_metadata = pd.DataFrame({
        'household_id': range(n_households),
        'cluster': household_clusters
    })
    
    metadata_path = output_dir / "household_metadata.csv"
    household_metadata.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")
    
    # Generate summary statistics
    print("Generating summary statistics...")
    
    # Consumption statistics by cluster
    cluster_stats = consumption_df.groupby('cluster')['consumption'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Seasonal statistics
    seasonal_stats = consumption_df.groupby(['cluster', 'season'])['consumption'].mean().unstack()
    
    # Save statistics
    stats_path = output_dir / "summary_statistics.csv"
    cluster_stats.to_csv(stats_path)
    
    seasonal_path = output_dir / "seasonal_statistics.csv"
    seasonal_stats.to_csv(seasonal_path)
    
    print(f"Statistics saved to: {stats_path} and {seasonal_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("REALISTIC TIME SERIES GENERATION SUMMARY")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Time range: {consumption_df['datetime'].min()} to {consumption_df['datetime'].max()}")
    print(f"Total data points: {len(consumption_df):,}")
    print(f"Households: {n_households}")
    print(f"Time steps per household: {len(consumption_df) // n_households:,}")
    print(f"Clusters: {n_clusters}")
    
    print("\nCluster distribution:")
    print(household_metadata['cluster'].value_counts().sort_index())
    
    print("\nConsumption statistics by cluster:")
    print(cluster_stats)
    
    print("\nFiles created:")
    print(f"  1. household_consumption_timeseries.csv ({consumption_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  2. weather_timeseries.csv ({weather_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  3. household_metadata.csv")
    print(f"  4. summary_statistics.csv")
    print(f"  5. seasonal_statistics.csv")
    print("="*60)


if __name__ == "__main__":
    main()
