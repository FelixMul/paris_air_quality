"""
Enhanced Feature Engineering for Air Quality Model
================================================

This script creates enhanced datasets with:
1. Cyclical encoding of hour, day_of_week, month
2. Interaction features like hour × day_of_week
3. Additional time-based features

Usage:
    python create_enhanced_features.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

def add_cyclical_features(df):
    """Add cyclical encoding for time features"""
    print("Adding cyclical encoding features...")
    
    # Hour cyclical encoding (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week cyclical encoding (7-day cycle)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month cyclical encoding (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print("  ✓ Added hour_sin, hour_cos")
    print("  ✓ Added day_of_week_sin, day_of_week_cos")
    print("  ✓ Added month_sin, month_cos")

def add_interaction_features(df):
    """Add interaction features between time variables"""
    print("Adding interaction features...")
    
    # Hour × Day of week interaction
    df['hour_day_interaction'] = df['hour'] * df['day_of_week']
    
    # Hour × Month interaction
    df['hour_month_interaction'] = df['hour'] * df['month']
    
    # Day of week × Month interaction
    df['day_month_interaction'] = df['day_of_week'] * df['month']
    
    # Weekend × Hour interaction
    df['weekend_hour_interaction'] = df['is_weekend'] * df['hour']
    
    # Weekend × Month interaction
    df['weekend_month_interaction'] = df['is_weekend'] * df['month']
    
    print("  ✓ Added hour_day_interaction")
    print("  ✓ Added hour_month_interaction")
    print("  ✓ Added day_month_interaction")
    print("  ✓ Added weekend_hour_interaction")
    print("  ✓ Added weekend_month_interaction")

def add_advanced_time_features(df):
    """Add advanced time-based features"""
    print("Adding advanced time features...")
    
    # Time since midnight (continuous)
    df['time_since_midnight'] = df['hour'] + df['datetime'].dt.minute / 60.0
    
    # Time since start of week (continuous, Monday = 0)
    df['time_since_week_start'] = df['day_of_week'] * 24 + df['hour']
    
    # Is it Monday? (Monday effect)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    
    # Is it Friday? (Friday effect)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Is it end of month? (last 3 days)
    df['is_end_of_month'] = (df['datetime'].dt.day >= 29).astype(int)
    
    # Is it beginning of month? (first 3 days)
    df['is_beginning_of_month'] = (df['datetime'].dt.day <= 3).astype(int)
    
    print("  ✓ Added time_since_midnight")
    print("  ✓ Added time_since_week_start")
    print("  ✓ Added is_monday, is_friday")
    print("  ✓ Added is_end_of_month, is_beginning_of_month")

def create_enhanced_datasets():
    """Create enhanced training and test datasets with new features"""
    
    print("=" * 60)
    print("CREATING ENHANCED FEATURE DATASETS")
    print("=" * 60)
    
    # Load original training data
    print("1. Loading original training data...")
    train_df = pd.read_csv('../data/train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['id'])
    
    print(f"   Original training data: {len(train_df)} rows")
    print(f"   Date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    
    # Create training data - ONLY data BEFORE Sept 3, 2023 (no data leakage)
    print("\n2. Creating training data (ONLY data before Sept 3, 2023)...")
    train_enhanced = train_df[
        train_df['datetime'] < '2023-09-03 00:00:00'
    ].copy()
    
    # Add basic time features
    print("\n3. Adding basic time features...")
    train_enhanced['hour'] = train_enhanced['datetime'].dt.hour
    train_enhanced['day_of_week'] = train_enhanced['datetime'].dt.dayofweek
    train_enhanced['month'] = train_enhanced['datetime'].dt.month
    train_enhanced['day_of_year'] = train_enhanced['datetime'].dt.dayofyear
    train_enhanced['is_weekend'] = (train_enhanced['day_of_week'] >= 5).astype(int)
    
    # Add enhanced features
    print("\n4. Adding enhanced features...")
    add_cyclical_features(train_enhanced)
    add_interaction_features(train_enhanced)
    add_advanced_time_features(train_enhanced)
    
    # Define feature columns (all new features)
    feature_cols = [
        # Basic features
        'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
        # Cyclical features
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
        'month_sin', 'month_cos',
        # Interaction features
        'hour_day_interaction', 'hour_month_interaction', 'day_month_interaction',
        'weekend_hour_interaction', 'weekend_month_interaction',
        # Advanced time features
        'time_since_midnight', 'time_since_week_start', 'is_monday', 'is_friday',
        'is_end_of_month', 'is_beginning_of_month'
    ]
    
    target_cols = ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']
    
    # Create enhanced training dataset
    train_enhanced_final = train_enhanced[['id'] + feature_cols + target_cols].copy()
    
    # Save enhanced training data
    train_enhanced_final.to_csv('train_enhanced.csv', index=False)
    print(f"\n5. Created train_enhanced.csv with {len(train_enhanced_final)} rows")
    print(f"   Features: {len(feature_cols)} (vs 5 in original)")
    print(f"   Training period: {train_enhanced['datetime'].min()} to {train_enhanced['datetime'].max()}")
    
    # Extract September 3-24, 2023 data for validation
    print("\n6. Creating enhanced validation data (Sept 3-24, 2023)...")
    test_2023 = train_df[
        (train_df['datetime'].dt.year == 2023) & 
        (train_df['datetime'].dt.month == 9) & 
        (train_df['datetime'].dt.day >= 3) & 
        (train_df['datetime'].dt.day <= 24)
    ].copy()
    
    print(f"   Found {len(test_2023)} rows for September 3-24, 2023")
    
    # Add basic time features
    test_2023['hour'] = test_2023['datetime'].dt.hour
    test_2023['day_of_week'] = test_2023['datetime'].dt.dayofweek
    test_2023['month'] = test_2023['datetime'].dt.month
    test_2023['day_of_year'] = test_2023['datetime'].dt.dayofyear
    test_2023['is_weekend'] = (test_2023['day_of_week'] >= 5).astype(int)
    
    # Add enhanced features
    add_cyclical_features(test_2023)
    add_interaction_features(test_2023)
    add_advanced_time_features(test_2023)
    
    # Create enhanced test dataset with results
    test_enhanced_with_results = test_2023[['id'] + feature_cols + target_cols].copy()
    test_enhanced_with_results.to_csv('test_enhanced_with_results.csv', index=False)
    print(f"   Created test_enhanced_with_results.csv with {len(test_enhanced_with_results)} rows")
    
    # Create enhanced test dataset without results (for prediction)
    test_enhanced_features_only = test_2023[['id'] + feature_cols].copy()
    test_enhanced_features_only.to_csv('test_enhanced.csv', index=False)
    print(f"   Created test_enhanced.csv with {len(test_enhanced_features_only)} rows")
    
    # Show feature summary
    print("\n" + "=" * 60)
    print("ENHANCED FEATURE SUMMARY")
    print("=" * 60)
    print(f"Total features: {len(feature_cols)}")
    print("\nFeature categories:")
    print("  Basic features (5): hour, day_of_week, month, day_of_year, is_weekend")
    print("  Cyclical features (6): hour_sin/cos, day_sin/cos, month_sin/cos")
    print("  Interaction features (5): hour×day, hour×month, day×month, weekend×hour, weekend×month")
    print("  Advanced features (6): time_since_midnight, time_since_week_start, is_monday, is_friday, is_end_of_month, is_beginning_of_month")
    print("=" * 60)
    
    # Show sample of enhanced data
    print("\nSample of enhanced training data:")
    print(train_enhanced_final[['id', 'hour', 'hour_sin', 'hour_cos', 'hour_day_interaction', 'is_monday']].head())
    
    print("\n" + "=" * 60)
    print("ENHANCED DATASETS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ Cyclical encoding added for hour, day_of_week, month")
    print("✅ Interaction features added (hour×day, hour×month, etc.)")
    print("✅ Advanced time features added")
    print("✅ Ready for fine-tuned model testing")
    print("=" * 60)
    
    return train_enhanced_final, test_enhanced_with_results

if __name__ == "__main__":
    create_enhanced_datasets()
