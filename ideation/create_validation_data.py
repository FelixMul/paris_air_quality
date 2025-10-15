import pandas as pd
import numpy as np

def create_validation_datasets():
    """Create training and validation datasets for ideation"""
    
    print("Loading original training data...")
    # Load the original training data
    train_df = pd.read_csv('../data/train.csv')
    
    # Convert id to datetime
    train_df['datetime'] = pd.to_datetime(train_df['id'])
    
    print(f"Original training data: {len(train_df)} rows")
    print(f"Date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    
    # Create training data - ONLY data BEFORE Sept 3, 2023 (no data leakage)
    print("\nCreating training data (ONLY data before Sept 3, 2023)...")
    print("   This prevents data leakage by not using future data for training")
    train_ideation = train_df[
        train_df['datetime'] < '2023-09-03 00:00:00'
    ].copy()
    
    # Add time features (same as your simplified approach)
    train_ideation['hour'] = train_ideation['datetime'].dt.hour
    train_ideation['day_of_week'] = train_ideation['datetime'].dt.dayofweek
    train_ideation['month'] = train_ideation['datetime'].dt.month
    train_ideation['day_of_year'] = train_ideation['datetime'].dt.dayofyear
    train_ideation['is_weekend'] = (train_ideation['day_of_week'] >= 5).astype(int)
    
    # Select features and targets
    feature_cols = ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend']
    target_cols = ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']
    
    # Create simplified training dataset
    train_simplified = train_ideation[['id'] + feature_cols + target_cols].copy()
    
    # Save training data
    train_simplified.to_csv('train_simplified.csv', index=False)
    print(f"Created train_simplified.csv with {len(train_simplified)} rows")
    print(f"Training period: {train_ideation['datetime'].min()} to {train_ideation['datetime'].max()}")
    
    # Extract September 3-24, 2023 data for validation
    print("\nCreating validation data (Sept 3-24, 2023)...")
    print("   This is our validation period with known results")
    test_2023 = train_df[
        (train_df['datetime'].dt.year == 2023) & 
        (train_df['datetime'].dt.month == 9) & 
        (train_df['datetime'].dt.day >= 3) & 
        (train_df['datetime'].dt.day <= 24)
    ].copy()
    
    print(f"Found {len(test_2023)} rows for September 3-24, 2023")
    print(f"Validation period: {test_2023['datetime'].min()} to {test_2023['datetime'].max()}")
    
    # Add time features
    test_2023['hour'] = test_2023['datetime'].dt.hour
    test_2023['day_of_week'] = test_2023['datetime'].dt.dayofweek
    test_2023['month'] = test_2023['datetime'].dt.month
    test_2023['day_of_year'] = test_2023['datetime'].dt.dayofyear
    test_2023['is_weekend'] = (test_2023['day_of_week'] >= 5).astype(int)
    
    # Create test dataset with features and targets
    test_simplified_with_results = test_2023[['id'] + feature_cols + target_cols].copy()
    
    # Save test data with results
    test_simplified_with_results.to_csv('test_simplified_with_results.csv', index=False)
    print(f"Created test_simplified_with_results.csv with {len(test_simplified_with_results)} rows")
    
    # Also create test data without results (for prediction)
    test_features_only = test_2023[['id'] + feature_cols].copy()
    test_features_only.to_csv('test_simplified.csv', index=False)
    print(f"Created test_simplified.csv with {len(test_features_only)} rows")
    
    # Show sample of validation data
    print("\nSample of validation data (Sept 3-24, 2023):")
    print(test_simplified_with_results.head())
    
    print("\n" + "=" * 60)
    print("VALIDATION DATASETS CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ NO DATA LEAKAGE - Training data stops before validation period")
    print("✅ PROPER TEMPORAL SPLIT - Train → Validate → Test")
    print("✅ REALISTIC EVALUATION - Model only uses past information")
    print("=" * 60)
    
    return train_simplified, test_simplified_with_results

if __name__ == "__main__":
    create_validation_datasets()
