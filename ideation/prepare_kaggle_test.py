"""
Prepare Kaggle Test Data with Basic Features
==========================================

This script prepares the actual test data (September 3-24, 2024) 
with the same basic features used in our best model.
"""

import pandas as pd
import numpy as np

def prepare_kaggle_test_data():
    """Prepare the actual test data with basic features for Kaggle submission"""
    
    print("=" * 60)
    print("PREPARING KAGGLE TEST DATA WITH BASIC FEATURES")
    print("=" * 60)
    
    # Load the actual test data
    print("1. Loading actual test data...")
    test_df = pd.read_csv('../data/test.csv')
    test_df['datetime'] = pd.to_datetime(test_df['id'])
    
    print(f"   Test data shape: {test_df.shape}")
    print(f"   Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    print(f"   Columns: {list(test_df.columns)}")
    
    # Add basic time features (same as our best model)
    print("\n2. Adding basic time features...")
    test_df['hour'] = test_df['datetime'].dt.hour
    test_df['day_of_week'] = test_df['datetime'].dt.dayofweek
    test_df['month'] = test_df['datetime'].dt.month
    test_df['day_of_year'] = test_df['datetime'].dt.dayofyear
    test_df['is_weekend'] = (test_df['day_of_week'] >= 5).astype(int)
    
    # Define feature columns (same as our best model)
    feature_cols = ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend']
    
    # Create test dataset with features
    test_with_features = test_df[['id'] + feature_cols].copy()
    
    # Save the prepared test data
    test_with_features.to_csv('test_kaggle_basic_features.csv', index=False)
    print(f"\n3. Created test_kaggle_basic_features.csv with {len(test_with_features)} rows")
    print(f"   Features: {feature_cols}")
    
    # Show sample of prepared data
    print("\n4. Sample of prepared test data:")
    print(test_with_features.head(10))
    
    print("\n" + "=" * 60)
    print("KAGGLE TEST DATA PREPARED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ Test data loaded (September 3-24, 2024)")
    print("✅ Basic time features added")
    print("✅ Ready for model prediction")
    print("=" * 60)
    
    return test_with_features

if __name__ == "__main__":
    prepare_kaggle_test_data()
