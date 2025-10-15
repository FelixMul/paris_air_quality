"""
Generate LightGBM Kaggle Submission
==================================

This script uses the LightGBM model to generate predictions for the actual Kaggle test set.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexible_air_quality_model_lightgbm import FlexibleAirQualityModelLightGBM

def generate_lightgbm_kaggle_submission():
    """Generate final Kaggle submission using LightGBM model"""
    
    print("=" * 60)
    print("GENERATING LIGHTGBM KAGGLE SUBMISSION")
    print("=" * 60)
    print("Model: LightGBM with fine-tuning")
    print("Expected MAE: 9.1368 (based on validation)")
    print("=" * 60)
    
    # Create the LightGBM model
    model = FlexibleAirQualityModelLightGBM("lightgbm_kaggle_submission")
    
    # Train with validation and hyperparameter optimization
    print("\n1. Training LightGBM model on ideation training data...")
    print("   - Using basic features (5 features)")
    print("   - LightGBM with hyperparameter optimization")
    print("   - Temporal validation split")
    model.train_with_validation("train_simplified.csv", validation_split=0.2, optimize_params=True)
    
    # Make predictions on actual test data
    print("\n2. Making predictions on actual Kaggle test data...")
    print("   (September 3-24, 2024)")
    predictions, submission = model.predict(
        "test_kaggle_basic_features.csv", 
        "kaggle_submission_lightgbm.csv"
    )
    
    # Fix the id format for Kaggle
    print("\n3. Fixing submission format for Kaggle...")
    submission['id'] = pd.to_datetime(submission['id']).dt.strftime('%Y-%m-%d %H')
    
    # Save the fixed submission
    submission.to_csv('kaggle_submission_lightgbm_fixed.csv', index=False)
    
    # Show prediction summary
    print("\n4. Prediction Summary:")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Date range: {predictions.index.min()} to {predictions.index.max()}")
    
    # Show sample predictions
    print("\n5. Sample predictions:")
    sample_predictions = predictions.head(10)
    for target in ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']:
        print(f"   {target}: {sample_predictions[target].mean():.4f} (avg)")
    
    # Show prediction statistics
    print("\n6. Prediction Statistics:")
    for target in ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']:
        pred_values = predictions[target]
        print(f"   {target:12}: min={pred_values.min():6.2f}, max={pred_values.max():6.2f}, mean={pred_values.mean():6.2f}")
    
    # Show model information
    print("\n7. Model Information:")
    info = model.get_model_info()
    print(f"   Model name: {info['model_name']}")
    print(f"   Features used: {len(info['feature_columns'])}")
    print(f"   Models trained: {info['num_models']}")
    
    if info['validation_scores']:
        print("\n   Internal validation scores:")
        for target, score in info['validation_scores'].items():
            print(f"     {target:12}: {score:8.4f}")
    
    print("\n" + "=" * 60)
    print("LIGHTGBM KAGGLE SUBMISSION GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ Model trained on ideation data")
    print("✅ Predictions made on actual test set")
    print("✅ Submission file created: kaggle_submission_lightgbm_fixed.csv")
    print("✅ Ready for Kaggle upload")
    print("=" * 60)
    
    # Show submission file info
    print(f"\nSubmission file details:")
    print(f"   File: kaggle_submission_lightgbm_fixed.csv")
    print(f"   Rows: {len(submission)}")
    print(f"   Columns: {list(submission.columns)}")
    
    # Show first few rows of submission
    print(f"\nFirst 5 rows of submission:")
    print(submission.head().to_string(index=False))
    
    return submission

if __name__ == "__main__":
    submission = generate_lightgbm_kaggle_submission()
