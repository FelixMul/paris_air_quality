import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexible_air_quality_model_lightgbm import FlexibleAirQualityModelLightGBM
from sklearn.metrics import mean_absolute_error

def validate_lightgbm_model():
    """Train LightGBM model and validate on September 2023 data"""
    
    print("=" * 60)
    print("VALIDATING LIGHTGBM MODEL ON SEPTEMBER 2023 DATA")
    print("=" * 60)
    
    # Create LightGBM model
    model = FlexibleAirQualityModelLightGBM("lightgbm_validation_model")
    
    # Train with validation and hyperparameter optimization
    print("\n1. Training LightGBM model with validation...")
    print("   - Using basic features (5 features)")
    print("   - LightGBM with hyperparameter optimization")
    print("   - Temporal validation split")
    print("   - Early stopping to prevent overfitting")
    model.train_with_validation("train_simplified.csv", validation_split=0.2, optimize_params=True)
    
    # Make predictions on ideation test data
    print("\n2. Making predictions on September 2023 data...")
    print("   (Sept 3-24, 2023 with known results)")
    predictions, _ = model.predict(
        "test_simplified.csv", 
        "predictions_lightgbm_2023.csv"
    )
    
    # Load actual results
    print("\n3. Loading actual results for comparison...")
    actual = pd.read_csv("test_simplified_with_results.csv")
    
    # Calculate MAE for each pollutant
    print("\n4. Calculating MAE scores...")
    mae_scores = {}
    
    for target in ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']:
        # Handle missing values - align indices properly
        actual_clean = actual[target].dropna()
        pred_clean = predictions[target].iloc[actual_clean.index]
        
        mae = mean_absolute_error(actual_clean, pred_clean)
        mae_scores[target] = mae
        
        print(f"   MAE for {target:12}: {mae:8.4f}")
    
    # Calculate overall score (average MAE)
    overall_score = np.mean(list(mae_scores.values()))
    
    print("\n" + "=" * 60)
    print("LIGHTGBM MODEL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Overall MAE Score: {overall_score:.4f}")
    print("\nThis represents your estimated accuracy on the actual test set")
    print("(September 3-24, 2024) with LightGBM and fine-tuning.")
    print("=" * 60)
    
    # Show model information
    print("\nModel Information:")
    info = model.get_model_info()
    print(f"  Model name: {info['model_name']}")
    print(f"  Features used: {len(info['feature_columns'])}")
    print(f"  Models trained: {info['num_models']}")
    
    if info['validation_scores']:
        print("\nInternal validation scores (from training):")
        for target, score in info['validation_scores'].items():
            print(f"  {target:12}: {score:8.4f}")
    
    # Show some sample predictions vs actual
    print("\nSample predictions vs actual (first 10 rows):")
    sample_data = pd.DataFrame({
        'id': actual['id'].head(10),
        'actual_CO': actual['valeur_CO'].head(10),
        'predicted_CO': predictions['valeur_CO'].head(10),
        'actual_NO2': actual['valeur_NO2'].head(10),
        'predicted_NO2': predictions['valeur_NO2'].head(10)
    })
    print(sample_data.to_string(index=False))
    
    return mae_scores, overall_score

if __name__ == "__main__":
    mae_scores, overall_score = validate_lightgbm_model()
