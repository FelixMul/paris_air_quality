import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexible_air_quality_model_finetuned import FlexibleAirQualityModelFinetuned
from sklearn.metrics import mean_absolute_error

def validate_enhanced_model():
    """Train fine-tuned model with enhanced features and validate on September 2023 data"""
    
    print("=" * 60)
    print("VALIDATING ENHANCED MODEL WITH FEATURE ENGINEERING")
    print("=" * 60)
    print("Features: 22 (vs 5 in original)")
    print("- Cyclical encoding: hour, day_of_week, month")
    print("- Interaction features: hour×day, hour×month, etc.")
    print("- Advanced time features: Monday effect, end of month, etc.")
    print("=" * 60)
    
    # Create fine-tuned model for enhanced features
    model = FlexibleAirQualityModelFinetuned("enhanced_feature_model")
    
    # Train with validation and hyperparameter optimization
    print("\n1. Training enhanced model with validation...")
    print("   - 22 features (vs 5 in original)")
    print("   - Temporal validation split (20% of training data)")
    print("   - Hyperparameter optimization")
    print("   - Early stopping to prevent overfitting")
    model.train_with_validation("train_enhanced.csv", validation_split=0.2, optimize_params=True)
    
    # Make predictions on enhanced test data
    print("\n2. Making predictions on September 2023 data...")
    print("   (Sept 3-24, 2023 with known results)")
    predictions, _ = model.predict(
        "test_enhanced.csv", 
        "predictions_enhanced_2023.csv"
    )
    
    # Load actual results
    print("\n3. Loading actual results for comparison...")
    actual = pd.read_csv("test_enhanced_with_results.csv")
    
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
    print("ENHANCED MODEL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Overall MAE Score: {overall_score:.4f}")
    print("\nThis represents your estimated accuracy on the actual test set")
    print("(September 3-24, 2024) with enhanced features and fine-tuning.")
    print("=" * 60)
    
    # Show model information
    print("\nModel Information:")
    info = model.get_model_info()
    print(f"  Model name: {info['model_name']}")
    print(f"  Features used: {len(info['feature_columns'])} (vs 5 in original)")
    print(f"  Models trained: {info['num_models']}")
    
    if info['validation_scores']:
        print("\nInternal validation scores (from training):")
        for target, score in info['validation_scores'].items():
            print(f"  {target:12}: {score:8.4f}")
    
    # Show feature importance (if available)
    print(f"\nFeature columns used:")
    for i, feature in enumerate(info['feature_columns'], 1):
        print(f"  {i:2d}. {feature}")
    
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
    mae_scores, overall_score = validate_enhanced_model()
