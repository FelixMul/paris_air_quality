"""
Flexible Air Quality Model - Works with Any Dataset
==================================================

This script provides a flexible model that can train and test on any dataset
you provide. Perfect for iterative feature engineering experiments.

Usage:
    model = FlexibleAirQualityModel()
    model.train("data/train_simplified.csv")
    model.predict("data/test_simplified.csv", "predictions/my_submission.csv")
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class FlexibleAirQualityModel:
    """
    Flexible XGBoost model that can work with any dataset structure
    """
    
    def __init__(self, model_name="flexible_model"):
        self.models = {}  # One model per pollutant
        self.feature_columns = []
        self.target_columns = ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']
        self.model_name = model_name
        self.is_trained = False
        
    def load_dataset(self, file_path, dataset_type="train"):
        """Load any dataset and automatically detect features"""
        print(f"Loading {dataset_type} dataset: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        df['id'] = pd.to_datetime(df['id'])
        df = df.set_index('id')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Period: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def detect_features(self, df, dataset_type="train"):
        """Automatically detect feature columns vs target columns"""
        print(f"Detecting features for {dataset_type} dataset...")
        
        # Find target columns that exist in the dataset
        available_targets = [col for col in self.target_columns if col in df.columns]
        
        # Find feature columns (everything except targets and id)
        feature_cols = [col for col in df.columns if col not in self.target_columns]
        
        print(f"Available target columns: {available_targets}")
        print(f"Feature columns: {feature_cols}")
        
        return feature_cols, available_targets
    
    def train(self, train_file_path, xgb_params=None):
        """Train models on any training dataset"""
        print("=" * 60)
        print(f"TRAINING {self.model_name.upper()}")
        print("=" * 60)
        
        # Load training data
        train_df = self.load_dataset(train_file_path, "train")
        
        # Detect features
        self.feature_columns, available_targets = self.detect_features(train_df, "train")
        
        # Prepare training data
        X_train = train_df[self.feature_columns]
        y_train = train_df[available_targets]
        
        print(f"Training on {len(self.feature_columns)} features: {self.feature_columns}")
        print(f"Predicting {len(available_targets)} targets: {available_targets}")
        
        # Default XGBoost parameters
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 300,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mae'
            }
        
        # Train separate model for each target
        for target in available_targets:
            print(f"Training model for {target}...")
            
            # Prepare data for this target
            y_target = y_train[target].dropna()
            X_target = X_train.loc[y_target.index]
            
            # Remove rows with NaN in features
            valid_idx = X_target.dropna().index
            X_clean = X_target.loc[valid_idx]
            y_clean = y_target.loc[valid_idx]
            
            # Train model
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_clean, y_clean)
            
            self.models[target] = model
            print(f"✓ {target} model trained on {len(X_clean)} samples")
        
        self.is_trained = True
        print(f"\nTraining completed! Trained {len(self.models)} models")
        
        return self
    
    def predict(self, test_file_path, output_file_path=None):
        """Make predictions on any test dataset"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        print("=" * 60)
        print(f"MAKING PREDICTIONS WITH {self.model_name.upper()}")
        print("=" * 60)
        
        # Load test data
        test_df = self.load_dataset(test_file_path, "test")
        
        # Check if test data has the required features
        missing_features = [col for col in self.feature_columns if col not in test_df.columns]
        if missing_features:
            print(f"Warning: Missing features in test data: {missing_features}")
            print("Adding missing features with default values...")
            
            # Add missing features with default values
            for feature in missing_features:
                test_df[feature] = 0.0
                print(f"  Added {feature} = 0.0")
        
        # Prepare test features
        X_test = test_df[self.feature_columns]
        
        print(f"Making predictions using {len(self.feature_columns)} features")
        
        # Make predictions
        predictions = pd.DataFrame(index=test_df.index)
        
        for target, model in self.models.items():
            print(f"Predicting {target}...")
            pred = model.predict(X_test)
            predictions[target] = pred
        
        # Create submission file
        if output_file_path:
            submission = self.create_submission(predictions, output_file_path)
            return predictions, submission
        else:
            return predictions
    
    def create_submission(self, predictions, output_path):
        """Create submission file in the required format"""
        print(f"Creating submission file: {output_path}")
        
        # Ensure we have all required columns
        required_cols = ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']
        
        # Create submission DataFrame
        submission = pd.DataFrame(index=predictions.index)
        
        for col in required_cols:
            if col in predictions.columns:
                submission[col] = predictions[col]
            else:
                # If column is missing, fill with 0 (fallback)
                submission[col] = 0.0
                print(f"Warning: {col} not found in predictions, filled with 0")
        
        # Reset index to get id column
        submission = submission.reset_index()
        submission.columns = ['id'] + required_cols
        
        # Save to file
        submission.to_csv(output_path, index=False)
        print(f"Submission file saved: {output_path}")
        print(f"Submission shape: {submission.shape}")
        
        return submission
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return "Model not trained yet"
        
        info = {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "feature_columns": self.feature_columns,
            "trained_targets": list(self.models.keys()),
            "num_features": len(self.feature_columns),
            "num_models": len(self.models)
        }
        return info

def main():
    """Example usage of the flexible model"""
    print("=" * 60)
    print("FLEXIBLE AIR QUALITY MODEL - EXAMPLE USAGE")
    print("=" * 60)
    
    # Create model instance
    model = FlexibleAirQualityModel("simplified_time_model")
    
    # Train on simplified dataset
    model.train("../data/train_simplified.csv")
    
    # Make predictions
    predictions, submission = model.predict(
        "../data/test_simplified.csv", 
        "../modeling/predictions/simplified_time_submission.csv"
    )
    
    # Show model info
    print("\nModel Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("FLEXIBLE MODEL COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
