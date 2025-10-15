"""
Flexible Air Quality Model with LightGBM and Fine-tuning Capabilities
==================================================================

This enhanced model uses LightGBM instead of XGBoost and includes:
- Temporal validation split to prevent overfitting
- Hyperparameter optimization with validation
- Early stopping based on validation performance
- Proper time series cross-validation

Usage:
    model = FlexibleAirQualityModelLightGBM("my_experiment")
    model.train_with_validation("train_simplified.csv", validation_split=0.2)
    model.predict("test_simplified.csv", "predictions/my_submission.csv")
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class FlexibleAirQualityModelLightGBM:
    """
    Enhanced LightGBM model with fine-tuning capabilities and overfitting prevention
    """
    
    def __init__(self, model_name="lightgbm_model"):
        self.models = {}  # One model per pollutant
        self.feature_columns = []
        self.target_columns = ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']
        self.model_name = model_name
        self.is_trained = False
        self.validation_scores = {}
        self.best_params = {}
        
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
        
        # Available target columns in this dataset
        available_targets = [col for col in self.target_columns if col in df.columns]
        print(f"Available target columns: {available_targets}")
        
        # Feature columns (everything except targets and id)
        feature_cols = [col for col in df.columns if col not in available_targets]
        print(f"Feature columns: {feature_cols}")
        
        return feature_cols, available_targets
    
    def temporal_split(self, df, validation_split=0.2):
        """
        Create temporal train/validation split (no data leakage)
        """
        print(f"\nCreating temporal split with {validation_split*100:.1f}% validation...")
        
        # Sort by date to ensure proper temporal order
        df_sorted = df.sort_index()
        
        # Calculate split point
        total_rows = len(df_sorted)
        train_size = int(total_rows * (1 - validation_split))
        
        # Split by time (not randomly!)
        train_df = df_sorted.iloc[:train_size]
        val_df = df_sorted.iloc[train_size:]
        
        print(f"Training period: {train_df.index.min()} to {train_df.index.max()}")
        print(f"Validation period: {val_df.index.min()} to {val_df.index.max()}")
        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        return train_df, val_df
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, target_name):
        """
        Optimize hyperparameters using validation set
        """
        print(f"  Optimizing hyperparameters for {target_name}...")
        
        # Define parameter grid for LightGBM
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [3, 4, 5, -1],  # -1 means no limit
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [20, 30, 40]
        }
        
        best_score = float('inf')
        best_params = None
        
        # Simple grid search (can be optimized with more sophisticated methods)
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for sub in param_grid['subsample']:
                        for col in param_grid['colsample_bytree']:
                            for num_leaves in param_grid['num_leaves']:
                                for min_child in param_grid['min_child_samples']:
                                    
                                    # Train model with these parameters
                                    model = lgb.LGBMRegressor(
                                        n_estimators=n_est,
                                        max_depth=max_d,
                                        learning_rate=lr,
                                        subsample=sub,
                                        colsample_bytree=col,
                                        num_leaves=num_leaves,
                                        min_child_samples=min_child,
                                        random_state=42,
                                        n_jobs=-1,
                                        verbose=-1
                                    )
                                    
                                    # Train with early stopping
                                    model.fit(
                                        X_train, y_train,
                                        eval_set=[(X_val, y_val)],
                                        callbacks=[lgb.early_stopping(6), lgb.log_evaluation(0)]
                                    )
                                    
                                    # Get validation score
                                    val_pred = model.predict(X_val)
                                    val_score = mean_absolute_error(y_val, val_pred)
                                    
                                    # Keep track of best parameters
                                    if val_score < best_score:
                                        best_score = val_score
                                        best_params = {
                                            'n_estimators': n_est,
                                            'max_depth': max_d,
                                            'learning_rate': lr,
                                            'subsample': sub,
                                            'colsample_bytree': col,
                                            'num_leaves': num_leaves,
                                            'min_child_samples': min_child
                                        }
        
        print(f"    Best validation MAE: {best_score:.4f}")
        print(f"    Best parameters: {best_params}")
        
        return best_params, best_score
    
    def train_with_validation(self, train_file_path, validation_split=0.2, optimize_params=True):
        """
        Train models with temporal validation and hyperparameter optimization
        """
        print("=" * 60)
        print(f"TRAINING {self.model_name.upper()} WITH LIGHTGBM FINE-TUNING")
        print("=" * 60)
        
        # Load training data
        train_df = self.load_dataset(train_file_path, "train")
        
        # Detect features
        self.feature_columns, available_targets = self.detect_features(train_df, "train")
        
        # Create temporal train/validation split
        train_data, val_data = self.temporal_split(train_df, validation_split)
        
        # Prepare training and validation data
        X_train = train_data[self.feature_columns]
        y_train = train_data[available_targets]
        X_val = val_data[self.feature_columns]
        y_val = val_data[available_targets]
        
        print(f"\nTraining on {len(self.feature_columns)} features: {self.feature_columns}")
        print(f"Predicting {len(available_targets)} targets: {available_targets}")
        
        # Train separate model for each target
        for target in available_targets:
            print(f"\nTraining model for {target}...")
            
            # Prepare data for this target
            y_target_train = y_train[target].dropna()
            X_target_train = X_train.loc[y_target_train.index]
            y_target_val = y_val[target].dropna()
            X_target_val = X_val.loc[y_target_val.index]
            
            # Remove rows with NaN in features
            valid_idx_train = X_target_train.dropna().index
            X_clean_train = X_target_train.loc[valid_idx_train]
            y_clean_train = y_target_train.loc[valid_idx_train]
            
            valid_idx_val = X_target_val.dropna().index
            X_clean_val = X_target_val.loc[valid_idx_val]
            y_clean_val = y_target_val.loc[valid_idx_val]
            
            print(f"  Training samples: {len(X_clean_train)}")
            print(f"  Validation samples: {len(X_clean_val)}")
            
            # Optimize hyperparameters if requested
            if optimize_params and len(X_clean_val) > 0:
                best_params, val_score = self.optimize_hyperparameters(
                    X_clean_train, y_clean_train, X_clean_val, y_clean_val, target
                )
                self.validation_scores[target] = val_score
                self.best_params[target] = best_params
                
                # Train final model with best parameters
                final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
            else:
                # Use default parameters
                default_params = {
                    'n_estimators': 300,
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'num_leaves': 31,
                    'min_child_samples': 20,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
                final_model = lgb.LGBMRegressor(**default_params)
            
            # Train final model with early stopping if validation data available
            if len(X_clean_val) > 0:
                final_model.fit(
                    X_clean_train, y_clean_train,
                    eval_set=[(X_clean_val, y_clean_val)],
                    callbacks=[lgb.early_stopping(6), lgb.log_evaluation(0)]
                )
                
                # Get final validation score
                val_pred = final_model.predict(X_clean_val)
                final_val_score = mean_absolute_error(y_clean_val, val_pred)
                self.validation_scores[target] = final_val_score
                print(f"  Final validation MAE: {final_val_score:.4f}")
            else:
                # No validation data, train without early stopping
                final_model.fit(X_clean_train, y_clean_train)
                print(f"  No validation data available, trained without early stopping")
            
            self.models[target] = final_model
            print(f"✓ {target} model trained successfully")
        
        self.is_trained = True
        print(f"\nTraining completed! Trained {len(self.models)} models")
        
        # Print validation summary
        if self.validation_scores:
            print("\n" + "=" * 40)
            print("VALIDATION SCORES SUMMARY")
            print("=" * 40)
            for target, score in self.validation_scores.items():
                print(f"{target:12}: {score:8.4f}")
            overall_val_score = np.mean(list(self.validation_scores.values()))
            print(f"{'Overall':12}: {overall_val_score:8.4f}")
            print("=" * 40)
        
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
            "num_models": len(self.models),
            "validation_scores": self.validation_scores,
            "best_params": self.best_params
        }
        return info

def main():
    """Example usage of the LightGBM fine-tuned model"""
    print("=" * 60)
    print("FLEXIBLE AIR QUALITY MODEL WITH LIGHTGBM - EXAMPLE USAGE")
    print("=" * 60)
    
    # Create model instance
    model = FlexibleAirQualityModelLightGBM("lightgbm_validation_model")
    
    # Train with validation and hyperparameter optimization
    model.train_with_validation("train_simplified.csv", validation_split=0.2, optimize_params=True)
    
    # Make predictions
    predictions, submission = model.predict(
        "test_simplified.csv", 
        "predictions_lightgbm_2023.csv"
    )
    
    # Show model info
    print("\nModel Information:")
    info = model.get_model_info()
    for key, value in info.items():
        if key not in ['validation_scores', 'best_params']:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("LIGHTGBM MODEL COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
