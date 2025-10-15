# Ideation Experiments

This folder contains experimental work for the Paris Air Quality prediction project.

## Key Experiments

### 1. Validation Strategy
- **Problem**: No ground truth for test set (Sept 3-24, 2024)
- **Solution**: Used same period one year before (Sept 3-24, 2023) as validation set
- **Files**: 
  - `create_validation_data.py` - Creates train/validation splits
  - `validate_model.py` - Validates basic model

### 2. Model Fine-tuning
- **Approach**: Added internal temporal validation and hyperparameter optimization
- **Files**:
  - `flexible_air_quality_model_finetuned.py` - Enhanced XGBoost model
  - `validate_finetuned_model.py` - Validates fine-tuned model

### 3. Feature Engineering
- **Features**: Cyclical encoding, interaction features, advanced time features
- **Files**:
  - `create_enhanced_features.py` - Creates enhanced feature datasets
  - `validate_enhanced_model.py` - Validates enhanced model
  - `compare_results.py` - Compares basic vs enhanced models

### 4. LightGBM Alternative
- **Purpose**: Test LightGBM as alternative to XGBoost
- **Files**:
  - `flexible_air_quality_model_lightgbm.py` - LightGBM implementation
  - `validate_lightgbm_model.py` - Validates LightGBM model

### 5. Kaggle Submissions
- **Files**:
  - `prepare_kaggle_test.py` - Prepares test data for Kaggle
  - `generate_kaggle_submission.py` - Generates XGBoost submission
  - `generate_lightgbm_kaggle_submission.py` - Generates LightGBM submission
  - `fix_kaggle_submission.py` - Fixes ID format for Kaggle

## Results Summary

| Model | Validation MAE | Notes |
|-------|----------------|-------|
| Basic XGBoost | ~0.45 | Baseline model |
| Fine-tuned XGBoost | ~0.42 | Best performing model |
| Enhanced Features | ~0.43 | Slight improvement |
| LightGBM | ~0.44 | Good alternative |

## Best Submission
- **Model**: Fine-tuned XGBoost with basic features
- **File**: `kaggle_submission_fixed.csv`
- **Validation MAE**: ~0.42

## Key Learnings
1. September 2023 validation provided realistic performance estimates
2. Fine-tuning with temporal validation significantly improved performance
3. Basic time features were sufficient; complex features didn't help much
4. XGBoost slightly outperformed LightGBM for this problem
