# Air Quality Modeling

This directory contains the flexible modeling framework for air quality prediction.

## 📁 Structure

```
modeling/
├── flexible_air_quality_model.py    # Main flexible model
├── predictions/                     # Generated predictions
│   ├── experiment_time_only.csv     # Time features only
│   ├── experiment_with_weather.csv  # With weather features
│   └── simplified_time_submission.csv # Main submission
└── README.md                       # This file
```

## 🚀 Quick Start

### Basic Usage
```python
from flexible_air_quality_model import FlexibleAirQualityModel

# Create model
model = FlexibleAirQualityModel("my_experiment")

# Train on any dataset
model.train("../data/train_simplified.csv")

# Predict on any test dataset
model.predict("../data/test_simplified.csv", "../modeling/predictions/my_results.csv")
```

### Run Experiments
```python
# Test different approaches
python flexible_air_quality_model.py
```

## 📊 Available Datasets

### Training Datasets
- `../data/train_simplified.csv` - Time features only (6 features)
- `../data/train_cleaned.csv` - Time + weather features (12 features)

### Test Datasets
- `../data/test_simplified.csv` - Time features only (6 features)
- `../data/test.csv` - Timestamps only (1 feature)

## 🎯 Model Features

- **Auto-detection**: Automatically finds features and targets
- **Flexible**: Works with any dataset structure
- **Robust**: Handles missing features gracefully
- **Realistic**: No lag features (proper time series approach)
- **Ready for Kaggle**: Creates proper submission format

## 📈 Generated Predictions

- `simplified_time_submission.csv` - Main submission (time features only)
- `experiment_time_only.csv` - Time features experiment
- `experiment_with_weather.csv` - Weather features experiment

## 🔧 Customization

The model automatically adapts to any dataset structure. Just provide:
- Training data with target columns
- Test data with matching feature columns
- The model handles the rest!
