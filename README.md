# Paris Air Quality Prediction - Kaggle Competition

## 🎯 Competition Overview

**Goal:** Predict hourly concentrations of five air pollutants in Paris over a three-week test period using historical air quality data.

**Competition Link:** [X-HEC TS 2025-26: Predicting Air Quality in Paris](https://www.kaggle.com/competitions/x-hec-ts-2025-26-predicting-air-quality-in-paris)

---

## 📊 What Are We Predicting?

You need to predict **ALL FIVE** of these pollutants for every hour over a 3-week period:

1. **CO (Carbon Monoxide)** - mg/m³
   - From vehicle exhaust and incomplete combustion
   
2. **NO₂ (Nitrogen Dioxide)** - µg/m³
   - Primarily from traffic and industrial sources
   
3. **O₃ (Ozone)** - µg/m³
   - Forms from reactions between other pollutants in sunlight
   
4. **PM10 (Particulate Matter ≤10µm)** - µg/m³
   - Coarse dust particles, pollen, mold
   
5. **PM2.5 (Particulate Matter ≤2.5µm)** - µg/m³
   - Fine particles from combustion, can penetrate deep into lungs

**Why all five?** Air pollutants are interconnected - predicting all of them together gives a complete picture of air quality.

---

## 🎯 Evaluation Metric: Mean Absolute Error (MAE)

### How Your Predictions Are Scored

Your model will be evaluated using **Mean Absolute Error (MAE)** - this measures how far off your predictions are from reality, on average.

**Formula:**
```
MAE = (1/n) × Σ |y_i - ŷ_i|
```

Where:
- `n` = number of observations
- `y_i` = actual pollutant level at time i
- `ŷ_i` = your predicted level at time i
- `|...|` = absolute value (distance, ignoring positive/negative)

**In plain English:** Add up all your prediction errors (ignoring whether you guessed too high or too low), then divide by the number of predictions. Lower = better!

### Final Score Calculation

Your **Final Score** is the average MAE across all five pollutants:

```
Final Score = (1/5) × (MAE_CO + MAE_NO₂ + MAE_O₃ + MAE_PM10 + MAE_PM2.5)
```

**Key Points:**
- ✅ **Equal Weighting**: Each pollutant counts for 20% of your score
- ✅ **Per-Pollutant Calculation**: MAE is computed separately for each pollutant
- ✅ **Consistency Matters**: You can't just nail one pollutant - you need to be good at all five!
- ✅ **Lower is Better**: MAE = 0 means perfect predictions (impossible in reality)

### Example Score Calculation

Let's say your model gets these MAE values:
- CO: 0.05 mg/m³
- NO₂: 8.2 µg/m³
- O₃: 12.5 µg/m³
- PM10: 7.8 µg/m³
- PM2.5: 5.3 µg/m³

**Your Final Score = (0.05 + 8.2 + 12.5 + 7.8 + 5.3) / 5 = 6.77**

⚠️ **Important:** Since pollutants have different units and ranges, this average doesn't have a direct physical meaning - it's just for ranking competitors!

---

## 📁 Project Structure

```
paris_air_quality/
│
├── data/
│   ├── train.csv              # Original training data
│   ├── train_cleaned.csv      # Cleaned training data with weather
│   ├── train_simplified.csv   # Simplified training data (time features only)
│   ├── test.csv               # Original test data (timestamps only)
│   ├── test_simplified.csv    # Simplified test data (time features only)
│   ├── weather_data.csv       # Weather data
│   └── sample_submission.csv  # Submission format template
│
├── modeling/
│   ├── flexible_air_quality_model.py  # Main flexible model
│   └── predictions/                   # Generated predictions
│
├── EDA.ipynb                  # Exploratory Data Analysis notebook
└── README.md                  # This file
```

---

## 📈 The Data Explained

### 1. **train.csv** - Your Historical Data
- **Time Period:** January 1, 2020 → September 3, 2024 (~40,000 hours!)
- **Frequency:** Hourly measurements
- **Key columns:**
  - `id`: Timestamp (datetime)
  - `valeur_CO`: Carbon Monoxide concentration
  - `valeur_NO2`: Nitrogen Dioxide concentration
  - `valeur_O3`: Ozone concentration
  - `valeur_PM10`: PM10 concentration
  - `valeur_PM25`: PM2.5 concentration

**Note:** Data may have missing values (sensor malfunctions, maintenance, etc.)

---

### 2. **test.csv** - What You Need to Predict
- **Time Period:** A 3-week period in 2025 (exact dates hidden)
- **What's missing:** All five pollutant values
- **Your Job:** Predict hourly values for all 5 pollutants for ~504 hours (3 weeks × 7 days × 24 hours)

---

### 3. **sample_submission.csv** - Your Answer Template
Shows exactly how to format your predictions.

**Required Format:**
```csv
id,valeur_CO,valeur_NO2,valeur_O3,valeur_PM10,valeur_PM25
0,0.250,25.3,45.2,15.8,8.4
1,0.235,23.1,48.7,14.2,7.9
...
```

**Critical:** Every row must have predictions for all 5 pollutants!

---

## 🎓 Why This Is Challenging

### 1. **Multi-Output Prediction**
You're not predicting just one thing - you need to predict 5 interconnected pollutants simultaneously.

### 2. **Time Series Complexity**
- Hourly patterns (rush hour traffic)
- Daily patterns (day vs. night)
- Weekly patterns (weekdays vs. weekends)
- Seasonal patterns (winter heating, summer ozone)

### 3. **Missing Data**
Sensors fail, maintenance happens - you'll need to handle gaps in the historical data.

### 4. **Pollutant Interactions**
The five pollutants are chemically and physically related:
- High NO₂ (traffic) often means lower O₃ (they react together)
- PM2.5 is often correlated with PM10
- Weather affects all pollutants differently

---

## 🚀 Getting Started

### Quick Start with Flexible Model
```python
from modeling.flexible_air_quality_model import FlexibleAirQualityModel

# Create and train model
model = FlexibleAirQualityModel("my_experiment")
model.train("data/train_simplified.csv")

# Make predictions
model.predict("data/test_simplified.csv", "modeling/predictions/my_submission.csv")
```

### Available Datasets
- **`train_simplified.csv`**: Time features only (6 features)
- **`train_cleaned.csv`**: Time + weather features (12 features)
- **`test_simplified.csv`**: Time features only (6 features)

### Run Experiments
```bash
cd modeling
python flexible_air_quality_model.py
```

---

## 💡 Pro Tips for Success

### 1. **Don't Ignore Any Pollutant**
Since each counts for 20%, a terrible prediction on one pollutant tanks your overall score.

### 2. **Leverage Pollutant Relationships**
- NO₂ and O₃ have inverse relationships
- PM2.5 and PM10 are correlated
- Use other pollutants as features when predicting each one

### 3. **Time-Based Validation Is Critical**
❌ **Wrong:** Random train/test split
✅ **Right:** Train on earlier dates, test on later dates (mimics real prediction task)

### 4. **Handle Missing Data Thoughtfully**
Options:
- Forward fill (use last known value)
- Interpolation (smooth gap between known values)
- Prediction (use other pollutants + time features to fill)

### 5. **Engineer Features Like Crazy**
Good features > complex models. Focus on:
- Temporal patterns
- Lagged values
- Weather interactions
- Pollutant ratios

---

## 📊 Questions to Explore in EDA

- [ ] What's the typical hourly pattern for each pollutant?
- [ ] Do weekends look different from weekdays?
- [ ] Which pollutants are most correlated?
- [ ] Are there seasonal trends?
- [ ] How much missing data do we have?
- [ ] What are the typical ranges for each pollutant?
- [ ] Do all pollutants spike at the same times?
- [ ] How did COVID lockdowns (2020-2021) affect pollution?

---

## 📅 Important Dates

- **Competition Start:** December 16, 2024
- **Submission Opens:** January 6, 2025
- **Final Deadline:** March 24, 2025 (11:59 PM)

---

## ✨ Your Progress Tracker

- [ ] Downloaded and loaded all data files
- [ ] Completed initial EDA for all 5 pollutants
- [ ] Handled missing values
- [ ] Created time-based features
- [ ] Built baseline model (MAE for each pollutant: ___, ___, ___, ___, ___)
- [ ] Engineered advanced features
- [ ] Trained ML model (MAE improved to: ___, ___, ___, ___, ___)
- [ ] Made first Kaggle submission
- [ ] Iterated based on leaderboard feedback
- [ ] Top 10% of leaderboard 🎯

---

## 📚 Useful Resources

- [Competition Discussion Forum](https://www.kaggle.com/competitions/x-hec-ts-2025-26-predicting-air-quality-in-paris/discussion)
- [Competition Data Page](https://www.kaggle.com/competitions/x-hec-ts-2025-26-predicting-air-quality-in-paris/data)
- [Time Series Forecasting Guide](https://www.kaggle.com/learn/time-series)

---

**Good luck! Remember: consistency across all five pollutants is key! 🍀**
