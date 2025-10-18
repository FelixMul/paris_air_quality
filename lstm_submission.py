import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import holidays
import os
from datetime import datetime
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- 0. Configuration and Constants ---
# Model Hyperparameters
TIMESTEPS = 24
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 20 # Final number of epochs for training on full data

# Feature Engineering Configuration
# These flags control train-test consistency to avoid overfitting:

USE_LAGGED_FEATURES = False  
# False: Cleaner approach - LSTM learns from sequence history without explicit lags
#        Avoids train-test mismatch (train has clean lags, test has noisy predicted lags)
# True:  Uses lag_1, lag_24, lag_168 features but risks overfitting

USE_ACTUAL_WEATHER = False   
# False: Uses climatology (day-of-year + hour averages) for both train & test
#        Better consistency when actual weather forecasts aren't available
# True:  Uses actual weather values (better if you have perfect forecasts)

# Feature Summary (always included):
# - Cyclical time encodings: hour_sin/cos, dayofweek_sin/cos, month_sin/cos
# - Rolling features: 6h, 24h, 168h rolling averages for all pollutants
# - Weather: temperature, humidity, wind_speed, precipitation, wind_direction, radiation
# - Temporal: hour, dayofweek, dayofyear, month, is_weekend, is_holiday

# Column Definitions
TARGET_COLS = ['valeur_NO2', 'valeur_CO', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']
WEATHER_COLS = ['temperature', 'relative_humidity', 'wind_speed', 'precipitation', 'wind_direction', 'shortwave_radiation']

# --- 1. Helper Functions for Data Preparation ---

def create_deterministic_features(df):
    """Creates time-based and holiday features with cyclical encodings."""
    # Linear time features
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclical encodings - capture circular nature of time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add holiday feature
    years = df.index.year.unique()
    fr_holidays = holidays.country_holidays('FR', years=years)
    df['is_holiday'] = df.index.normalize().isin(fr_holidays).astype(int)
    return df

def create_climatology_features(df_full, df_train_portion):
    """Creates local climatology weather features based on the training data portion.
    
    If USE_ACTUAL_WEATHER is False, replaces actual weather columns with climatology
    for better train-test consistency.
    """
    print("Creating local climatology weather features...")
    climatology_map = df_train_portion.groupby(['dayofyear', 'hour'])[WEATHER_COLS].mean().reset_index()
    
    df_full = pd.merge(df_full.reset_index(), climatology_map, on=['dayofyear', 'hour'], how='left', suffixes=('', '_climatology'))
    df_full = df_full.set_index('id').sort_index()
    
    climatology_cols = [f'{col}_climatology' for col in WEATHER_COLS]
    
    # Fill any potential NaNs for test set combinations not seen in train
    for col in climatology_cols:
        global_mean = df_train_portion[col.replace('_climatology', '')].mean()
        df_full[col].fillna(global_mean, inplace=True)
    
    # If we want to use only climatology (not actual weather), replace the original columns
    if not USE_ACTUAL_WEATHER:
        print("Replacing actual weather with climatology for train-test consistency...")
        for i, col in enumerate(WEATHER_COLS):
            df_full[col] = df_full[climatology_cols[i]]
        # Drop the _climatology columns as they're now redundant
        df_full.drop(columns=climatology_cols, inplace=True)
        
    print("Climatology features created.")
    return df_full

def create_lagged_features(df):
    """Creates lagged pollutant features if USE_LAGGED_FEATURES is True."""
    if not USE_LAGGED_FEATURES:
        print("Skipping lagged features (USE_LAGGED_FEATURES=False)")
        return df
    
    print("Creating lagged pollutant features for training data...")
    df_lagged = df.copy()
    for lag in [1, 24, 168]:
        for col in TARGET_COLS:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    return df_lagged

def create_rolling_features(df):
    """Creates rolling window features for pollutants.
    
    Rolling features provide smoothed temporal context that is more stable
    than point lags, especially for forecasting.
    """
    print("Creating rolling window features...")
    df_rolling = df.copy()
    
    for col in TARGET_COLS:
        # 6-hour rolling average (captures recent short-term trend)
        df_rolling[f'{col}_roll_6h'] = df_rolling[col].rolling(window=6, min_periods=1).mean()
        
        # 24-hour rolling average (captures daily pattern)
        df_rolling[f'{col}_roll_24h'] = df_rolling[col].rolling(window=24, min_periods=1).mean()
        
        # 168-hour (1 week) rolling average (captures weekly pattern)
        df_rolling[f'{col}_roll_168h'] = df_rolling[col].rolling(window=168, min_periods=1).mean()
    
    print(f"Rolling features created: 3 windows × {len(TARGET_COLS)} pollutants = {3*len(TARGET_COLS)} features")
    return df_rolling

def create_sequences(X, y, timesteps):
    """Reshapes flat data into sequences for an LSTM."""
    X_out, y_out = [], []
    for i in range(len(X) - timesteps):
        X_out.append(X[i:(i + timesteps)])
        y_out.append(y[i + timesteps])
    return np.array(X_out), np.array(y_out)


# --- 2. Helper Functions for Modeling and Prediction ---

def build_and_train_model(X_train_reshaped, y_train_reshaped, sample_weights, y_scaler):
    """Builds, compiles, and trains the final LSTM model.
    
    Tracks both scaled and unscaled MAE for better interpretability.
    """
    print("Building and training the final model...")
    model = Sequential([
        LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        Dropout(0.2),
        Dense(5, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mae')
    
    # verbose=1 provides the training progress bar
    history = model.fit(
        X_train_reshaped, 
        y_train_reshaped, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        sample_weight=sample_weights,
        verbose=1 
    )
    
    # Calculate unscaled MAE for the last epoch to show real-world performance
    print("\nCalculating unscaled MAE on training data (last epoch)...")
    y_pred_scaled = model.predict(X_train_reshaped, verbose=0)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled)
    y_true_unscaled = y_scaler.inverse_transform(y_train_reshaped)
    
    unscaled_mae = np.mean(np.abs(y_true_unscaled - y_pred_unscaled))
    per_pollutant_mae = np.mean(np.abs(y_true_unscaled - y_pred_unscaled), axis=0)
    
    print(f"\n{'='*60}")
    print(f"TRAINING PERFORMANCE (on original scale):")
    print(f"{'='*60}")
    print(f"Overall MAE (unscaled): {unscaled_mae:.4f}")
    print(f"\nPer-Pollutant MAE:")
    for i, col in enumerate(TARGET_COLS):
        print(f"  {col:15s}: {per_pollutant_mae[i]:.4f}")
    print(f"{'='*60}\n")
    
    # Plot training loss (scaled values)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training MAE (scaled)')
    plt.title('Final Model Training Loss (Scaled Values)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE) - Scaled [0,1]')
    plt.legend()
    plt.grid(True)
    os.makedirs('graphs', exist_ok=True)
    plt.savefig('graphs/final_model_training_loss.png')
    plt.show()

    return model

def run_recursive_forecast(model, x_scaler, y_scaler, history_buffer, future_scaffold, full_history_df=None):
    """Generates the full 3-week forecast.
    
    If USE_LAGGED_FEATURES=True, uses historical lags when available.
    Rolling features are always recalculated from the full history to ensure accuracy.
    """
    if USE_LAGGED_FEATURES:
        print("Starting recursive forecast with historical lag preservation...")
    else:
        print("Starting forecast without lagged features (using rolling features)...")
    
    predictions = []
    features_to_drop = TARGET_COLS + WEATHER_COLS
    
    # Maintain combined history for both rolling and lag calculations
    if full_history_df is not None:
        combined_history = full_history_df.copy()
    else:
        # Initialize with just the buffer's target columns
        combined_history = history_buffer[TARGET_COLS].copy()

    for i in tqdm(range(len(future_scaffold)), desc="Forecasting test set"):
        # Prepare input for the LSTM
        input_sequence = history_buffer.tail(TIMESTEPS)
        X_history = input_sequence.drop(columns=features_to_drop)
        # Ensure same column order/names as during scaler fit
        X_history = X_history.reindex(columns=x_scaler.feature_names_in_)
        X_history_scaled = x_scaler.transform(X_history)
        X_input = np.reshape(X_history_scaled, (1, TIMESTEPS, X_history_scaled.shape[1]))
        
        # Predict one step ahead
        pred_scaled = model.predict(X_input, verbose=0)
        pred_inversed = y_scaler.inverse_transform(pred_scaled)
        predictions.append(pred_inversed[0])
        
        # --- Update History Buffer for the Next Prediction ---
        # 1. Get known future features from the scaffold
        next_features = future_scaffold.iloc[[i]]
        
        # 2. Combine with our new prediction
        new_pred_row = pd.DataFrame(pred_inversed, index=[next_features.index[0]], columns=TARGET_COLS)
        new_row_combined = pd.concat([new_pred_row, next_features], axis=1)
        
        # 3. Add prediction to combined history
        combined_history = pd.concat([combined_history, new_pred_row])
        
        # 4. Calculate rolling features from the full history (including new prediction)
        current_timestamp = next_features.index[0]
        for col in TARGET_COLS:
            # Get the historical window for rolling calculations
            # 6-hour rolling
            if len(combined_history) >= 6:
                window_6h = combined_history[col].iloc[-6:]
                new_row_combined[f'{col}_roll_6h'] = window_6h.mean()
            else:
                new_row_combined[f'{col}_roll_6h'] = combined_history[col].mean()
            
            # 24-hour rolling
            if len(combined_history) >= 24:
                window_24h = combined_history[col].iloc[-24:]
                new_row_combined[f'{col}_roll_24h'] = window_24h.mean()
            else:
                new_row_combined[f'{col}_roll_24h'] = combined_history[col].mean()
            
            # 168-hour rolling
            if len(combined_history) >= 168:
                window_168h = combined_history[col].iloc[-168:]
                new_row_combined[f'{col}_roll_168h'] = window_168h.mean()
            else:
                new_row_combined[f'{col}_roll_168h'] = combined_history[col].mean()
        
        # 5. Calculate lagged features if enabled
        if USE_LAGGED_FEATURES:
            for lag in [1, 24, 168]:
                lag_timestamp = current_timestamp - pd.Timedelta(hours=lag)
                # Use actual values from combined_history (prioritizes real data over predictions)
                if lag_timestamp in combined_history.index:
                    for col in TARGET_COLS:
                        new_row_combined[f'{col}_lag_{lag}'] = combined_history.loc[lag_timestamp, col]
                else:
                    # Fallback: use the oldest available data
                    for col in TARGET_COLS:
                        new_row_combined[f'{col}_lag_{lag}'] = combined_history[col].iloc[-1 - lag]
        
        # 6. Append the new row and drop the oldest to maintain buffer size
        history_buffer = pd.concat([history_buffer.iloc[1:], new_row_combined])
        
    return pd.DataFrame(predictions, index=future_scaffold.index, columns=TARGET_COLS)


# --- 3. Main Execution Block ---
"""Main function to run the entire pipeline."""
# --- Phase 1: Final Data Preparation ---
print("--- Phase 1: Final Data Preparation ---")
df_train = pd.read_csv('data/train_cleaned.csv', parse_dates=['id'], index_col='id')
df_test = pd.read_csv('data/test.csv', parse_dates=['id'], index_col='id')

# Combine train and test to apply features consistently
df_full = pd.concat([df_train, df_test[[]]], axis=0)

df_full = create_deterministic_features(df_full)
df_full = create_climatology_features(df_full, df_train)

# Add rolling features first (they need the target columns)
df_train_with_rolling = create_rolling_features(df_full[df_full.index < df_test.index.min()])

# Then add lagged features
df_final_train = create_lagged_features(df_train_with_rolling)

df_final_train.dropna(inplace=True)

# Separate final features (X) and targets (y)
features_to_drop = TARGET_COLS + WEATHER_COLS
X_train_full = df_final_train.drop(columns=features_to_drop)
y_train_full = df_final_train[TARGET_COLS]

# Fit the scalers on the entire training data
x_scaler = MinMaxScaler().fit(X_train_full)
y_scaler = MinMaxScaler().fit(y_train_full)

# --- Phase 2: Final Model Training ---
print("\n--- Phase 2: Final Model Training ---")

# Show scale information for context
print(f"\nTarget variable ranges (before scaling):")
for col in TARGET_COLS:
    print(f"  {col:15s}: min={y_train_full[col].min():7.2f}, max={y_train_full[col].max():7.2f}, mean={y_train_full[col].mean():7.2f}")
print(f"\nNote: Model trains on scaled values [0,1], but predictions are unscaled for submission.")
print(f"      Training loss shows scaled MAE (~0.05-0.10)")
print(f"      Kaggle evaluates unscaled MAE (~5-10)\n")

X_train_full_scaled = x_scaler.transform(X_train_full)
y_train_full_scaled = y_scaler.transform(y_train_full)
X_train_reshaped, y_train_reshaped = create_sequences(X_train_full_scaled, y_train_full_scaled, TIMESTEPS)

# Create sample weights for the training data
y_train_dates = y_train_full.index[TIMESTEPS:]
sample_weights = pd.Series(1.0, index=y_train_dates)
sample_weights[sample_weights.index.month == 9] = 5.0 # Weight for all Septembers
august_2024_mask = (sample_weights.index.month == 8) & (sample_weights.index.year == 2024)
sample_weights[august_2024_mask] = 2.5

# Train the model
final_model = build_and_train_model(X_train_reshaped, y_train_reshaped, sample_weights.to_numpy(), y_scaler)

# --- Phase 3: Prediction on the Test Set ---
print("\n--- Phase 3: Prediction on the Test Set ---")

# Prepare the scaffold of known future features
future_scaffold = df_full[df_full.index >= df_test.index.min()].drop(columns=TARGET_COLS + WEATHER_COLS)

# Initialize the history buffer from the end of the training data
# Buffer size needs to be at least TIMESTEPS for LSTM input
# We always keep full history for rolling feature calculations
buffer_size = max(TIMESTEPS, 168)  # Need 168 for roll_168h calculations
history_buffer = df_final_train.tail(buffer_size)

# Full history of targets for rolling and lag calculations
full_history = df_final_train[TARGET_COLS].copy()

# Generate predictions
predictions_df = run_recursive_forecast(final_model, x_scaler, y_scaler, history_buffer, future_scaffold, full_history)

# --- Phase 4: Save Submission ---
print("\n--- Phase 4: Saving Submission ---")
submission_df = predictions_df.reset_index()
submission_df.rename(columns={'index': 'id'}, inplace=True)

# Format the id column to match sample submission format (YYYY-MM-DD HH)
submission_df['id'] = submission_df['id'].dt.strftime('%Y-%m-%d %H')

# Ensure all predictions are non-negative
submission_df[TARGET_COLS] = submission_df[TARGET_COLS].clip(lower=0)

# Create filename with timestamp
timestamp = datetime.now().strftime('%H_%M')
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"LSTM_{timestamp}.csv")

submission_df.to_csv(filename, index=False)
print(f"Submission file saved successfully to: {filename}")
