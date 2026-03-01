"""
Rainfall Prediction using LSTM
Time Series Forecasting for Monthly Rainfall
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Step 1: Load Dataset
# ==========================================
print("="*60)
print("RAINFALL PREDICTION USING LSTM")
print("="*60)

data = pd.read_csv("rainfall.csv")

print("\nDataset Info:")
print(f"Total records: {len(data)}")
print(f"\nFirst 5 rows:")
print(data.head())
print(f"\nLast 5 rows:")
print(data.tail())
print(f"\nBasic Statistics:")
print(data.describe())

# Extract rainfall values
rainfall = data['rainfall'].values.reshape(-1, 1)

print(f"\nRainfall range: {rainfall.min():.2f} mm to {rainfall.max():.2f} mm")

# ==========================================
# Step 2: Visualize Original Data
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(data['month'], data['rainfall'], marker='o', linewidth=2, markersize=6)
plt.title('Monthly Rainfall Data', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Rainfall (mm)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# Step 3: Normalize Data
# ==========================================
scaler = MinMaxScaler(feature_range=(0, 1))
rainfall_scaled = scaler.fit_transform(rainfall)

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)
print(f"Original data range: [{rainfall.min():.2f}, {rainfall.max():.2f}]")
print(f"Scaled data range: [{rainfall_scaled.min():.4f}, {rainfall_scaled.max():.4f}]")

# ==========================================
# Step 4: Create Sequences
# ==========================================
def create_sequences(data, time_steps=3):
    """
    Create input-output sequences for LSTM
    
    Parameters:
    - data: Scaled rainfall data
    - time_steps: Number of previous months to use
    
    Returns:
    - X: Input sequences
    - y: Target values
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 3
X, y = create_sequences(rainfall_scaled, time_steps=time_steps)

print(f"\nSequence Creation:")
print(f"Time steps (lookback): {time_steps} months")
print(f"Total sequences: {len(X)}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Show example
print(f"\nExample sequence:")
print(f"Input (last {time_steps} months): {scaler.inverse_transform(X[0]).flatten()}")
print(f"Target (next month): {scaler.inverse_transform(y[0].reshape(-1, 1)).flatten()[0]:.2f}")

# ==========================================
# Step 5: Train-Test Split
# ==========================================
# Time series split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Train ratio: {len(X_train)/len(X)*100:.1f}%")
print(f"Test ratio: {len(X_test)/len(X)*100:.1f}%")

# ==========================================
# Step 6: Build LSTM Model
# ==========================================
print("\n" + "="*60)
print("BUILDING LSTM MODEL")
print("="*60)

model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

print("\nModel Summary:")
model.summary()

# ==========================================
# Step 7: Train Model
# ==========================================
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining Complete!")

# ==========================================
# Step 8: Make Predictions
# ==========================================
print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)

train_predictions = model.predict(X_train, verbose=0)
test_predictions = model.predict(X_test, verbose=0)

# Inverse transform
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train)
y_test_actual = scaler.inverse_transform(y_test)

# ==========================================
# Step 9: Calculate Metrics
# ==========================================
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
train_mae = mean_absolute_error(y_train_actual, train_predictions)
test_mae = mean_absolute_error(y_test_actual, test_predictions)

print("\nModel Performance:")
print(f"Training RMSE: {train_rmse:.2f} mm")
print(f"Test RMSE: {test_rmse:.2f} mm")
print(f"Training MAE: {train_mae:.2f} mm")
print(f"Test MAE: {test_mae:.2f} mm")

# Accuracy calculation
mape_test = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
accuracy = 100 - mape_test
print(f"\nTest MAPE: {mape_test:.2f}%")
print(f"Test Accuracy: {accuracy:.2f}%")

# ==========================================
# Step 10: Visualize Results
# ==========================================
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

# Plot 1: Training History
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss (MSE)', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE During Training', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('MAE', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 2: Predictions vs Actual
plt.figure(figsize=(14, 6))

# Training data
train_months = range(time_steps, time_steps + len(y_train_actual))
plt.plot(train_months, y_train_actual, label='Actual (Train)', marker='o', linewidth=2)
plt.plot(train_months, train_predictions, label='Predicted (Train)', marker='x', linewidth=2, alpha=0.7)

# Test data
test_months = range(time_steps + len(y_train_actual), time_steps + len(y_train_actual) + len(y_test_actual))
plt.plot(test_months, y_test_actual, label='Actual (Test)', marker='o', linewidth=2)
plt.plot(test_months, test_predictions, label='Predicted (Test)', marker='x', linewidth=2, alpha=0.7)

plt.axvline(x=time_steps + len(y_train_actual), color='red', linestyle='--', linewidth=2, label='Train/Test Split')
plt.title('Rainfall Prediction: Actual vs Predicted', fontsize=14)
plt.xlabel('Month Index', fontsize=12)
plt.ylabel('Rainfall (mm)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 3: Test Set Only (Detailed)
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Rainfall', marker='o', linewidth=2, markersize=8)
plt.plot(test_predictions, label='Predicted Rainfall', marker='x', linewidth=2, markersize=8)
plt.title('Test Set: Rainfall Prediction using LSTM', fontsize=14)
plt.xlabel('Test Sample', fontsize=12)
plt.ylabel('Rainfall (mm)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# Step 11: Future Prediction
# ==========================================
print("\n" + "="*60)
print("FUTURE RAINFALL FORECAST")
print("="*60)

def predict_future(model, last_sequence, scaler, n_months=6):
    """
    Predict future rainfall for next n months
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_months):
        # Predict next month
        next_pred = model.predict(current_sequence.reshape(1, time_steps, 1), verbose=0)
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence (slide window)
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Inverse transform
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Get last sequence from original data
last_sequence = rainfall_scaled[-time_steps:]

# Predict next 6 months
future_months = 6
future_rainfall = predict_future(model, last_sequence, scaler, n_months=future_months)

print(f"\nFuture {future_months}-Month Rainfall Forecast:")
for i, rainfall_value in enumerate(future_rainfall, 1):
    print(f"Month +{i}: {rainfall_value[0]:.2f} mm")

# Plot future forecast
plt.figure(figsize=(12, 6))

# Historical data
historical_months = range(1, len(data) + 1)
plt.plot(historical_months, data['rainfall'], label='Historical Rainfall', marker='o', linewidth=2)

# Future forecast
future_months_range = range(len(data) + 1, len(data) + 1 + future_months)
plt.plot(future_months_range, future_rainfall, label='Forecasted Rainfall', marker='s', linewidth=2, color='red')

plt.axvline(x=len(data), color='green', linestyle='--', linewidth=2, label='Forecast Start')
plt.title('Historical Rainfall + Future Forecast', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Rainfall (mm)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# Step 12: Save Model
# ==========================================
model.save('rainfall_lstm_model.h5')
print("\n" + "="*60)
print("Model saved: rainfall_lstm_model.h5")
print("="*60)

# ==========================================
# Final Summary
# ==========================================
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"Dataset: {len(data)} months of rainfall data")
print(f"Time steps: {time_steps} months lookback")
print(f"Test MAE: {test_mae:.2f} mm")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Future forecast: {future_months} months generated")
print("="*60)