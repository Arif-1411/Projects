"""
Zomato Stock Price Prediction - Complete Workflow
End-to-End Time Series Forecasting using LSTM/GRU
"""

# ================================
# Step 1: Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ================================
# Step 2: Load Data
# ================================
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/zomato.csv'
df = pd.read_csv(file_path)

print("="*60)
print("DATASET INFO")
print("="*60)
print("\nFirst 5 rows:")
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ================================
# Step 3: Data Preprocessing
# ================================
# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

print("\n" + "="*60)
print("PREPROCESSED DATA")
print("="*60)
print(df.head())
print(f"Date range: {df.index.min()} to {df.index.max()}")

# ================================
# Step 4: Visualization - Original Data
# ================================
plt.figure(figsize=(14, 5))
plt.plot(df['Close'], linewidth=2)
plt.title('Zomato Stock Price - Close Price History', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (INR)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================
# PART 1: BASELINE MODEL (Single Feature - Close)
# ================================
print("\n" + "="*60)
print("PART 1: BASELINE MODEL (LSTM - Close Price Only)")
print("="*60)

# Prepare data
close_prices = df['Close'].values.reshape(-1, 1)

# Scale data
scaler_baseline = MinMaxScaler(feature_range=(0, 1))
scaled_data_baseline = scaler_baseline.fit_transform(close_prices)

# Train-test split (80-20)
train_size = int(len(scaled_data_baseline) * 0.8)
train_data_baseline = scaled_data_baseline[:train_size]
test_data_baseline = scaled_data_baseline[train_size:]

print(f"\nTrain size: {len(train_data_baseline)}")
print(f"Test size: {len(test_data_baseline)}")

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length_baseline = 60
X_train_baseline, y_train_baseline = create_sequences(train_data_baseline, seq_length_baseline)
X_test_baseline, y_test_baseline = create_sequences(test_data_baseline, seq_length_baseline)

# Reshape for LSTM
X_train_baseline = X_train_baseline.reshape(X_train_baseline.shape[0], X_train_baseline.shape[1], 1)
X_test_baseline = X_test_baseline.reshape(X_test_baseline.shape[0], X_test_baseline.shape[1], 1)

print(f"\nX_train shape: {X_train_baseline.shape}")
print(f"X_test shape: {X_test_baseline.shape}")

# Build LSTM Model
model_baseline = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length_baseline, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model_baseline.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

print("\nBaseline Model Summary:")
model_baseline.summary()

# Train model
print("\nTraining Baseline Model...")
history_baseline = model_baseline.fit(
    X_train_baseline, y_train_baseline,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_baseline, y_test_baseline),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Predictions
train_predict_baseline = model_baseline.predict(X_train_baseline)
test_predict_baseline = model_baseline.predict(X_test_baseline)

# Inverse transform
train_predict_baseline = scaler_baseline.inverse_transform(train_predict_baseline)
test_predict_baseline = scaler_baseline.inverse_transform(test_predict_baseline)
y_train_actual_baseline = scaler_baseline.inverse_transform(y_train_baseline.reshape(-1, 1))
y_test_actual_baseline = scaler_baseline.inverse_transform(y_test_baseline.reshape(-1, 1))

# Calculate metrics
train_rmse_baseline = np.sqrt(mean_squared_error(y_train_actual_baseline, train_predict_baseline))
test_rmse_baseline = np.sqrt(mean_squared_error(y_test_actual_baseline, test_predict_baseline))
train_mae_baseline = mean_absolute_error(y_train_actual_baseline, train_predict_baseline)
test_mae_baseline = mean_absolute_error(y_test_actual_baseline, test_predict_baseline)

print("\n" + "="*60)
print("BASELINE MODEL METRICS")
print("="*60)
print(f"Training RMSE: Rs.{train_rmse_baseline:.2f}")
print(f"Test RMSE: Rs.{test_rmse_baseline:.2f}")
print(f"Training MAE: Rs.{train_mae_baseline:.2f}")
print(f"Test MAE: Rs.{test_mae_baseline:.2f}")

# Plot results
plt.figure(figsize=(14, 5))
plt.plot(df.index[seq_length_baseline:train_size], y_train_actual_baseline, label='Actual Train', linewidth=2)
plt.plot(df.index[seq_length_baseline:train_size], train_predict_baseline, label='Predicted Train', linewidth=2, alpha=0.7)
plt.plot(df.index[train_size+seq_length_baseline:], y_test_actual_baseline, label='Actual Test', linewidth=2)
plt.plot(df.index[train_size+seq_length_baseline:], test_predict_baseline, label='Predicted Test', linewidth=2, alpha=0.7)
plt.title('Baseline Model: Zomato Stock Price Prediction (LSTM)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================
# PART 2: IMPROVED MODEL (Multi-Feature GRU)
# ================================
print("\n" + "="*60)
print("PART 2: IMPROVED MODEL (GRU - Multi-Feature)")
print("="*60)

# Use multiple features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
dataset = df[features].values

print(f"\nFeatures used: {features}")
print(f"Dataset shape: {dataset.shape}")

# Scale features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# Train-test split
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for multi-feature
def create_sequences_multifeature(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3])  # Close price is at index 3
    return np.array(X), np.array(y)

seq_length = 90
X_train, y_train = create_sequences_multifeature(train_data, seq_length)
X_test, y_test = create_sequences_multifeature(test_data, seq_length)

print(f"\nSequence length: {seq_length}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# GRU Model
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

print("\nImproved Model Summary:")
model.summary()

# Train model
print("\nTraining Improved Model...")
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
    verbose=1
)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform (only Close column)
close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']])

train_predict = close_scaler.inverse_transform(train_predict)
test_predict = close_scaler.inverse_transform(test_predict)
y_train_actual = close_scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
train_mae = mean_absolute_error(y_train_actual, train_predict)
test_mae = mean_absolute_error(y_test_actual, test_predict)

print("\n" + "="*60)
print("IMPROVED MODEL METRICS")
print("="*60)
print(f"Training RMSE: Rs.{train_rmse:.2f}")
print(f"Test RMSE: Rs.{test_rmse:.2f}")
print(f"Training MAE: Rs.{train_mae:.2f}")
print(f"Test MAE: Rs.{test_mae:.2f}")

# Plot results
plt.figure(figsize=(14, 5))
plt.plot(df.index[seq_length:train_size], y_train_actual, label='Actual Train', linewidth=2)
plt.plot(df.index[seq_length:train_size], train_predict, label='Predicted Train', linewidth=2, alpha=0.7)
plt.plot(df.index[train_size+seq_length:], y_test_actual, label='Actual Test', linewidth=2)
plt.plot(df.index[train_size+seq_length:], test_predict, label='Predicted Test', linewidth=2, alpha=0.7)
plt.title('Improved Model: Zomato Stock Price Prediction (GRU - Multi-Feature)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# PART 3: 30 DAYS FUTURE FORECAST
# ================================
print("\n" + "="*60)
print("PART 3: 30-DAY FUTURE FORECAST")
print("="*60)

future_days = 30
last_sequence = scaled_data[-seq_length:].copy()
last_sequence = last_sequence.reshape(1, seq_length, X_train.shape[2])

future_predictions = []

print("\nGenerating 30-day forecast...")
for day in range(future_days):
    next_price = model.predict(last_sequence, verbose=0)[0][0]
    future_predictions.append(next_price)
    
    # Create next row with predicted Close
    next_row = np.zeros((1, X_train.shape[2]))
    next_row[0, 3] = next_price  # Close column
    
    # Slide window
    last_sequence = np.append(last_sequence[:, 1:, :], next_row.reshape(1, 1, -1), axis=1)

# Inverse transform predictions
future_predictions = close_scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

print(f"\nForecast complete!")
print(f"Current price: Rs.{df['Close'].iloc[-1]:.2f}")
print(f"Predicted price after 30 days: Rs.{future_predictions[-1][0]:.2f}")

# Plot future forecast
plt.figure(figsize=(12, 5))
plt.plot(range(1, future_days+1), future_predictions, marker='o', linewidth=2, markersize=6)
plt.title("Next 30 Days Stock Price Forecast", fontsize=14)
plt.xlabel("Days into Future", fontsize=12)
plt.ylabel("Predicted Price (INR)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Combined plot
plt.figure(figsize=(14, 6))
historical_days = 60
plt.plot(range(-historical_days, 0), df['Close'].iloc[-historical_days:], label='Historical', linewidth=2)
plt.plot(range(0, future_days), future_predictions, label='Forecast', marker='o', linewidth=2, markersize=4)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Today')
plt.title("Zomato Stock: Historical + 30-Day Forecast", fontsize=14)
plt.xlabel("Days", fontsize=12)
plt.ylabel("Price (INR)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================
# PART 4: SAVE & LOAD MODEL
# ================================
print("\n" + "="*60)
print("PART 4: SAVE & LOAD MODEL")
print("="*60)

# Save model
model_filename = "zomato_stock_gru_model.h5"
model.save(model_filename)
print(f"\nModel saved: {model_filename}")

# Load model
loaded_model = load_model(model_filename)
print(f"Model loaded successfully!")

# Verify loaded model
test_prediction = loaded_model.predict(X_test[:1], verbose=0)
print(f"\nTest prediction from loaded model: Rs.{close_scaler.inverse_transform(test_prediction)[0][0]:.2f}")
print(f"Actual value: Rs.{y_test_actual[0][0]:.2f}")

# ================================
# PART 5: DEPLOYMENT FUNCTION
# ================================
print("\n" + "="*60)
print("PART 5: DEPLOYMENT-STYLE PREDICTION")
print("="*60)

def predict_next_day(last_n_days_data, model, scaler, close_scaler, seq_length):
    """
    Predict next day closing price
    
    Parameters:
    - last_n_days_data: Recent OHLCV data (shape: seq_length x 5)
    - model: Trained model
    - scaler: Feature scaler
    - close_scaler: Close price scaler
    - seq_length: Sequence length used in training
    
    Returns:
    - Predicted next day closing price
    """
    # Scale input
    scaled_input = scaler.transform(last_n_days_data)
    scaled_input = scaled_input.reshape(1, seq_length, last_n_days_data.shape[1])
    
    # Predict
    prediction = model.predict(scaled_input, verbose=0)
    
    # Inverse transform
    predicted_price = close_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
    
    return predicted_price

# Example usage
last_90_days = df[features].values[-seq_length:]
next_day_price = predict_next_day(last_90_days, model, scaler, close_scaler, seq_length)

print(f"\nLast known price: Rs.{df['Close'].iloc[-1]:.2f}")
print(f"Predicted next day price: Rs.{next_day_price:.2f}")
print(f"Predicted change: Rs.{next_day_price - df['Close'].iloc[-1]:.2f}")

# ================================
# FINAL SUMMARY
# ================================
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"\nDataset: {len(df)} days of Zomato stock data")
print(f"Features: {features}")
print(f"Sequence Length: {seq_length} days")
print(f"\nBaseline Model (LSTM - Close only):")
print(f"  Test RMSE: Rs.{test_rmse_baseline:.2f}")
print(f"  Test MAE: Rs.{test_mae_baseline:.2f}")
print(f"\nImproved Model (GRU - Multi-feature):")
print(f"  Test RMSE: Rs.{test_rmse:.2f}")
print(f"  Test MAE: Rs.{test_mae:.2f}")
print(f"\n30-Day Forecast Generated")
print(f"Model Saved: {model_filename}")
print(f"\nDeployment Function Ready!")
print("="*60)