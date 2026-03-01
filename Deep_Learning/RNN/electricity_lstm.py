import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------
# Load Dataset
# --------------------------
data = pd.read_csv("electricity.csv")
consumption = data['consumption'].values.reshape(-1, 1)

print(f"Total data points: {len(consumption)}")
print(f"Data range: {consumption.min()} to {consumption.max()}")

# --------------------------
# Normalize Data
# --------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(consumption)

# --------------------------
# Create Time Series Sequences
# --------------------------
def create_sequences(data, time_steps=5):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, time_steps=5)

print(f"\nSequence shape:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# --------------------------
# Train-Test Split
# --------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# --------------------------
# Build LSTM Model
# --------------------------
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

print("\nModel Summary:")
model.summary()

# --------------------------
# Train Model
# --------------------------
print("\nTraining Started...\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\nTraining Complete!")

# --------------------------
# Predictions
# --------------------------
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test)

# Calculate metrics
mse = np.mean((actual - predicted) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual - predicted))

print(f"\nModel Performance:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# --------------------------
# Plot Results
# --------------------------
plt.figure(figsize=(12, 6))

# Subplot 1: Predictions vs Actual
plt.subplot(1, 2, 1)
plt.plot(actual, label="Actual Consumption", marker='o')
plt.plot(predicted, label="Predicted Consumption", marker='x')
plt.title("Electricity Consumption Forecasting")
plt.xlabel("Time")
plt.ylabel("Electricity Units")
plt.legend()
plt.grid(True)

# Subplot 2: Training Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("electricity_forecast_results.png")
print("\nPlot saved as 'electricity_forecast_results.png'")
plt.show()

# --------------------------
# Future Prediction Demo
# --------------------------
print("\nFuture Prediction Demo:")

# Take last 5 values to predict next
last_sequence = scaled_data[-5:]
last_sequence = last_sequence.reshape(1, 5, 1)

next_prediction = model.predict(last_sequence)
next_prediction = scaler.inverse_transform(next_prediction)

print(f"Last 5 days consumption: {consumption[-5:].flatten()}")
print(f"Predicted next day consumption: {next_prediction[0][0]:.2f}")