import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the Google stock prices dataset
df = pd.read_csv('GOOG.csv')

# Take only the 'Close' column for analysis
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define constants
LOOKBACK_WINDOW = 60  # Number of past days to consider for prediction
TRAIN_SIZE = int(len(scaled_data) * 0.8)

# Create sequences of data for training
X_train, y_train = [], []
for i in range(LOOKBACK_WINDOW, TRAIN_SIZE):
    X_train.append(scaled_data[i-LOOKBACK_WINDOW:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape data for LSTM input [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prepare test data
test_data = scaled_data[TRAIN_SIZE - LOOKBACK_WINDOW:, :]
X_test, y_test = [], data[TRAIN_SIZE:, 0]
for i in range(LOOKBACK_WINDOW, len(test_data)):
    X_test.append(test_data[i-LOOKBACK_WINDOW:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(df['Date'][TRAIN_SIZE:], y_test, color='blue', label='Actual Google Stock Price')
plt.plot(df['Date'][TRAIN_SIZE:], predictions, color='red', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
