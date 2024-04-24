import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load the Boston housing dataset
boston = load_boston()

# Split the data into features and target
X = boston.data
y = boston.target.reshape(-1, 1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(1)  # Linear output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)

# Predict on the same data for simplicity (not recommended for real use)
predictions = model.predict(X_scaled)

# Print the first 5 predictions
for i in range(5):
    print("Predicted:", predictions[i][0], "Actual:", y[i][0])
