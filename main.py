import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and inspect the dataset
data = pd.read_csv("/content/synthetic_traffic_data.csv")
print(data.info())

# Ensure necessary columns exist
required_columns = ['timestamp', 'temperature', 'humidity', 'traffic_flow']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Handle missing values
data.ffill(inplace=True)

# Convert 'timestamp' column to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values('timestamp', inplace=True)

# Extract time-based features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x > 4 else 0)

# Normalize numerical features
scaler = MinMaxScaler()
data[['temperature', 'humidity', 'traffic_flow']] = scaler.fit_transform(
    data[['temperature', 'humidity', 'traffic_flow']]
)

# Add lag features for traffic_flow
for lag in range(1, 7):  # Previous 6 hours
    data[f'lag_{lag}'] = data['traffic_flow'].shift(lag)
data.dropna(inplace=True)  # Drop rows with NaN values due to lagging

# Prepare input (X) and target (y) variables
X = data.drop(['traffic_flow', 'timestamp'], axis=1)
y = data['traffic_flow']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Define the model with an explicit Input layer
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Explicit input layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Increased dropout rate for better regularization
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)  # Single output for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Additional evaluation: R-squared and MAPE
y_pred = model.predict(X_test).flatten()

# R-squared calculation
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print(f"R-squared: {r_squared:.4f}")

# MAPE calculation
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
