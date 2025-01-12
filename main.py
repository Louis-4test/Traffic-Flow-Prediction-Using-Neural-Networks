import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from google.colab import files



data = pd.read_csv("/content/synthetic_traffic_data.csv")
print(data.info())
#print(data.describe())

# Handle missing values
#data.ffill(inplace=True)

# Convert 'timestamp' column to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values('timestamp', inplace=True)

# Extract time-based features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x > 4 else 0)

#weather_data = pd.read_csv("weather_data.csv")
#data = pd.merge(data, weather_data, on='timestamp', how='inner')
print(tf.__version__)
data['hour'] = data['timestamp'].dt.hour
data['weekday'] = data['timestamp'].dt.dayofweek

scaler = MinMaxScaler()
data[['temperature', 'humidity']] = scaler.fit_transform(data[['temperature', 'humidity']])

for lag in range(1, 4):  # Previous 3 hours
    data[f'lag_{lag}'] = data['traffic_flow'].shift(lag)
data.dropna(inplace=True)

data['traffic_flow'] = scaler.fit_transform(data[['traffic_flow']])

X = data.drop(['traffic_flow', 'timestamp'], axis=1)
y = data['traffic_flow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Generate dummy data for demonstration
num_samples = 1000
input_dim = 20
X_train = np.random.random((num_samples, input_dim))
y_train = np.random.randint(10, size=(num_samples,))

# Create a simple model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
