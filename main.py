import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and inspect the dataset
data = pd.read_csv("/content/synthetic_traffic_data.csv")
print(data.info())

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
for lag in range(1, 4):  # Previous 3 hours
    data[f'lag_{lag}'] = data['traffic_flow'].shift(lag)
data.dropna(inplace=True)  # Drop rows with NaN values due to lagging

# Prepare input (X) and target (y) variables
X = data.drop(['traffic_flow', 'timestamp'], axis=1)
y = data['traffic_flow']

# Define hyperparameter grid
param_grid = {
    'units': [50],  # Number of LSTM units
    'dropout_rate': [0.2],  # Dropout rates
    'batch_size': [32],  # Batch sizes
    'learning_rate': [0.001]  # Learning rates
}

# Initialize variables to store best model and results
best_model = None
best_params = None
lowest_rmse = float('inf')

# Perform k-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for units in param_grid['units']:
    for dropout_rate in param_grid['dropout_rate']:
        for batch_size in param_grid['batch_size']:
            for learning_rate in param_grid['learning_rate']:

                fold_rmse = []

                for train_index, val_index in kf.split(X):
                    X_train, X_val = X.values[train_index], X.values[val_index]
                    y_train, y_val = y.values[train_index], y.values[val_index]

                    # Build LSTM model with Input layer
                    model = tf.keras.Sequential([
                        layers.Input(shape=(X_train.shape[1], 1)),  # Use Input layer to specify input shape
                        layers.LSTM(units, activation='tanh', return_sequences=True),
                        layers.Dropout(dropout_rate),
                        layers.LSTM(units, activation='tanh'),
                        layers.Dropout(dropout_rate),
                        layers.Dense(1)  # Regression output
                    ])

                    # Compile the model
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

                    # Train the model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=5,
                        batch_size=batch_size,
                        verbose=0
                    )

                    # Evaluate the model on the validation set
                    y_pred = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    fold_rmse.append(rmse)

                # Calculate average RMSE for this hyperparameter set
                avg_rmse = np.mean(fold_rmse)

                # Update best model if current RMSE is lower
                if avg_rmse < lowest_rmse:
                    lowest_rmse = avg_rmse
                    best_model = model
                    best_params = {
                        'units': units,
                        'dropout_rate': dropout_rate,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate
                    }

# Display best parameters and lowest RMSE
print(f"Best Parameters: {best_params}")
print(f"Lowest RMSE: {lowest_rmse:.4f}")

# Evaluate the best model on the test set
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
y_test_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label='Actual', marker='o')
plt.plot(y_test_pred[:100], label='Predicted', marker='x')
plt.title('Predicted vs Actual Traffic Flow')
plt.xlabel('Sample Index')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()

# Visualize the relationship between weather and traffic flow
plt.figure(figsize=(10, 6))
plt.scatter(data['temperature'], data['traffic_flow'], alpha=0.5, label='Temperature vs Traffic Flow')
plt.scatter(data['humidity'], data['traffic_flow'], alpha=0.5, label='Humidity vs Traffic Flow', color='orange')
plt.title('Weather vs Traffic Flow')
plt.xlabel('Weather Conditions')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()
