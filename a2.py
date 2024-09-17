from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)



data = pd.read_csv("city_day.csv")
data.pop("City")
data.pop("Date")
data.pop("AQI_Bucket")

x=data.drop(columns=['AQI'])
y=data['AQI']


pd.set_option('display.max_rows', y.shape[0]+1)



X_np=x.to_numpy()
y_np=y.to_numpy()

# # Convert data types to float32
X_np = X_np.astype(np.float32)
y_np = y_np.astype(np.float32)

# # Check for NaNs and Infs and handle them
X_np = np.nan_to_num(X_np, nan=0.0, posinf=None, neginf=None)
y_np = np.nan_to_num(y_np, nan=0.0, posinf=None, neginf=None)

# # Standardize features
scaler = StandardScaler()
X_np = scaler.fit_transform(X_np)



# # Create TensorFlow datasets
# dataset = tf.data.Dataset.from_tensor_slices((X_np, y_np))

X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)


# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression (AQI prediction)
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])


# Train the model
history = model.fit(X_train, y_train, epochs=40,batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f'Test MAE: {test_mae}')



