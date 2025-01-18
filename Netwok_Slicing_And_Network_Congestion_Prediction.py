



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import streamlit as st

# Create dummy data
np.random.seed(42)
num_samples = 1000
data = {
    'signal_strength': np.random.uniform(-120, -30, num_samples),
    'latency': np.random.uniform(1, 100, num_samples),
    'bandwidth': np.random.uniform(10, 1000, num_samples),
    'ue_mobility': np.random.uniform(0, 120, num_samples),
    'network_load': np.random.uniform(0, 100, num_samples),
    'interference_level': np.random.uniform(-110, -50, num_samples),
    'cell_id': np.random.randint(1, 100, num_samples),
    'frequency_band': np.random.choice([700, 1800, 2600, 3500], num_samples),
    'throughput': np.random.uniform(1, 1000, num_samples),
    'resource_allocation': np.random.uniform(0, 1, num_samples),  # target for resource allocation
    'congestion': np.random.randint(0, 2, num_samples)  # binary target for congestion
}

df = pd.DataFrame(data)

# Normalize the data except the target features
scaler = MinMaxScaler()
features = df.drop(['resource_allocation', 'congestion'], axis=1)
normalized_features = scaler.fit_transform(features)

# Convert the normalized features back to a DataFrame
normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
normalized_df['resource_allocation'] = df['resource_allocation']
normalized_df['congestion'] = df['congestion']

# Split the data into features (X) and targets (y)
X = normalized_df.drop(['resource_allocation', 'congestion'], axis=1)
y_resource_allocation = normalized_df['resource_allocation']
y_congestion = normalized_df['congestion']

# Split the data into training and testing sets
X_train, X_test, y_train_resource_allocation, y_test_resource_allocation = train_test_split(X, y_resource_allocation, test_size=0.2, random_state=42)
X_train, X_test, y_train_congestion, y_test_congestion = train_test_split(X, y_congestion, test_size=0.2, random_state=42)

# Build the resource allocation model
model_resource_allocation = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the resource allocation model
model_resource_allocation.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the resource allocation model
model_resource_allocation.fit(X_train, y_train_resource_allocation, epochs=50, batch_size=32, validation_data=(X_test, y_test_resource_allocation))

# Build the congestion model
model_congestion = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the congestion model
model_congestion.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the congestion model
model_congestion.fit(X_train, y_train_congestion, epochs=50, batch_size=32, validation_data=(X_test, y_test_congestion))

def predict_resource_allocation(model, scaler, user_input):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Normalize the user input
    normalized_user_input = scaler.transform(user_df)
    
    # Predict resource allocation
    prediction = model.predict(normalized_user_input)
    
    return prediction[0][0]

def predict_congestion(model, scaler, user_input):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Normalize the user input
    normalized_user_input = scaler.transform(user_df)
    
    # Predict congestion
    prediction = model.predict(normalized_user_input)
    
    return prediction[0][0]

# Streamlit GUI
st.title("5G Network Resource Allocation and Congestion Prediction")

signal_strength = st.slider("Signal Strength (dBm)", -120, -30, -85)
latency = st.slider("Latency (ms)", 1, 100, 50)
bandwidth = st.slider("Bandwidth (Mbps)", 10, 1000, 500)
ue_mobility = st.slider("UE Mobility (km/h)", 0, 120, 60)
network_load = st.slider("Network Load (%)", 0, 100, 50)
interference_level = st.slider("Interference Level (dBm)", -110, -50, -80)
cell_id = st.slider("Cell ID", 1, 100, 10)
frequency_band = st.selectbox("Frequency Band (MHz)", [700, 1800, 2600, 3500])
throughput = st.slider("Throughput (Mbps)", 1, 1000, 300)

user_input = {
    'signal_strength': signal_strength,
    'latency': latency,
    'bandwidth': bandwidth,
    'ue_mobility': ue_mobility,
    'network_load': network_load,
    'interference_level': interference_level,
    'cell_id': cell_id,
    'frequency_band': frequency_band,
    'throughput': throughput
}

if st.button("Predict Resource Allocation"):
    predicted_allocation = predict_resource_allocation(model_resource_allocation, scaler, user_input) * 100
    st.write(f"Predicted Resource Allocation: {predicted_allocation:.2f}%")

if st.button("Predict Congestion"):
    predicted_congestion = predict_congestion(model_congestion, scaler, user_input) * 100
    st.write(f"Predicted Congestion: {predicted_congestion:.2f}%")
