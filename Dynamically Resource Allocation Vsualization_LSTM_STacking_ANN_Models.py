
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Step 1: Data Collection
def collect_data():
    data = {
        'device_type': np.random.choice(['Smartphone', 'Tablet', 'Laptop', 'IoT Device'], 1000),
        'resource_required': np.random.randint(1, 100, 1000),
        'service_request': np.random.choice(['Video Streaming', 'Web Browsing', 'Online Gaming', 'IoT Communication'], 1000),
        'bandwidth': np.random.randint(1, 100, 1000),
        'user_demand': np.random.randint(1, 100, 1000),
        'network_conditions': np.random.randint(1, 100, 1000),
        'performance_metrics': np.random.randint(1, 100, 1000)
    }
    return pd.DataFrame(data)

# Step 2: Data Preprocessing
def preprocess_data(data):
    numerical_features = ['resource_required', 'bandwidth', 'user_demand', 'network_conditions']
    categorical_features = ['device_type', 'service_request']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    preprocessed_data = preprocessor.fit_transform(data)
    return preprocessed_data

# Step 3: Model Training
def build_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(input_shape, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_models(data, target):
    X = data
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Neural Network
    nn_model = build_nn_model(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = build_lstm_model(X_train.shape[1])
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Stacking
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('lr', LinearRegression())
    ]
    stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stack_model.fit(X_train, y_train)
    
    return nn_model, lstm_model, stack_model, X_test, y_test

# Step 4: Model Deployment
def deploy_models(nn_model, lstm_model, stack_model, new_data):
    # Neural Network Predictions
    nn_predictions = nn_model.predict(new_data)
    
    # LSTM Predictions
    new_data_lstm = new_data.reshape((new_data.shape[0], new_data.shape[1], 1))
    lstm_predictions = lstm_model.predict(new_data_lstm)
    
    # Stacking Predictions
    stack_predictions = stack_model.predict(new_data)
    
    return nn_predictions, lstm_predictions, stack_predictions

# Step 5: Resource Allocation
def allocate_resources(predictions):
    allocation = ['High' if pred > 50 else 'Low' for pred in predictions]
    return allocation

# Step 6: Network Slicing
def create_network_slices(data, allocation):
    slices = []
    for i in range(len(data)):
        if allocation[i] == 'High':
            slices.append('eMBB')
        else:
            slices.append('mMTC')
    return slices

# Step 7: Performance Monitoring
def monitor_performance(data):
    performance = data['performance_metrics'].mean()
    return performance

# Step 8: Feedback Loop
def feedback_loop(models, data, target):
    nn_model, lstm_model, stack_model = models
    new_models = train_models(data, target)
    return new_models[:3]  # Ensure only models are returned

# Streamlit App
st.title('5G+AI/ML Resource Allocation and Network Slicing')

# Collect and display data
data = collect_data()
st.write('Collected Data:', data.head())

# Preprocess data
preprocessed_data = preprocess_data(data)
st.write('Preprocessed Data:', preprocessed_data[:5])

# Train models
target = data['performance_metrics']
nn_model, lstm_model, stack_model, X_test, y_test = train_models(preprocessed_data, target)

# Deploy models and make predictions
new_data = collect_data()
preprocessed_new_data = preprocess_data(new_data)
nn_predictions, lstm_predictions, stack_predictions = deploy_models(nn_model, lstm_model, stack_model, preprocessed_new_data)
st.write('Neural Network Predictions:', nn_predictions[:5])
st.write('LSTM Predictions:', lstm_predictions[:5])
st.write('Stacking Predictions:', stack_predictions[:5])

# Resource allocation
resource_allocation = allocate_resources(stack_predictions)
st.write('Resource Allocation:', resource_allocation[:5])

# Network slicing
network_slices = create_network_slices(new_data, resource_allocation)
st.write('Network Slices:', network_slices[:5])

# Performance monitoring
performance = monitor_performance(new_data)
st.write(f'Average Performance: {performance}')

# Feedback loop
new_models = feedback_loop((nn_model, lstm_model, stack_model), preprocessed_data, target)
st.write('Models retrained with new data')

# Deploy updated models and make predictions
nn_model, lstm_model, stack_model = new_models
nn_predictions, lstm_predictions, stack_predictions = deploy_models(nn_model, lstm_model, stack_model, preprocessed_new_data)
st.write('Updated Neural Network Predictions:', nn_predictions[:5])
st.write('Updated LSTM Predictions:', lstm_predictions[:5])
st.write('Updated Stacking Predictions:', stack_predictions[:5])

"""


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Step 1: Data Collection
def collect_data():
    data = {
        'device_type': np.random.choice(['Smartphone', 'Tablet', 'Laptop', 'IoT Device'], 1000),
        'resource_required': np.random.randint(1, 100, 1000),
        'service_request': np.random.choice(['Video Streaming', 'Web Browsing', 'Online Gaming', 'IoT Communication'], 1000),
        'bandwidth': np.random.randint(1, 100, 1000),
        'user_demand': np.random.randint(1, 100, 1000),
        'network_conditions': np.random.randint(1, 100, 1000),
        'performance_metrics': np.random.randint(1, 100, 1000)
    }
    return pd.DataFrame(data)

# Step 2: Data Preprocessing
def preprocess_data(data):
    numerical_features = ['resource_required', 'bandwidth', 'user_demand', 'network_conditions']
    categorical_features = ['device_type', 'service_request']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    preprocessed_data = preprocessor.fit_transform(data)
    return preprocessed_data

# Step 3: Model Training
def build_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(input_shape, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_models(data, target):
    X = data
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Neural Network
    nn_model = build_nn_model(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model = build_lstm_model(X_train.shape[1])
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Stacking
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('lr', LinearRegression())
    ]
    stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stack_model.fit(X_train, y_train)
    
    return nn_model, lstm_model, stack_model, X_test, y_test

# Step 4: Model Deployment
def deploy_models(nn_model, lstm_model, stack_model, new_data):
    # Neural Network Predictions
    nn_predictions = nn_model.predict(new_data)
    
    # LSTM Predictions
    new_data_lstm = new_data.reshape((new_data.shape[0], new_data.shape[1], 1))
    lstm_predictions = lstm_model.predict(new_data_lstm)
    
    # Stacking Predictions
    stack_predictions = stack_model.predict(new_data)
    
    return nn_predictions, lstm_predictions, stack_predictions

# Step 5: Resource Allocation
def allocate_resources(predictions):
    allocation = ['High' if pred > 50 else 'Low' for pred in predictions]
    return allocation

# Step 6: Network Slicing
def create_network_slices(data, allocation):
    slices = []
    for i in range(len(data)):
        if allocation[i] == 'High':
            slices.append('eMBB')
        else:
            slices.append('mMTC')
    return slices

# Step 7: Performance Monitoring
def monitor_performance(data):
    performance = data['performance_metrics'].mean()
    return performance

# Step 8: Feedback Loop
def feedback_loop(models, data, target):
    nn_model, lstm_model, stack_model = models
    new_models = train_models(data, target)
    return new_models[:3]  # Ensure only models are returned

# Streamlit App
st.title('5G+AI/ML Resource Allocation and Network Slicing')

# Collect and display data
data = collect_data()
st.write('Collected Data:', data.head())

# Preprocess data
preprocessed_data = preprocess_data(data)
st.write('Preprocessed Data:', preprocessed_data[:5])

# Train models
target = data['performance_metrics']
nn_model, lstm_model, stack_model, X_test, y_test = train_models(preprocessed_data, target)

# Deploy models and make predictions
new_data = collect_data()
preprocessed_new_data = preprocess_data(new_data)
nn_predictions, lstm_predictions, stack_predictions = deploy_models(nn_model, lstm_model, stack_model, preprocessed_new_data)
st.write('Neural Network Predictions:', nn_predictions[:5])
st.write('LSTM Predictions:', lstm_predictions[:5])
st.write('Stacking Predictions:', stack_predictions[:5])

# Resource allocation
resource_allocation = allocate_resources(stack_predictions)
st.write('Resource Allocation:', resource_allocation[:5])

# Network slicing
network_slices = create_network_slices(new_data, resource_allocation)
st.write('Network Slices:', network_slices[:5])

# Performance monitoring
performance = monitor_performance(new_data)
st.write(f'Average Performance: {performance}')

# Feedback loop
new_models = feedback_loop((nn_model, lstm_model, stack_model), preprocessed_data, target)
st.write('Models retrained with new data')

# Deploy updated models and make predictions
nn_model, lstm_model, stack_model = new_models
nn_predictions, lstm_predictions, stack_predictions = deploy_models(nn_model, lstm_model, stack_model, preprocessed_new_data)
st.write('Updated Neural Network Predictions:', nn_predictions[:5])
st.write('Updated LSTM Predictions:', lstm_predictions[:5])
st.write('Updated Stacking Predictions:', stack_predictions[:5])

# Plotting predictions for better visualization
st.subheader('Predictions Visualization')
fig, ax = plt.subplots()
ax.plot(nn_predictions, label='Neural Network Predictions')
ax.plot(lstm_predictions, label='LSTM Predictions')
ax.plot(stack_predictions, label='Stacking Predictions')
ax.legend()
st.pyplot(fig)

