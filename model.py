import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries

# Fetch stock data using Alpha Vantage API
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"

def fetch_stock_data(ticker, start_date, end_date):
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    data, _ = ts.get_daily(symbol=ticker, outputsize="full")
    
    data = data[start_date:end_date]
    if data.empty:
        raise ValueError("No data found. Try another date range.")
    
    return data

# Prepare data for training
def prepare_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(stock_data["4. close"].values.reshape(-1,1))

    X, y = [], []
    for i in range(30, len(scaled_data)):
        X.append(scaled_data[i-30:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    return X[: -10], X[-10:], y[: -10], y[-10:], scaler

# Train LSTM Model
def train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dense(units=1)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=16)
    
    return model

# Predict future stock prices
def predict(model, X_test, scaler):
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions).tolist()
