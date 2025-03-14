from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Prepare data for training
def prepare_data(stock_data):
    stock_data['Prediction'] = stock_data['Close'].shift(-30)  # Predict 30 days into the future
    X = np.array(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']][:-30])
    y = np.array(stock_data['Prediction'][:-30])
    return X, y

# Train the model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        ticker = request.form["ticker"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        stock_data = fetch_stock_data(ticker, start_date, end_date)
        X, y = prepare_data(stock_data)
        model = train_model(X, y)
        prediction = model.predict([X[-1]])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)