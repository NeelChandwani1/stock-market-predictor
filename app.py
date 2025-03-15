from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        ticker = request.form["ticker"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        # Convert dates to datetime objects
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Validate date range
        if end_date_dt <= start_date_dt:
            error = "End date must be after start date."
        elif (end_date_dt - start_date_dt).days < 30:
            error = "Date range must be at least 30 days."
        else:
            try:
                stock_data = fetch_stock_data(ticker, start_date, end_date)
                if stock_data.empty:
                    error = "No data found for the specified date range."
                else:
                    X, y = prepare_data(stock_data)
                    model = train_model(X, y)
                    prediction = model.predict([X[-1]])[0]
            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

app = Flask(__name__)

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError("No data found for the specified date range.")
    stock_data.dropna(inplace=True)  # Remove rows with missing values
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