from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from model import fetch_stock_data, prepare_data, train_model, predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        ticker = request.form["ticker"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        try:
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            X_train, X_test, y_train, y_test, scaler = prepare_data(stock_data)
            model = train_model(X_train, y_train)
            prediction = predict(model, X_test, scaler)
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
