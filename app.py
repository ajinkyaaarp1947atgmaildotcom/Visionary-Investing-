import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

def get_stock_data(symbol, period="1y"):
    data = yf.download(symbol, period=period)
    return data[['Close']]  # Only use the 'Close' price for prediction

def train_model(stock_data):
    stock_data['Days'] = np.arange(len(stock_data))
    X = stock_data[['Days']]  # Features (days as independent variable)
    y = stock_data['Close']   # Target (closing price)
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def predict_future(model, last_day, forecast_days=30):
    future_days = np.arange(last_day, last_day + forecast_days).reshape(-1, 1)
    predicted_prices = model.predict(future_days)
    return predicted_prices

@app.route('/predict', methods=['GET'])
def predict():
    # Extract stock symbol and forecast days from URL query parameters
    symbol = request.args.get('symbol', default='AAPL')
    forecast_days = int(request.args.get('forecast_days', default=30))
    
    # Get stock data and train the model
    stock_data = get_stock_data(symbol)
    model = train_model(stock_data)
    
    # Predict future prices
    last_day = len(stock_data)
    predicted_prices = predict_future(model, last_day, forecast_days)
    
    # Prepare the result
    result = {
        'symbol': symbol,
        'predicted_prices': predicted_prices.tolist()
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
