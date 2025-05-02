# HEX SOFTWARES - Stock Price Prediction (Talk1)
# Author: Karabo Mokgatsane

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(ticker='AAPL', start='2020-01-01', end='2023-12-31'):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

def visualize_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.title('Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

def prepare_data(data):
    data['Prev_Close'] = data['Close'].shift(1)
    data = data.dropna()
    X = data[['Prev_Close']]
    y = data['Close']
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    
    return predictions, y_test

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, predictions, label='Predicted', linestyle='--')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Index')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = load_data()
    visualize_data(data)
    X_train, X_test, y_train, y_test = prepare_data(data)
    predictions, y_test = train_and_evaluate(X_train, X_test, y_train, y_test)
    plot_predictions(y_test, predictions)



