"""
Stock Price Prediction
Author: durre-cmd
Date: 2026-01-12

This script predicts future stock prices using historical data.
It uses Linear Regression to forecast closing prices.
"""

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf  # Optional, for direct stock download

# Step 2: Load data
# Option 1: Use CSV file
# data = pd.read_csv('your_stock_data.csv')

# Option 2: Download data from Yahoo Finance
ticker = 'AAPL'  # You can change this to any stock symbol
data = yf.download(ticker, start='2020-01-01', end='2025-01-01')
data.reset_index(inplace=True)  # Move Date from index to column

print("First 5 rows of data:\n", data.head())

# Step 3: Preprocess data
# Use features: Open, High, Low, Volume to predict Close price
features = ['Open', 'High', 'Low', 'Volume']
X = data[features]
y = data['Close']

# Optional: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False)  # shuffle=False to respect time series

# Step 5: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 score): {r2:.2f}")

# Step 8: Plot actual vs predicted prices
plt.figure(figsize=(12,6))
plt.plot(data['Date'][-len(y_test):], y_test, label='Actual Price', color='blue')
plt.plot(data['Date'][-len(y_test):], y_pred, label='Predicted Price', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Step 9: Optional - Predict next day price
last_row = X_scaled[-1].reshape(1, -1)
next_day_pred = model.predict(last_row)[0]
print(f"\nPredicted next day closing price: {next_day_pred:.2f}")
