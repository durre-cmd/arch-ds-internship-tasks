
# Stock Price Prediction

**Task:** Predict future stock closing prices based on historical stock data.

---

## Description

This project is part of the Arch Technologies Data Science Internship.  
The goal is to predict Apple (AAPL) stock closing prices using historical data.  

The model uses the following features to predict the **Close price**:  
- Open price  
- High price  
- Low price  
- Trading volume  

A **Linear Regression model** is used for prediction. The performance is evaluated using **Mean Squared Error (MSE)** and **R-squared (R²)**.

---

## Dataset

- The data contains daily stock prices for Apple (AAPL) from **2020-01-02 to 2025-01-01**.  
- Columns include: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.  
- The data can be obtained using the **yfinance library** or from any historical stock CSV file.

Example:

| Date       | Close    | High     | Low      | Open     | Volume    |
|------------|----------|----------|----------|----------|-----------|
| 2020-01-02 | 72.46826 | 72.52858 | 71.22326 | 71.47660 | 135480400 |
| 2020-01-03 | 71.76371 | 72.52373 | 71.53932 | 71.69615 | 146322800 |

---

## How to Run

1. Install required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn yfinance
````

2. Run the Python script:

```bash
python stock_price_prediction.py
```

3. The script will:

   * Load stock data (from CSV or Yahoo Finance)
   * Preprocess the data
   * Train a Linear Regression model
   * Predict closing prices
   * Display a plot comparing actual vs predicted prices
   * Print evaluation metrics (MSE and R²)
   * Predict next day closing price

---

## Output

* **Mean Squared Error (MSE)**: Indicates how close predicted prices are to actual prices.
* **R-squared (R²)**: Shows how well the model explains variation in stock prices.

Example output:

```
Mean Squared Error (MSE): 1.16
R-squared (R2 score): 1.00
Predicted next day closing price: 73.12
```

* The **plot window** shows actual vs predicted prices over time.
* Blue line = Actual closing prices
* Red line = Predicted closing prices

---

## Notes

* The data used is historical and old; this is for learning and practice purposes.
* Stock price prediction in real markets is highly volatile and complex; this simple model shows basic predictive modeling techniques.
* For more accurate predictions, advanced models like **LSTM** or **ARIMA** can be used.

---

