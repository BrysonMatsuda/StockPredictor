# -*- coding: utf-8 -*-
"""Vector Auto Regression Stock Price Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JEZ28-DrkVM3uKLc-fJQVq7Dhl99zCIh

# Vector Auto Regression Stock Price Analysis

Name: Saaswath Kumar

Make sure to include the GOOG.csv file locally when running
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../data/raw/GOOG.csv', index_col = 'Date', parse_dates = True)

df = df.sort_index()

df.head()

"""For time series datasets we use a rolling window, previous days closing price.

Previous day volume, close.
"""

df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Open_Diff_2'] = df['Open'].diff().diff()
df['High_Diff_2'] = df['High'].diff().diff()
df['Low_Diff_2'] = df['Low'].diff().diff()
df['Close_Diff_2'] = df['Close'].diff().diff()
df['Adj_Close_Diff_2'] = df['Adj Close'].diff().diff()
volatility_window = 10
df['Volatility'] = df['High'] - df['Low']
df['Volatility'] = df['Volatility'].rolling(window=10).std()



df['return_3d'] = df['Close'].pct_change(3).shift(1)
df['Naive_3_day_Close'] = np.exp(df['return_3d']) * df['Close'].shift(1)
# df['Volatility_Diff'] = df['Volatility'].diff()

df.dropna(inplace = True)

# df['Open_2nd_Diff'] = df['Open'].diff().diff()
# df['High_2nd_Diff'] = df['High'].diff().diff()
# df['Low_2nd_Diff'] = df['Low'].diff().diff()
# df['Adj_Close_2nd_Diff'] = df['Adj Close'].diff().diff()

"""Run ADF Test to determine relevant features"""

from statsmodels.tsa.stattools import adfuller

for col in df.columns:
    result = adfuller(df[col])
    print(f'Col: {col}')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('\n')

features = [ 'Log_Return', 'Volume']
# features = ['Log_Return', 'Volume', 'Open_Diff_2', 'High_Diff_2', 'Low_Diff_2']

from statsmodels.tsa.seasonal import seasonal_decompose

df_model = df[features].copy()

"""Normalize"""

scaler = StandardScaler()
df_model_scaled = pd.DataFrame(
    scaler.fit_transform(df_model),
    columns=df_model.columns,
    index=df_model.index
    )

"""Train Test Split

During Training, record the loss values and say if it converges to some stable values.
"""

train_size = int(len(df_model_scaled) * 0.75)
train_data = df_model_scaled.iloc[:train_size]
test_data = df_model_scaled.iloc[train_size:]

df_train = df_model.iloc[:train_size]
df_test = df_model.iloc[train_size:]

print(df_train.shape)
print(df_test.shape)

"""Model"""

forecasted_prices = []
actual_prices = []

model = VAR(train_data)
lag_results = {}
for lag in range(1, 11):
    result = model.fit(lag)
    lag_results[lag] = result.aic
    print(f"Lag {lag}: AIC = {result.aic}")

optimal_lag = min(lag_results, key=lag_results.get)
print(f"\nOptimal lag order: {optimal_lag}")

"""Fitting Model"""

optimal_lag = 10
var_model = VAR(train_data)
var_result = var_model.fit(optimal_lag)
print(var_result.summary())

"""Forecasting Prices"""

for i in range(len(test_data)):
    if i < optimal_lag:
        input_part_1 = train_data.values[-(optimal_lag - i):]
        input_part_2 = test_data.values[:i]
        input_data = np.vstack([input_part_1, input_part_2])
    else:
        input_data = test_data.values[i - optimal_lag:i]

    if input_data.shape[0] != optimal_lag:
        raise ValueError(f"Expected input_data with {optimal_lag} rows, got {input_data.shape[0]}")

    forecast_output = var_result.forecast(input_data, steps=1)
    forecast_log_return = forecast_output[0][0]

    yesterday_close = df['Close'].iloc[train_size + i - 1]
    predicted_price = yesterday_close * np.exp(forecast_log_return)

    forecasted_prices.append(predicted_price)
    actual_prices.append(df['Close'].iloc[train_size + i])

mse_var = mean_squared_error(actual_prices, forecasted_prices)
r2_var = r2_score(actual_prices, forecasted_prices)

predicted_returns = np.log(np.array(forecasted_prices[1:])/ np.array(forecasted_prices[:-1]))
actual_returns = df['Log_Return'].iloc[train_size + 1 : train_size + 1 + len(predicted_returns)]

sign_accuracy = np.mean(np.sign(predicted_returns) == np.sign(actual_returns))

print(f"VAR MSE: {mse_var:.6f}")
print(f"VAR R²: {r2_var:.4f}")
print(f"VAR Sign Accuracy: {sign_accuracy:.4f}")

"""Naive Previous Close"""

naive_predictions = df['Close'].shift(1).iloc[train_size : train_size + len(actual_prices)]
mse_naive = mean_squared_error(actual_prices, naive_predictions)
r2_naive = r2_score(actual_prices, naive_predictions)
naive_returns = df['Log_Return'].shift(1).iloc[train_size : train_size + len(actual_returns)].reset_index(drop=True)
actual_returns = actual_returns.reset_index(drop=True)

sign_accuracy_prev = np.mean(np.sign(naive_returns) == np.sign(actual_returns))

print(f"Naive Baseline MSE: {mse_naive:.6f}")
print(f"Naive Baseline R²: {r2_naive:.4f}")
print(f"Naive Previous Sign Accuracy: {sign_accuracy_prev:.4f}")

"""Naive 3-Day Average"""

df_baseline_3_day_drop_1st_row = df.dropna(subset=['Close', 'Naive_3_day_Close'])

naive_pred = df_baseline_3_day_drop_1st_row.loc[df_test.index, 'Naive_3_day_Close']
naive_mse = mean_squared_error(df_baseline_3_day_drop_1st_row.loc[df_test.index, 'Close'], naive_pred)
naive_r2 = r2_score(df_baseline_3_day_drop_1st_row.loc[df_test.index, 'Close'], naive_pred)

naive_3_day_returns = df['return_3d'].iloc[train_size:train_size + len(actual_returns)].reset_index(drop=True)
actual_returns = actual_returns.iloc[:len(naive_3_day_returns)].reset_index(drop=True)
sign_accuracy_3_day = np.mean(np.sign(naive_3_day_returns) == np.sign(actual_returns))

print(f"Naive 3-Day Baseline MSE: {naive_mse:.6f}")
print(f"Naive 3-Day Baseline R²: {naive_r2:.4f}")
print(f"Naive 3-Day Baseline Sign Accuracy: {sign_accuracy_3_day:.4f}")

"""Comparison"""

forecast_index = df.index[train_size:train_size + len(forecasted_prices)]

plt.figure(figsize=(20,6))
plt.plot(df['Close'].iloc[train_size:], label='Actual')
plt.plot(forecast_index, forecasted_prices, label='VAR Prediction', linestyle=(0, (1, 1)))
plt.legend()
plt.title('Close Prices: VAR Prediction vs. Actual')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.grid(True)
plt.show()

from sklearn.metrics import classification_report, mean_squared_error, r2_score
import numpy as np

predicted_labels = np.sign(np.array(forecasted_prices[1:]) - np.array(forecasted_prices[:-1]))  # 1 for up, -1 for down
actual_labels = np.sign(np.array(df['Close'].iloc[train_size + 1:]) - np.array(df['Close'].iloc[train_size:-1]))

print("=== VAR Model Performance ===")
accuracy_var = np.mean(predicted_labels == actual_labels)
print(f"Accuracy: {accuracy_var:.3f}\n")
print("Detailed report:")
print(classification_report(actual_labels, predicted_labels, target_names=["Down", "Up"]))