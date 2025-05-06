# Note: The context for this model is that it is currently open time (start of trading day) and we
# are trying to predict if/how to invest our money. Therefore we use past data (yesterday and before) 
# to predict the stock trends within the next few coming days. The model uses multi-day trends in predictions
# since predicting trends for a single day has a lot of noise involved. The baseline model naively assumes
# the next trend will follow the average of the return from the last 3 days.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import ta
import joblib

# === CONFIG ===
CSV_PATH = 'GOOG.csv'
TARGET_TYPE = 'return_smoothed'  # options: 'return_smoothed', 'price', 'price_smoothed', 'return'
TEST_SIZE = 0.25

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# === FEATURE ENGINEERING ===

# Price lags
for lag in range(1, 6):
    df[f'Close_t-{lag}'] = df['Close'].shift(lag)

# Returns over past intervals (last day, last 3 days, last 5 days)
# shifted 1 day forward so we don't leak info about EOD today
df['return_1d'] = df['Close'].pct_change(1).shift(1)
df['return_3d'] = df['Close'].pct_change(3).shift(1)
df['return_5d'] = df['Close'].pct_change(5).shift(1)

# Moving averages
# first we get the sum of the past 4 or 19 days and current day closing price, average them, and shift it forward one day in the dataset so we only use past prices (i.e MA for last 5 days)
df['ma5'] = df['Close'].rolling(5).mean().shift(1)
df['ma20'] = df['Close'].rolling(20).mean().shift(1)

# Standard deviations
# first we get the sum of the past 4 or 19 days and current day closing price, get the the standard deviation, and shift it forward one day in the dataset so we only use past prices (i.e STD for last 5 days)
df['std5'] = df['Close'].rolling(5).std().shift(1)
df['std20'] = df['Close'].rolling(20).std().shift(1)

# Technical indicators
# shifted 1 day forward so we don't leak info about EOD today
df['rsi14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().shift(1)
macd = ta.trend.MACD(df['Close'])
df['macd'] = macd.macd().shift(1)
df['macd_signal'] = macd.macd_signal().shift(1)

# Lag attributes so we don't use data from EOD today to train
df['Volume_t-1'] = df['Volume'].shift(1)
df['Open_t-1'] = df['Open'].shift(1)
df['High_t-1'] = df['High'].shift(1)
df['Low_t-1'] = df['Low'].shift(1)
df['Adj Close_t-1'] = df['Adj Close'].shift(1)

# === TARGET ===
if TARGET_TYPE == 'return_smoothed':
    df['Target'] = (df['Close'].pct_change().shift(-1).rolling(3).mean()) # smoothed return starting EOD today and ending 2 days from now (3 total)
elif TARGET_TYPE == 'price':
    df['Target'] = df['Close'] # price at the end of the day
elif TARGET_TYPE == 'price_smoothed':
    df['Target'] = df['Close'].shift(-1).rolling(3).mean() # average price at the end of the day for next 3 days
elif TARGET_TYPE == 'return': # return from EOD today to EOD tomorrow
    df['Target'] = (df['Close'].pct_change().shift(-1))
else:
    raise ValueError("Invalid TARGET_TYPE selected.")

# Drop NaNs and remove todays attributes so they aren't used to train and bleed info
df.dropna(inplace=True)
df.drop(columns=['Volume'], inplace=True)
df.drop(columns=['Close'], inplace=True)
df.drop(columns=['High'], inplace=True)
df.drop(columns=['Low'], inplace=True)
df.drop(columns=['Adj Close'], inplace=True)

# === MODEL SETUP ===
features = [col for col in df.columns if col != 'Target']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

param_distributions = {
    "n_estimators": randint(200, 701),               # range 200–700
    "max_depth": [None, 6, 8, 10, 12],     # None, 6, 8, 10, 12
    "min_samples_split": randint(2, 11),             # 2–10
    "min_samples_leaf": randint(1, 5),               # 1–4
    "max_features": ["sqrt", "log2", uniform(0.3, 0.7)],  # discrete and continuous options
    "bootstrap": [True, False],
    "random_state": [42],
}

tscv = TimeSeriesSplit(n_splits=5)

model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_split=2, random_state=42)

grid = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=tscv,
    scoring="r2",
    n_jobs=-1,
    verbose=10,
    random_state=42,
)

#grid.fit(X_train, y_train)
model.fit(X_train, y_train)

print("\n=== Best hyper‑parameters (CV) ===")
#print(grid.best_params_)

#model = grid.best_estimator_
#joblib.dump(model, "best_random_forest_regressor.pkl")

# Training evaluation
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_directional_acc = np.mean((np.sign(y_train_pred) == np.sign(y_train)).astype(float))

print("\n=== Training Performance ===")
print(f"{TARGET_TYPE.upper()} | MSE: {train_mse:.6f} | RMSE: {train_rmse:.6f} | R²: {train_r2:.4f} | Directional Accuracy: {train_directional_acc:.2f}")


y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
# directional_acc = np.mean((np.sign(y_pred) == np.sign(y_test)).astype(float))

print("\n=== Model Performance ===")
#print(f"{TARGET_TYPE.upper()} | MSE: {mse:.6f} | R²: {r2:.4f}")
directional_acc = np.mean((np.sign(y_pred) == np.sign(y_test)).astype(float))
print(f"{TARGET_TYPE.upper()} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | R²: {r2:.4f} | Directional Accuracy: {directional_acc:.2f}")

# Baseline 
naive_pred = X_test['Close_t-1'] if TARGET_TYPE.startswith('price') else X_test['return_3d']/3
naive_mse = mean_squared_error(y_test, naive_pred)
naive_r2 = r2_score(y_test, naive_pred)
# naive_directional_acc = np.mean((np.sign(naive_pred) == np.sign(y_test)).astype(float))

print("\n=== Naive Baseline ===")
#print(f"MSE: {naive_mse:.6f} | R²: {naive_r2:.4f}")
naive_directional_acc = np.mean((np.sign(naive_pred) == np.sign(y_test)).astype(float))
print(f"MSE: {naive_mse:.6f} | R²: {naive_r2:.4f} | Directional Accuracy: {naive_directional_acc:.2f}")

# Plot results
plt.figure(figsize=(18, 6))
plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
plt.plot(y_test.index, y_pred, label='Prediction', alpha=0.7)
plt.plot(y_test.index, naive_pred.values, label='Naive', alpha=0.5)
plt.title(f"Stock Prediction ({TARGET_TYPE})")
plt.xlabel("Date")
plt.ylabel("Target Value")
plt.legend()
plt.tight_layout()
plt.show()