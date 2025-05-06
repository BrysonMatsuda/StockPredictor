# Note: The context for this model is that it is currently open time (start of trading day) and we
# are trying to predict if/how to invest our money. Therefore we use past data (yesterday and before) 
# to predict the stock trends within the next few coming days. The model uses multi-day trends in predictions
# since predicting trends for a single day has a lot of noise involved. The baseline model naively assumes
# the next trend will follow the average of the return from the last 3 days.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ta
from scipy.stats import randint, uniform
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
    df['Target'] = (df['Close'].pct_change().shift(-1).rolling(3).mean())
    
    # smoothed return starting EOD today and ending 2 days from now (3 total)

    
    df['Target'] = (df['Target'] > 0).astype(int)
elif TARGET_TYPE == 'return': # return from EOD today to EOD tomorrow
    df['Target'] = df['Close'].pct_change().shift(-1)
    df['Target'] = (df['Target'] > 0).astype(int)
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

param_dist = {
    "n_estimators":   randint(100, 701),      # draws integers 100‑700
    "max_depth":      [None, 8, 9, 10, 11, 12], # best model (shallow): [3, 4, 5, 8, 9, 10, 11, 12]    [2, 3, 4, 5, 8, 9, 10, 11, 12]  Deeper model (overfit): [None, 8, 9, 10, 11, 12]
    "min_samples_split":  randint(2, 11),
    "min_samples_leaf":   randint(1, 6),
    "max_features":   uniform(0.2, 0.8),      # real numbers 0.2‑1.0
    "bootstrap":      [True, False],
    "random_state":   [42],
}

tscv = TimeSeriesSplit(n_splits=2)

# Model
model = RandomForestClassifier()
#model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=2, random_state=42)
#model.fit(X_train, y_train)

grid = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    cv=tscv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=10,
    random_state=42,
)

grid.fit(X_train, y_train)

print("\n=== Best hyper‑parameters (CV) ===")
print(grid.best_params_)

model = grid.best_estimator_

y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"\n=== Training Accuracy ===\nAccuracy: {train_acc:.3f}")

joblib.dump(model, "best_random_forest_classifier.pkl")
y_pred = model.predict(X_test)

# feature importance
feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Top 20 features ===")
print(feat_imp.head(20).to_string(float_format="%.4f"))

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Model Performance ===")
print(f"Accuracy: {acc:.3f}")
print("\nDetailed report:\n", classification_report(y_test, y_pred))

# Baseline 
naive_pred = (X_test['return_3d'] > 0).astype(int)
naive_acc  = accuracy_score(y_test, naive_pred)
print(f"\n=== Naive Baseline ===\nAccuracy: {naive_acc:.3f}")
print("\nDetailed report:\n", classification_report(y_test, naive_pred))
