# Note: The context for this model is that it is currently open time (start of trading day) and we
# are trying to predict if/how to invest our money. Therefore we use past data (yesterday and before) 
# to predict if the stock goes up/down within the next few coming days. The model uses multi-day trends in predictions
# since predicting trends for a single day has a lot of noise involved. The baseline model naively assumes
# the next trend will follow the average of the return from the last 3 days.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)
import ta

# CONFIG 
CSV_PATH = 'GOOG.csv'
TEST_SIZE = 0.2
THRESHOLD = 0.5
TARGET_TYPE = 'smoothed_return' # options 'smoothed_return' , 'oneday_return'

# LOAD DATA 
df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# FEATURE ENGINEERING

# Price lags
for lag in range(1, 6):
    df[f'Close_t{lag}'] = df['Close'].shift(lag)

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

# TARGET
if TARGET_TYPE == "smoothed_return":
    forward_return = df['Close'].pct_change().shift(-2).rolling(4).mean() # smoothed return starting EOD yesterday and ending 3 days from now (4 total)
    df['Target'] = (forward_return > 0).astype(int) # up/down prediction is if it goes up/down within the next 3 days
elif TARGET_TYPE == "oneday_return":
    forward_return = df['Close'].pct_change()
    df['Target'] = (forward_return > 0).astype(int)

# CLEANUP 
# Drop NaNs and remove todays attributes so they aren't used to train and bleed info
df.dropna(inplace=True)
df.drop(columns=['Volume', 'Close', 'High', 'Low', 'Adj Close'], inplace=True)

# TRAIN/TEST SPLIT 
features = [col for col in df.columns if col != 'Target']
X = df[features]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

# MODEL 
model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

# CLASSIFICATION REPORT 
print("\nClassification Performance ")
print(f"Accuracy (Threshold = {THRESHOLD}): {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

# BASELINE 
# Just if the average of the return from the last 3 days is positive or negative
naive_pred = (X_test['return_3d'] > 0).astype(int)
print("\nNaive Baseline:")
print(f"Accuracy: {accuracy_score(y_test, naive_pred):.2f}")
print(classification_report(y_test, naive_pred, target_names=["Down", "Up"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, naive_pred))