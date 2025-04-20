import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
df = pd.read_csv('data/raw/GOOG.csv', index_col='Date', parse_dates=True)

# Feature engineering
df['5_MA'] = df['Close'].rolling(window=5).mean()
df['10_MA'] = df['Close'].rolling(window=10).mean()
df['10_STD'] = df['Close'].rolling(window=10).std()
df['Daily_Return'] = df['Close'].pct_change()
df['Momentum'] = df['Close'] - df['Close'].shift(10)

df.dropna(inplace=True)

features = ['Close', 'Open', 'High', 'Low', 'Volume', '5_MA', '10_MA', '10_STD', 'Daily_Return', 'Momentum']
data = df[features].values

# Normalize each feature
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Sliding window
def sliding_windows(data, seq_length, future_offset=30):
    x, y_price, y_direction, current_close = [], [], [], []
    for i in range(len(data) - seq_length - future_offset):
        _x = data[i:(i + seq_length)]
        close_now = data[i + seq_length - 1][0]
        close_future = data[i + seq_length + future_offset - 1][0]

        x.append(_x)
        y_price.append(close_future)
        y_direction.append(1 if close_future > close_now else 0)
        current_close.append(close_now)
    return np.array(x), np.array(y_price), np.array(y_direction), np.array(current_close)

seq_length = 30
x, y_price, y_direction, current_close = sliding_windows(data_scaled, seq_length)

# Train-test split
train_size = int(len(y_price) * 0.75)

trainX = torch.Tensor(x[:train_size])
trainY_price = torch.Tensor(y_price[:train_size])
trainY_direction = torch.Tensor(y_direction[:train_size])
trainClose = torch.Tensor(current_close[:train_size])

testX = torch.Tensor(x[train_size:])
testY_price = torch.Tensor(y_price[train_size:])
testY_direction = torch.Tensor(y_direction[train_size:])
testClose = torch.Tensor(current_close[train_size:])

# Define LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_price = nn.Linear(hidden_size, 1)
        self.fc_direction = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Use last output for prediction
        return self.fc_price(out), self.fc_direction(out)

# Hyperparameters
input_size = x.shape[2]
hidden_size = 64
num_layers = 1
num_epochs = 250
learning_rate = 0.005

# Initialize model
lstm = LSTM(input_size, hidden_size, num_layers)
price_loss_fn = nn.MSELoss()
direction_loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    lstm.train()
    price_pred, direction_pred = lstm(trainX)

    price_loss = price_loss_fn(price_pred.squeeze(), trainY_price)
    direction_loss = direction_loss_fn(direction_pred.squeeze(), trainY_direction)
    total_loss = price_loss + direction_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Total Loss: {total_loss.item():.5f}, "
              f"Price Loss: {price_loss.item():.5f}, Direction Loss: {direction_loss.item():.5f}")

# Save model
torch.save(lstm.state_dict(), "model.pt")

# Reload model for inference
# Initialize model again
lstm = LSTM(input_size, hidden_size, num_layers)

# Load model weights
lstm.load_state_dict(torch.load("model.pt"))

# Set model to evaluation mode
lstm.eval()

# Inference
train_price_pred, train_direction_pred = lstm(trainX)
test_price_pred, test_direction_pred = lstm(testX)

# Inverse transform price predictions
train_price_pred = train_price_pred.detach().numpy()
test_price_pred = test_price_pred.detach().numpy()

train_price_true = scaler.inverse_transform(np.hstack([train_price_pred, np.zeros((len(train_price_pred), input_size - 1))]))[:, 0]
test_price_true = scaler.inverse_transform(np.hstack([test_price_pred, np.zeros((len(test_price_pred), input_size - 1))]))[:, 0]

train_y_price_true = scaler.inverse_transform(np.hstack([trainY_price.view(-1,1).numpy(), np.zeros((len(trainY_price), input_size - 1))]))[:, 0]
test_y_price_true = scaler.inverse_transform(np.hstack([testY_price.view(-1,1).numpy(), np.zeros((len(testY_price), input_size - 1))]))[:, 0]

# Direction accuracy
train_direction_pred = (train_direction_pred.detach().numpy() > 0).astype(int)
test_direction_pred = (test_direction_pred.detach().numpy() > 0).astype(int)

train_acc = np.mean(train_direction_pred.flatten() == trainY_direction.numpy())
test_acc = np.mean(test_direction_pred.flatten() == testY_direction.numpy())

print(f"\nTrain Direction Accuracy: {train_acc:.2f}")
print(f"Test Direction Accuracy: {test_acc:.2f}")

# RMSE
train_rmse = np.sqrt(np.mean((train_price_true - train_y_price_true) ** 2))
test_rmse = np.sqrt(np.mean((test_price_true - test_y_price_true) ** 2))

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Classification Report
print("\nClassification Report for Direction Prediction:")
print(classification_report(testY_direction.numpy(), test_direction_pred.flatten(), target_names=["Down", "Up"]))

# ---- Naive Baseline Model ----
# Naive price prediction: predict the price will be the same as current close
naive_price_pred = testClose.numpy()

# RMSE for naive model
naive_rmse = np.sqrt(np.mean((naive_price_pred - test_y_price_true) ** 2))
print(f"\nNaive Baseline RMSE: {naive_rmse:.2f}")

# Naive direction prediction: predict the price will go up (1) always
naive_direction_pred = np.ones_like(testY_direction.numpy())

# Direction accuracy for naive model
naive_acc = np.mean(naive_direction_pred.flatten() == testY_direction.numpy())
print(f"Naive Baseline Direction Accuracy: {naive_acc:.2f}")

# Classification report for naive model
print("\nClassification Report for Naive Direction Prediction:")
print(classification_report(testY_direction.numpy(), naive_direction_pred.flatten(), target_names=["Down", "Up"]))

