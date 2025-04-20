import pandas as pd
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import classification_report

# Load the pretrained model
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_price = torch.nn.Linear(hidden_size, 1)
        self.fc_direction = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Use last output for prediction
        return self.fc_price(out), self.fc_direction(out)

# Load the scaler used for preprocessing in the original model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load NVIDIA stock data (NVDA)
df_nvda = pd.read_csv('data/raw/NVDA.csv', index_col='Date', parse_dates=True)

# Feature engineering on NVIDIA data (same as Google data)
df_nvda['5_MA'] = df_nvda['Close'].rolling(window=5).mean()
df_nvda['10_MA'] = df_nvda['Close'].rolling(window=10).mean()
df_nvda['10_STD'] = df_nvda['Close'].rolling(window=10).std()
df_nvda['Daily_Return'] = df_nvda['Close'].pct_change()
df_nvda['Momentum'] = df_nvda['Close'] - df_nvda['Close'].shift(10)

df_nvda.dropna(inplace=True)

features = ['Close', 'Open', 'High', 'Low', 'Volume', '5_MA', '10_MA', '10_STD', 'Daily_Return', 'Momentum']
data_nvda = df_nvda[features].values

# Normalize NVIDIA data using the same scaler
data_nvda_scaled = scaler.transform(data_nvda)

# Sliding window function for NVDA (same as for GOOG)
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

# Set the sequence length for the window
seq_length = 30
x_nvda, y_price_nvda, y_direction_nvda, current_close_nvda = sliding_windows(data_nvda_scaled, seq_length)

# Convert to torch tensors
x_nvda_tensor = torch.Tensor(x_nvda)
y_price_nvda_tensor = torch.Tensor(y_price_nvda)
y_direction_nvda_tensor = torch.Tensor(y_direction_nvda)

# Initialize the LSTM model (same structure as before)
input_size = x_nvda.shape[2]  # This should be the same as the input size used in training
hidden_size = 64
num_layers = 1

lstm = LSTM(input_size, hidden_size, num_layers)

# Load the pretrained weights
lstm.load_state_dict(torch.load("model.pt"))

# Set the model to evaluation mode
lstm.eval()

# Inference for NVIDIA stock data
with torch.no_grad():
    nvda_price_pred, nvda_direction_pred = lstm(x_nvda_tensor)

# Inverse transform the predictions to the original price scale
nvda_price_pred = nvda_price_pred.detach().numpy()
nvda_price_true = scaler.inverse_transform(np.hstack([nvda_price_pred, np.zeros((len(nvda_price_pred), input_size - 1))]))[:, 0]

# Calculate RMSE for price prediction
nvda_rmse = np.sqrt(np.mean((nvda_price_true - y_price_nvda) ** 2))

# Evaluate the direction accuracy
nvda_direction_pred = (nvda_direction_pred.detach().numpy() > 0).astype(int)

# Classification Report for Direction Prediction
classification_report_nvda = classification_report(y_direction_nvda, nvda_direction_pred.flatten(), target_names=["Down", "Up"])

# Direction accuracy
nvda_acc = np.mean(nvda_direction_pred.flatten() == y_direction_nvda)

# Print the results
print(f"NVIDIA Direction Prediction Accuracy: {nvda_acc:.2f}")
print(f"NVIDIA RMSE: {nvda_rmse:.2f}")
print("\nClassification Report for Direction Prediction:")
print(classification_report_nvda)
