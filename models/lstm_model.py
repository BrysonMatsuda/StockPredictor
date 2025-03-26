import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

# Load dataset
df = pd.read_csv('data/raw/GOOG.csv', index_col='Date', parse_dates=True)

# Select features (Close, Open, High, Low, Volume)
features = ['Close', 'Open', 'High', 'Low', 'Volume']
df = df[features]

# Generate binary target: 1 if next day's close is higher, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop last row (since it has no target)
df = df[:-1]

# Normalize all features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

plt.figure(figsize=(12,6))  # Set figure size
plt.plot(df.index, df['Close'], label='Close Price')  # x-axis: date, y-axis: close price
plt.xlabel('Date')
plt.ylabel('Close Price (Normalized)')
plt.title('Stock Close Price Over Time')
plt.legend()
plt.show()


SEQ_LEN = 30  # Sequence length for LSTM

def create_sequences(data, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

# Convert DataFrame to numpy arrays
X, y = create_sequences(df[features].values, df['Target'].values, SEQ_LEN)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(StockLSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = SEQ_LEN   

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        ula, (h_out, _) = self.lstm(x, (h_0, c_0)) # call the lstm layer
        
        h_out = h_out.view(-1, self.hidden_size) # change the tensor shape
        
        out = self.fc(h_out) # call the fully connected layer

        out = self.sigmoid(out)
        
        return out

input_size = 5 # we only have one feature
hidden_size = 50 # controls how large the model is
num_layers = 1 # number of LSTMs (if more than one, then one LSTM goes into the other)

num_classes = 1 # we only have one class
model = StockLSTM(num_classes, input_size, hidden_size, num_layers)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Lower learning rate

# Training loop
EPOCHS = 500
BATCH_SIZE = 32
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
    y_pred_class = (y_pred > 0.5).astype(int)

# Accuracy
accuracy = (y_pred_class.flatten() == y_test.numpy().flatten()).mean()
print(f"Test Accuracy: {accuracy:.2f}")

# Plot predictions
plt.plot(y_test.numpy(), label="Actual", alpha=0.7)
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.show()
