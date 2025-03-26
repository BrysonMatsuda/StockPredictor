import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/GOOG.csv', index_col='Date', parse_dates=True)

# Select features (Close, Open, High, Low, Volume)
features = ['Close', 'Open', 'High', 'Low', 'Volume']
df = df[features]

# Generate binary target: 1 if next day's close is higher, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop last row (since it has no target)
df = df[:-1]

import matplotlib.pyplot as plt

training_set = df.iloc[:, 0:1].values
plt.plot(training_set)
plt.show()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

import pickle

with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)


def sliding_windows(data, seq_length):
  # initialize the x and y parts
  x = []
  y = []
  
  for i in range(len(data)-seq_length-1): # sliding window
    _x = data[i:(i+seq_length)] # get the input array of size seq_length
    _y = data[i+seq_length] # get the corresponding  output
    x.append(_x) # add the input array to the x part
    y.append(_y) # add the output to the y part
  
  return np.array(x),np.array(y) # convert the parts to numpy arrays and return them

seq_length = 30 # sequence length of 30 days
x, y = sliding_windows(training_data, seq_length)



import torch
from torch.autograd import Variable

train_size = int(len(y) * 0.75) # training set is 75% of data
test_size = len(y) - train_size # test set is the rest

# saving the entire data to evaluate performance later
dataX = Variable(torch.Tensor(np.array(x))) # convert the x part to a tensor
dataY = Variable(torch.Tensor(np.array(y))) # convert the y part to a tensor

# training set
trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

# test set
testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))



import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # initial hidden state
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # initial cell state
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        ula, (h_out, _) = self.lstm(x, (h_0, c_0)) # call the lstm layer
        
        h_out = h_out.view(-1, self.hidden_size) # change the tensor shape
        
        out = self.fc(h_out) # call the fully connected layer
        
        return out


num_epochs = 1000 # number of times training goes through entire dataset
learning_rate = 0.01 # controls how fast the model's weights change

input_size = 1 # we only have one feature
hidden_size = 50 # controls how large the model is
num_layers = 1 # number of LSTMs (if more than one, then one LSTM goes into the other)

num_classes = 1 # we only have one class


lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



torch.save(lstm, "model.pt")



lstm.eval() # essentially switched the "mode" to be used for inference
train_predict = lstm(dataX) # run the model on the entire dataset (both training and test)

data_predict = train_predict.data.numpy() # get the predictions as a numpy array
dataY_plot = dataY.data.numpy() # get the ground truth as a numpy array

# undo the scaling
data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--') # draw a line between training and test data

# plot the lines
plt.plot(dataY_plot)
plt.plot(data_predict)
plt.show()




# gather predictions
train_predict = lstm(trainX)
test_predict = lstm(testX)

# convert data to numpy arrays
train_predict = train_predict.data.numpy()
test_predict = test_predict.data.numpy()
trainY = trainY.data.numpy()
testY = testY.data.numpy()

# undo the scaling
train_predict = sc.inverse_transform(train_predict)
trainY = sc.inverse_transform(trainY)
test_predict = sc.inverse_transform(test_predict)
testY = sc.inverse_transform(testY)

# calculate the RMSE
train_score = np.sqrt(np.mean((train_predict - trainY) ** 2))
print('Train Score: %.2f RMSE' % (train_score))
test_score = np.sqrt(np.mean((test_predict - testY) ** 2))
print('Test Score: %.2f RMSE' % (test_score))