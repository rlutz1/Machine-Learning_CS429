"""

THIS IS FOR TESTING DATA PREP ONLY

full disclosure, code is adapated from:
https://codesignal.com/learn/courses/introduction-to-rnns-for-time-series-analysis-1/lessons/building-a-basic-rnn-model-with-pytorch

"""


import torch
import torch.nn as nn
import numpy as np

class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :])  # Take the last output of the sequence
        return out
    
    def predict(self, X):
        # probabilities = self.predict_proba(X)
        # return (probabilities >= 0.5).astype(int)
        # pass
        return self.forward(X)

    def score(self, X, y):
        self.eval() # hypothetically good, need to research
        torch.no_grad()
        # use MAPE
        predictions = self.predict(X).detach().numpy().reshape(-1)
        mape = 0
        for actual, p in zip(y, predictions):
          mape += abs((actual - p) / actual) 

        mape = (mape / y.shape[0]) * 100
        # correct = 0
        # for pred in predictions:  
        #   correct += (pred == y).type(torch.float).sum().item()
        return mape
        