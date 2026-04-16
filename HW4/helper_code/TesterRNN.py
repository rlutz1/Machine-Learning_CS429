"""

THIS IS FOR TESTING DATA PREP ONLY

full disclosure, code is adapated from:
https://codesignal.com/learn/courses/introduction-to-rnns-for-time-series-analysis-1/lessons/building-a-basic-rnn-model-with-pytorch

"""

import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :])  # Take the last output of the sequence
        return out
        