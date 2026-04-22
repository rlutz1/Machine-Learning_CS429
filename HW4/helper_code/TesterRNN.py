"""

THIS IS FOR TESTING DATA PREP ONLY

full disclosure, code is adapated from:
https://codesignal.com/learn/courses/introduction-to-rnns-for-time-series-analysis-1/lessons/building-a-basic-rnn-model-with-pytorch

"""


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error

class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        # print(f"out: {out}\n hidden: {hidden}")
        out = self.linear(out[:, -1, :])  # Take the last output of the sequence
        return out
    
    # completely ripped from internet for TESTING DATA PREPROCESS ONLY
    def fit(self, X, y):
      # Define loss function and optimizer
      criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
      optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer for updating model weights

      # Training loop parameters
      epochs = 25
      batch_size = 16

      # List to store loss values for each epoch (for plotting later)
      losses = []

      for epoch in range(epochs):
          self.train()  # Set the model to training mode
          epoch_loss = 0  # Track loss for this epoch

          # Loop over the training data in batches
          for i in range(0, len(X), batch_size):
              # Prepare batch data as PyTorch tensors
              X_batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
              y_batch = torch.tensor(y[i:i+batch_size], dtype=torch.float32)
              # print(X_batch)
              # ---- Forward Pass ----
              # Pass the input batch through the model to get predictions
              outputs = self.forward(X_batch)
              # outputs = model(X_batch) # i've heard this is better practice, yielding same results for now

              # Compute the loss between predictions and actual values
              # print(outputs.size())
              loss = criterion(outputs.squeeze(-1), y_batch) # squeezing to avoid scary error

              # ---- Backpropagation ----
              # Zero the gradients from the previous step
              optimizer.zero_grad()
              # Compute gradients of the loss with respect to model parameters
              loss.backward()
              # Update model parameters using the optimizer
              optimizer.step()

              # Accumulate loss for this batch
              epoch_loss += loss.item()

          # Calculate average loss for the epoch and store it
          avg_loss = epoch_loss / (len(X) // batch_size)
          losses.append(avg_loss)
          print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    def predict(self, X):
        return self.forward(X)

    def mape(self, X, y, dp, epsilon=1e-8):
        self.eval() # hypothetically good, need to research
        with torch.no_grad():

          # use MAPE for testing
          predictions = self.predict(X).detach().numpy().reshape(-1)


        # heads up that mape is VERY sensitive to near 0 values. 
        # either we can change the scale range to something like (0.01, 1.01) 
        # with accuracy impacts OR deal with the 0's here.
        # caution suggested lol
        # mape = 0
        # for actual, p in zip(y, predictions):
        #   mape += abs((actual - p) / (actual)) # idk wtf is going on with the training data
        #   # if (actual != 0):
        #   #   mape += abs((actual - p) / (actual))
        # mape = (mape / y.shape[0]) * 100

        # invert the scaled costs back to originals due to MAPE sensitivity
        predictions_orginals = dp.inverse_target(predictions) 
        y_orginals = dp.inverse_target(y)

        # another way to get mape, yields same as above
        mape = np.mean(np.abs((y_orginals - predictions_orginals) / y_orginals)) * 100

        return mape
    
    def mape_loss(y_pred, y_true, epsilon=1e-8):
      # Calculate absolute relative error
      absolute_percentage_error = torch.abs((y_true - y_pred) / (y_true + epsilon))
      # Return mean across all samples
      return torch.mean(absolute_percentage_error) * 100
        