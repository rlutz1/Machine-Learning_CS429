import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

"""
Feed-Forward Neural Network using PyTorch with CUDA support

Will automatically use CUDA if available.
"""

class FeedForwardNN(nn.Module):
    def __init__(self, n_features, hidden_layers=[128, 64], dropout=0.3):

        super(FeedForwardNN, self).__init__()

        layers = []
        input_size = n_features

        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU()) # Use ReLU but may change to sigmoid for tuning
            # layers.append(nn.Dropout(dropout)) # TODO commenting JUST to be safe
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetworkClassifier:
    """
    Wrapper class for training and evaluation
    """
    def __init__(self, n_features, hidden_layers=[128, 64], eta=0.001,
                 n_iter=50, batch_size=64, dropout=0.3, random_state=42):

        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

        # Random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Error print to figure out which device im actually using and if it's using  cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Initialize model
        self.model = FeedForwardNN(n_features, hidden_layers, dropout).to(self.device)

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=eta)

        # Track losses and timing
        self.losses_ = []
        self.training_time_ = 0.0
        self.test_time_ = 0.0

    def fit(self, X, y):
        start_time = time.time()

        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Copies to ensure arrays are writable (got a warning i dont like)
        X = np.array(X, copy=True) # NOTE: i got an error on laptop since it was trying to allocate 23.3 GB, lmao, and couldn't
        y = np.array(y, copy=True)
        # looking further: potentially very bad idea -- since we are testing accuracy on this array later and python passes addresses. moral: try running elsewhere, ha
        # X.setflags(write=True) # NOTE: i see the warning, potential fix to avoid a huge copy. sounds like numpy, torch share the memory and that's why its a big scary warning.
        # y.setflags(write=True) # so maybe undefined behaviour is if you're looking at the data later? and things were overwritten? need more research

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.n_iter):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            # Average loss for epoch
            avg_loss = epoch_loss / len(dataset)
            self.losses_.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{self.n_iter}, Loss: {avg_loss:.4f}')

        self.training_time_ = time.time() - start_time
        print(f'\nTraining completed in {self.training_time_:.2f} seconds')

        return self

    def predict_proba(self, X, time_it=False):
        if time_it:
            start_time = time.time()

        self.model.eval()

        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        result = outputs.cpu().numpy().flatten()

        if time_it:
            elapsed = time.time() - start_time
            self.test_time_ = elapsed
            print(f'Prediction completed in {elapsed:.4f} seconds')

        return result

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    from DataProcessor import DataProcessor
    import matplotlib.pyplot as plt

    dp = DataProcessor()

    print(f"Training samples: {dp.train_reviews.shape[0]}")
    print(f"Test samples: {dp.test_reviews.shape[0]}")
    print(f"Feature dimension: {dp.train_reviews.shape[1]}")

    # Initialize neural network
    nn = NeuralNetworkClassifier(
        n_features=dp.train_reviews.shape[1],
        hidden_layers=[256, 128, 64], # Three hidden layers
        eta=0.001, # Learning rate
        n_iter=30, # Epochs
        batch_size=128, # Mini-batch size
        dropout=0.0, # Dropout rate # TODO: ZEROING OUT FOR TESTING BASELINE
        random_state=42
    )

    nn.fit(dp.train_reviews, dp.train_sentiments)

    train_start = time.time()
    train_acc = nn.score(dp.train_reviews, dp.train_sentiments)
    train_eval_time = time.time() - train_start

    test_start = time.time()
    test_acc = nn.score(dp.test_reviews, dp.test_sentiments)
    test_eval_time = time.time() - test_start

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nTraining Time: {nn.training_time_:.2f} seconds")
    print(f"Train Evaluation Time: {train_eval_time:.4f} seconds")
    print(f"Test Evaluation Time: {test_eval_time:.4f} seconds")

    # Plot loss convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(nn.losses_) + 1), nn.losses_, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Binary Cross-Entropy)')
    plt.title('Training Loss Convergence')
    plt.grid(True)
    plt.show()