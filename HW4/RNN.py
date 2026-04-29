"""
Task 2: Defining an RNN
Predict next day's closing price (N=1) from the past M=60 days.
Baseline: 1 recurrent layer.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from DataProcessor import DataProcessor


class StockRNN(nn.Module):

    def __init__(self, input_size=5, hidden_size=32, output_size=1):
        super(StockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last timestep's hidden state
        return out


def train(model, X_train, y_train, epochs=100, lr=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    y_tensor = y_train.clone().detach().float()

    losses = []
    start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)


    elapsed = time.time() - start
    return losses, elapsed


def mape(model, X, y, dp):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze(-1).numpy()
    preds_orig = dp.inverse_target(preds)
    y_orig = dp.inverse_target(y)
    return np.mean(np.abs((y_orig - preds_orig) / y_orig)) * 100


def run():
    torch.manual_seed(42)
    np.random.seed(42)

    dp = DataProcessor(
        start_date="2020-01-01",
        end_date="2024-01-02",
        window_size=60,
        data_grab=False,
        clean=False
    )

    results = {}

    for name, symbol in dp.company_symbols.items():

        print(f"\nStock: {name} ({symbol})")

        dp.split(symbol)

        train_windows = len(dp.X_train)
        test_windows = len(dp.X_test)
        total_windows = train_windows + test_windows
        print(f"  Train windows : {train_windows} ({100 * train_windows / total_windows:.1f}%)")
        print(f"  Test windows  : {test_windows} ({100 * test_windows / total_windows:.1f}%)")
        print(f"  Total windows : {total_windows}")

        hidden_size = 64 if symbol == "NVDA" else 32
        model = StockRNN(hidden_size=hidden_size)

        losses, elapsed = train(model, dp.X_train, dp.y_train, epochs=100, lr=0.001, batch_size=32)

        train_mape = mape(model, dp.X_train, dp.y_train, dp)
        test_mape = mape(model, dp.X_test, dp.y_test, dp)

        print(f"  Training time : {elapsed:.2f}s")
        print(f"  Train MAPE    : {train_mape:.4f}%  (accuracy: {100 - train_mape:.2f}%)")
        print(f"  Test  MAPE    : {test_mape:.4f}%  (accuracy: {100 - test_mape:.2f}%)")

        results[name] = {"losses": losses, "train_mape": train_mape, "test_mape": test_mape, "elapsed": elapsed}

        # --- loss curve ---
        plt.figure()
        plt.plot(losses)
        plt.title(f"RNN Training Loss — {name} ({symbol})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.tight_layout()
        plt.savefig(f"data/{symbol}_rnn_loss.png")
        plt.close()

        # --- prediction vs actual ---
        model.eval()
        with torch.no_grad():
            preds = model(dp.X_test).squeeze(-1).numpy()
        preds_orig = dp.inverse_target(preds)
        y_orig = dp.inverse_target(dp.y_test)

        plt.figure()
        plt.plot(y_orig, label="Actual")
        plt.plot(preds_orig, label="Predicted")
        plt.title(f"RNN Predictions vs Actual — {name} ({symbol})")
        plt.xlabel("Day")
        plt.ylabel("Close Price ($)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/{symbol}_rnn_predictions.png")
        plt.close()

    return results


if __name__ == "__main__":
    run()
