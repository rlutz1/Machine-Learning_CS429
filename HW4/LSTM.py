from sklearn.preprocessing import MinMaxScaler
from DataProcessor import DataProcessor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt  
import os                         


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        pred = self.fc(last_out)
        return pred.squeeze(-1)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        lr=0.001,
        weight_decay=1e-5,
        patience=10,
        verbose=False
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_loader.dataset)

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    X_val_d = X_val.to(device)
                    y_val_d = y_val.to(device)
                    val_preds = self(X_val_d)
                    val_loss = criterion(val_preds, y_val_d).item()

                if verbose:
                    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print("Early stopping triggered.")
                        break
            else:
                if verbose:
                    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f}")

        if best_state is not None:
            self.load_state_dict(best_state)

    def predict(self, X):
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            preds = self(X.to(device)).cpu().numpy()
        return preds

    def mape(self, X, y, dp):
        y_true_scaled = y.detach().cpu().numpy()
        y_pred_scaled = self.predict(X)

        y_true = dp.inverse_target(y_true_scaled)
        y_pred = dp.inverse_target(y_pred_scaled)

        epsilon = 1e-8
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


# plotting function
def plot_predictions(model, dp, stock_name, symbol):
    y_pred_scaled = model.predict(dp.X_test)
    y_true_scaled = dp.y_test.detach().cpu().numpy()

    y_pred = dp.inverse_target(y_pred_scaled)
    y_true = dp.inverse_target(y_true_scaled)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True Price")
    plt.plot(y_pred, label="Predicted Price")

    plt.title(f"{stock_name} ({symbol}) - True vs Predicted Prices")
    plt.xlabel("Time (days)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{symbol}_prediction.png"
    plt.savefig(filename)
    plt.close()

    print(f"Saved plot: {filename}")


# reproducibility
torch.manual_seed(42)
np.random.seed(42)

dp = DataProcessor(
    data_grab=False,
    clean=False,
    target="Close",
    start_date="2020-01-01",
    end_date="2024-01-02",
    scaler=MinMaxScaler(),
    training_percent=0.8,
    window_size=60
)

stocks = ["Microsoft", "Google", "Amazon", "NVIDIA"]

results = []

for stock in stocks:
    symbol = dp.company_symbols[stock]

    dp.split(symbol)

    train_windows = dp.X_train.shape[0]
    test_windows = dp.X_test.shape[0]
    total_windows = train_windows + test_windows

    train_pct = (train_windows / total_windows) * 100
    test_pct = (test_windows / total_windows) * 100

    model = LSTMModel(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1
    )

    start_time = time.time()

    model.fit(
        dp.X_train,
        dp.y_train,
        X_val=dp.X_test,
        y_val=dp.y_test,
        epochs=100,
        batch_size=32,
        lr=0.001,
        weight_decay=1e-5,
        patience=10,
        verbose=False
    )

    end_time = time.time()
    training_time = end_time - start_time

    train_mape = model.mape(dp.X_train, dp.y_train, dp)
    test_mape = model.mape(dp.X_test, dp.y_test, dp)

    train_acc = 100 - train_mape
    test_acc = 100 - test_mape

    print(f"\nStock: {stock} ({symbol})")
    print(f"  Train windows : {train_windows} ({train_pct:.1f}%)")
    print(f"  Test windows  : {test_windows} ({test_pct:.1f}%)")
    print(f"  Total windows : {total_windows}")
    print(f"  Training time : {training_time:.2f}s")
    print(f"  Train MAPE    : {train_mape:.4f}%  (accuracy: {train_acc:.2f}%)")
    print(f"  Test  MAPE    : {test_mape:.4f}%  (accuracy: {test_acc:.2f}%)")

    # NEW: generate plot
    plot_predictions(model, dp, stock, symbol)

    results.append((stock, train_mape, test_mape))


print("\n===== Final Results =====")
for stock, train_mape, test_mape in results:
    print(f"{stock:10s} | Train MAPE: {train_mape:.4f}% | Test MAPE: {test_mape:.4f}%")