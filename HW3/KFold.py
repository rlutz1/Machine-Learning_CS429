import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = "HW3/data/csv_clean_data.csv"

BATCH_SIZE = 128
MAX_EPOCHS = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

MAX_FEATURES = 10000
MIN_DF = 5
MAX_DF = 0.8
NGRAM_RANGE = (1, 2)

K = 5
RANDOM_STATE = 42
PATIENCE = 3 # if after three batches of decreasing performance don't continue because it's not going to get getter. 


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(filepath):
    df = pd.read_csv(filepath)

    print("Columns found in CSV:", list(df.columns))

    possible_text_cols = ["review", "text"]
    possible_label_cols = ["sentiment", "label"]

    text_col = None
    label_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if text_col is None:
        raise ValueError("Could not find a text column. Expected one of: review, text")

    if label_col is None:
        raise ValueError("Could not find a label column. Expected one of: sentiment, label")

    texts = df[text_col].astype(str).values
    labels = df[label_col].values

    if isinstance(labels[0], str):
        labels = np.array([1 if label.lower() == "positive" else 0 for label in labels])

    labels = labels.astype(np.float32)
    return texts, labels


# -----------------------------
# FNN Model
# -----------------------------
class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# Evaluate model
# -----------------------------
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


# -----------------------------
# Train model with early stopping (patience)
# -----------------------------
def train_model(model, train_loader, val_loader, max_epochs, lr, weight_decay, device, patience):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    start_time = time.time()

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate_model(model, train_loader, device)
        val_acc = evaluate_model(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{max_epochs}, "
            f"Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    elapsed_time = time.time() - start_time

    return elapsed_time, best_epoch, best_val_acc


# -----------------------------
# Single train/test split
# -----------------------------
def run_single_split(texts, y, device):
    print("\n===== Single 70/30 Train-Test Split =====")

    texts_train, texts_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Fit TF-IDF only on the training split
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE
    )

    X_train = vectorizer.fit_transform(texts_train).toarray().astype(np.float32)
    X_test = vectorizer.transform(texts_test).toarray().astype(np.float32)

    print("Single split TF-IDF train shape:", X_train.shape)
    print("Single split TF-IDF test shape: ", X_test.shape)

    # Create a split from the training data for early stopping
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    X_subtrain_tensor = torch.tensor(X_subtrain, dtype=torch.float32)
    y_subtrain_tensor = torch.tensor(y_subtrain, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    subtrain_dataset = TensorDataset(X_subtrain_tensor, y_subtrain_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    subtrain_loader = DataLoader(subtrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FNN(input_dim=X_train.shape[1])

    train_time, best_epoch, best_val_acc = train_model(
        model,
        subtrain_loader,
        val_loader,
        MAX_EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        device,
        PATIENCE
    )

    train_acc = evaluate_model(model, train_loader, device)
    test_acc = evaluate_model(model, test_loader, device)

    print(f"Best validation accuracy during single split training: {best_val_acc:.4f}")
    print(f"Best epoch: {best_epoch}")

    return train_acc, test_acc, train_time


# -----------------------------
# K-Fold Cross Validation
# -----------------------------
def run_kfold_cv(texts, y, device, k=5):
    print(f"\n===== {k}-Fold Cross Validation =====")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    fold_train_accuracies = []
    fold_val_accuracies = []
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y), start=1):
        print(f"\n--- Fold {fold}/{k} ---")

        texts_train, texts_val = texts[train_idx], texts[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit TF-IDF on training fold only
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=NGRAM_RANGE
        )

        X_train = vectorizer.fit_transform(texts_train).toarray().astype(np.float32)
        X_val = vectorizer.transform(texts_val).toarray().astype(np.float32)

        print("Fold TF-IDF train shape:", X_train.shape)
        print("Fold TF-IDF val shape:  ", X_val.shape)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = FNN(input_dim=X_train.shape[1])

        fold_time, best_epoch, best_val_acc = train_model(
            model,
            train_loader,
            val_loader,
            MAX_EPOCHS,
            LEARNING_RATE,
            WEIGHT_DECAY,
            device,
            PATIENCE
        )

        train_acc = evaluate_model(model, train_loader, device)
        val_acc = evaluate_model(model, val_loader, device)

        fold_train_accuracies.append(train_acc)
        fold_val_accuracies.append(val_acc)
        fold_times.append(fold_time)

        print(f"Fold {fold} best epoch: {best_epoch}")
        print(f"Fold {fold} best validation accuracy seen: {best_val_acc:.4f}")
        print(f"Fold {fold} final train accuracy: {train_acc:.4f}")
        print(f"Fold {fold} final validation accuracy: {val_acc:.4f}")
        print(f"Fold {fold} time cost: {fold_time:.2f} seconds")

    avg_train_acc = np.mean(fold_train_accuracies)
    avg_val_acc = np.mean(fold_val_accuracies)
    avg_time = np.mean(fold_times)
    total_time = np.sum(fold_times)

    return avg_train_acc, avg_val_acc, avg_time, total_time


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    texts, labels = load_data(DATA_PATH)
    print(f"Number of reviews: {len(texts)}")

    # Show the approximate feature size
    print(
        f"TF-IDF settings -> max_features={MAX_FEATURES}, "
        f"min_df={MIN_DF}, max_df={MAX_DF}, ngram_range={NGRAM_RANGE}"
    )

    # Single split
    single_train_acc, single_test_acc, single_time = run_single_split(texts, labels, device)

    # K-fold
    kfold_train_acc, kfold_val_acc, kfold_avg_time, kfold_total_time = run_kfold_cv(
        texts, labels, device, k=K
    )

    print("\n===== Final Comparison =====")
    print(f"Single Split Training Accuracy: {single_train_acc:.4f}")
    print(f"Single Split Test Accuracy:     {single_test_acc:.4f}")
    print(f"Single Split Time Cost:         {single_time:.2f} seconds")
    print()
    print(f"{K}-Fold Average Training Accuracy:   {kfold_train_acc:.4f}")
    print(f"{K}-Fold Average Validation Accuracy: {kfold_val_acc:.4f}")
    print(f"{K}-Fold Average Time Cost:           {kfold_avg_time:.2f} seconds")
    print(f"{K}-Fold Total Time Cost:             {kfold_total_time:.2f} seconds")


if __name__ == "__main__":
    main()