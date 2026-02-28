import numpy as np
import time
import os
import matplotlib.pyplot as plt
from P1 import LinearSVC

DATASET_DIR = "datasets"
DIMS        = [10, 50, 100]
SIZES       = [500, 5000, 50000]
EPOCHS      = 50
ETA         = 0.0001
C           = 1.0

def read_csv(filepath):
    X, y = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            X.append([float(v) for v in parts[:-1]])
            y.append(int(parts[-1]))
    return np.array(X), np.array(y)

results = {}

for d in DIMS:
    for n in SIZES:
        train_path = os.path.join(DATASET_DIR, f"d{d}n{n}_TRAIN.csv")
        test_path  = os.path.join(DATASET_DIR, f"d{d}n{n}_TEST.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            results[(d, n)] = None
            continue

        X_train, y_train = read_csv(train_path)
        X_test,  y_test  = read_csv(test_path)

        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std

        svc = LinearSVC(eta=ETA, epochs=EPOCHS, C=C, random_state=42)
        t0  = time.time()
        svc.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred   = svc.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100

        results[(d, n)] = {
            "train_time_s":  elapsed,
            "loss_epoch1":   svc.losses_[0],
            "loss_final":    svc.losses_[-1],
            "loss_drop_pct": (svc.losses_[0] - svc.losses_[-1]) / (svc.losses_[0] + 1e-10) * 100,
            "test_acc_pct":  accuracy,
            "losses":        svc.losses_,
        }

# Loss convergence plot (3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("LinearSVC — Loss Convergence Across Datasets", fontsize=14)
for i, d in enumerate(DIMS):
    for j, n in enumerate(SIZES):
        ax = axes[i][j]
        ax.set_title(f"d={d}, n={n}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hinge Loss")
        r = results.get((d, n))
        if r:
            ax.plot(range(1, EPOCHS + 1), r["losses"], color='steelblue', linewidth=1.5)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
plt.tight_layout()
plt.savefig("P3-convergence.png", dpi=150)
plt.show()