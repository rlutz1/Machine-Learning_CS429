"""
Investigate the scalability of the LinearSVC class you have implemented. Using the dataset
generator developed in the previous task, you may produce random datasets regarding to the 9 combinations
of the following scales: d = 10, 50, 100 and n = 500, 5000, 50000. You may assign a large constant such
as 100 to u. (Please feel free to slightly adjust the scales according to your computer's hardware.) Evaluate
the time cost and loss convergence of your linear SVC on the 9 datasets. The comparison should be given
by tables along with explanations
"""

import numpy as np
import time
import os
from P1 import LinearSVC

Dataset_Directory = "datasets"

# I know he talked about converting to CSV but did he say anything about reading?
def read_data_sets(filepath):
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

# I wouldn't change these values it takes a while to run on my computer
DIMS = [10, 50, 100]
SIZES = [500, 5000, 50000]
EPOCHS = 50
ETA = 0.0001  # Smaller learning rate because of large num of features
C = 1.0 # Keeping the model simple. Can increase to better fit

results = {}

print("=" * 70)
print(f"LinearSVC Scalability Study  |  epochs={EPOCHS}  eta={ETA}  C={C}")
print("=" * 70)

for d in DIMS:
    for n in SIZES:
        key = (d, n)
        train_path = os.path.join(Dataset_Directory, f"d{d}n{n}_TRAIN.csv")
        test_path = os.path.join(Dataset_Directory, f"d{d}n{n}_TEST.csv")

        # Ask me about how many times i failed before i added this lol
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"\n[SKIP] d={d}, n={n}  — dataset files not found.")
            results[key] = None
            continue

        X_train, y_train = read_data_sets(train_path)
        X_test, y_test = read_data_sets(test_path)

        # Standardize features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8 # Small num to avoid division by 0 if no variance
        X_train_std = (X_train - mean) / std
        X_test_std = (X_test - mean) / std

        svc = LinearSVC(eta=ETA, epochs=EPOCHS, C=C, random_state=42)

        t0 = time.time()
        svc.fit(X_train_std, y_train)
        elapsed = time.time() - t0

        # Accuracy on test set
        y_pred = svc.predict(X_test_std)
        accuracy = np.mean(y_pred == y_test) * 100

        first_loss = svc.losses_[0]
        final_loss = svc.losses_[-1]
        # Convergence = relative drop in loss over training
        loss_drop_pct = (first_loss - final_loss) / (first_loss + 1e-10) * 100

        results[key] = {
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "train_time_s": elapsed,
            "loss_epoch1": first_loss,
            "loss_final": final_loss,
            "loss_drop_pct": loss_drop_pct,
            "test_acc_pct": accuracy,
            "losses": svc.losses_,
        }

        print(f"\nd={d:3d}, n={n:6d} | "
              f"train_time={elapsed:6.2f}s | "
              f"loss: {first_loss:.1f} → {final_loss:.1f} ({loss_drop_pct:.1f}% drop) | "
              f"test_acc={accuracy:.1f}%")

 # Formatter helper
def fmt(val, spec):
    return f"{val:{spec}}" if val is not None else "  N/A  "


print("\n")
print("=" * 70)
print("TABLE 1 — Training Time (seconds)")
print("=" * 70)
header = f"{'d \\ n':>8s} | {'n=500':>10s} | {'n=5000':>10s} | {'n=50000':>10s}"
print(header)
print("-" * len(header))
for d in DIMS:
    row = f"{'d=' + str(d):>8s} |"
    for n in SIZES:
        r = results.get((d, n))
        val = fmt(r["train_time_s"], "10.2f") if r else "  N/A    "
        row += f" {val} |"
    print(row)

print("\n")
print("=" * 70)
print("TABLE 2 — Final Epoch Hinge Loss")
print("=" * 70)
print(header)
print("-" * len(header))
for d in DIMS:
    row = f"{'d=' + str(d):>8s} |"
    for n in SIZES:
        r = results.get((d, n))
        val = fmt(r["loss_final"], "10.1f") if r else "  N/A    "
        row += f" {val} |"
    print(row)

print("\n")
print("=" * 70)
print("TABLE 3 — Loss Reduction over Training  (epoch 1 → epoch 50, %)")
print("=" * 70)
print(header)
print("-" * len(header))
for d in DIMS:
    row = f"{'d=' + str(d):>8s} |"
    for n in SIZES:
        r = results.get((d, n))
        val = (f"{r['loss_drop_pct']:9.1f}%" if r else "  N/A    ")
        row += f" {val} |"
    print(row)

print("\n")
print("=" * 70)
print("TABLE 4 — Test Set Accuracy (%)")
print("=" * 70)
print(header)
print("-" * len(header))
for d in DIMS:
    row = f"{'d=' + str(d):>8s} |"
    for n in SIZES:
        r = results.get((d, n))
        val = (f"{r['test_acc_pct']:9.1f}%" if r else "  N/A    ")
        row += f" {val} |"
    print(row)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("LinearSVC — Loss Convergence Across Datasets", fontsize=14)

for i, d in enumerate(DIMS):
    for j, n in enumerate(SIZES):
        ax = axes[i][j]
        r = results.get((d, n))
        ax.set_title(f"d={d}, n={n}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hinge Loss")
        if r:
            ax.plot(range(1, EPOCHS + 1), r["losses"], color='steelblue', linewidth=1.5)
            ax.annotate(f"final={r['loss_final']:.0f}", xy=(EPOCHS, r['loss_final']),
                        fontsize=7, ha='right')
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

plt.tight_layout()
plt.show()

print("\n")
print("=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)
print("""
1. TRAINING TIME (Table 1):
   Training time scales roughly linearly with the number of samples n,
   since SGD performs one weight update per sample per epoch.The n=50000 
   datasets are ~100x slower to train than n=500, as expected.

2. FINAL HINGE LOSS (Table 2):
   Larger n datasets produce higher absolute hinge loss because more samples
   contribute to the total. Normalizing by n would give per-sample loss, which
   is more comparable across dataset sizes. Higher d does not strongly increase
   loss since the data is linearly separable since that's how we made the generator.

3. LOSS CONVERGENCE (Table 3):
   Loss drops substantially in all cases. Larger datasets tend to 
   show higher relative drops because the initial loss is larger and there is 
   more signal to learn from. Smaller datasets converge quickly but may show 
   some oscillation.

4. TEST ACCURACY (Table 4):
   Accuracy is high across all configurations because the datasets are linearly
   separable by generation. This probably could be tuned further with adjustments to
   learning rate or other params. The LinearSVC implementation generalizes well (or at 
   least similarly) regardless of scale.
""")
