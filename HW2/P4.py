import numpy as np
import time
import os
import warnings
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
from sklearn.exceptions import ConvergenceWarning

DATASETS = [
    ("d10n500",    10,   500),
    ("d50n500",    50,   500),
    ("d100n500",   100,  500),
    ("d10n5000",   10,   5000),
    ("d50n5000",   50,   5000),
    ("d100n5000",  100,  5000),
    ("d10n50000",  10,   50000),
    ("d50n50000",  50,   50000),
    ("d100n50000", 100,  50000),
]

MAX_ITER = 10000
C = 1.0

def read_csv(filepath):
    X, y = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(',')
            X.append([float(v) for v in vals[:-1]])
            y.append(int(vals[-1]))
    return np.array(X), np.array(y)

def train_and_evaluate(X_train, y_train, X_test, y_test, dual):
    loss_fn = 'hinge' if dual else 'squared_hinge'
    clf = LinearSVC(loss=loss_fn, dual=dual, C=C, max_iter=MAX_ITER, random_state=42)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    train_hl = hinge_loss(y_train, clf.decision_function(X_train), labels=[-1, 1])
    test_hl  = hinge_loss(y_test,  clf.decision_function(X_test),  labels=[-1, 1])
    return elapsed, train_hl, test_hl, int(clf.n_iter_)

def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, 'datasets')

    for name, d, n in DATASETS:
        train_path = os.path.join(dataset_dir, f"{name}_TRAIN.csv")
        test_path  = os.path.join(dataset_dir, f"{name}_TEST.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            continue

        X_train, y_train = read_csv(train_path)
        X_test,  y_test  = read_csv(test_path)

        p_time, p_tr, p_te, p_it = train_and_evaluate(X_train, y_train, X_test, y_test, dual=False)
        d_time, d_tr, d_te, d_it = train_and_evaluate(X_train, y_train, X_test, y_test, dual=True)

        print(f"{name}  d={d} n={n}")
        print(f"  Primal: time={p_time:.4f}s  train_loss={p_tr:.6f}  test_loss={p_te:.6f}  iters={p_it}")
        print(f"  Dual:   time={d_time:.4f}s  train_loss={d_tr:.6f}  test_loss={d_te:.6f}  iters={d_it}")

if __name__ == '__main__':
    main()