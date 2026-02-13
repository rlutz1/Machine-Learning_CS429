"""
PROBLEM 4 DRIVER

Mini-batch SGD Comparison

Stochastic gradient descent (SGD) and mini-batch gradient descent are described in our textbook
(Page 45). Their combination is called mini-batch SGD which has been intensively used to handle large-scale
datasets in machine learning. Please implement a new function fit_mini_batch_SGD in the logistic regression
class for mini-batch SGD. Choose a small batch size, e.g., 32, and compare the performance of GD, SGD
and mini-batch SGD in the aspects of time cost and loss convergence speed.
"""

from helper_code.Absorbed_LogReg import LogisticRegressionGD
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Try to use interactive backend
import matplotlib.pyplot as plt
import time

"""
===================================================================
SCRIPTING
===================================================================
"""

def fit_sgd(self, X, y, shuffle=True):
    """
    Stochastic Gradient Descent - updates weights after EACH example.

    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
    self.w_[-1] = np.float64(0.)
    self.losses_ = []

    for i in range(self.n_iter):
        # Shuffle data
        if shuffle:
            indices = np.random.permutation(len(y))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        else:
            X_shuffled = X
            y_shuffled = y

        epoch_losses = []

        # Update weights after each individual example
        for xi, yi in zip(X_shuffled, y_shuffled):
            # Make prediction for this individual example
            net_input = self.net_input(xi.reshape(1, -1))
            output = self.activation(net_input)
            error = yi - output

            # Extend this example with 1 for bias
            xi_extended = self.extend_samples(xi.reshape(1, -1))

            # Update weights
            self.w_ += self.eta * 2.0 * xi_extended.ravel() * error

            # Calculate loss for this example
            loss = -yi * np.log(output) - (1 - yi) * np.log(1 - output)
            epoch_losses.append(loss[0])

        # Store average loss for this epoch
        self.losses_.append(np.mean(epoch_losses))

    return self


def fit_mini_batch_sgd(self, X, y, batch_size=32, shuffle=True):
    """
    Mini-batch SGD - updates weights after small BATCHES of examples.

    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
    self.w_[-1] = np.float64(0.)
    self.losses_ = []

    n_examples = X.shape[0]

    for epoch in range(self.n_iter):
        # Shuffle data
        if shuffle:
            indices = np.random.permutation(n_examples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        else:
            X_shuffled = X
            y_shuffled = y

        epoch_losses = []

        for start_idx in range(0, n_examples, batch_size):
            end_idx = min(start_idx + batch_size, n_examples)

            # Extract mini-batch
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            batch_actual_size = len(y_batch)

            # Make predictions for this batch
            net_input = self.net_input(X_batch)
            output = self.activation(net_input)
            errors = y_batch - output

            # Extend batch with 1's for bias
            X_batch_extended = self.extend_samples(X_batch)

            # Calculate gradient and update weights
            gradient = X_batch_extended.T.dot(errors) / batch_actual_size
            self.w_ += self.eta * 2.0 * gradient

            # Calculate loss for this batch
            batch_loss = (-y_batch.dot(np.log(output)) -
                          (1 - y_batch).dot(np.log(1 - output))) / batch_actual_size
            epoch_losses.append(batch_loss)

        # Store average loss for this epoch
        self.losses_.append(np.mean(epoch_losses))

    return self

LogisticRegressionGD.fit_sgd = fit_sgd
LogisticRegressionGD.fit_mini_batch_sgd = fit_mini_batch_sgd


"""
-------------------------------------------------------------------
LOAD WINE DATASET
-------------------------------------------------------------------
"""

def load_wine_dataset():
    """Load Wine dataset from UCI repository"""
    try:
        wine = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        df = pd.read_csv(wine, header=None, encoding='utf-8')
    except:
        np.random.seed(1)
        n_class1, n_class2, n_features = 59, 71, 13
        X_class1 = np.random.randn(n_class1, n_features) * 0.5 + np.array([
            13.7, 2.0, 2.4, 17, 106, 2.8, 2.9, 0.3, 1.9, 5.5, 1.0, 3.2, 1100
        ])
        X_class2 = np.random.randn(n_class2, n_features) * 0.5 + np.array([
            12.5, 2.5, 2.3, 20, 94, 2.3, 2.0, 0.4, 1.6, 3.0, 1.1, 2.8, 520
        ])
        X = np.vstack([X_class1, X_class2])
        y = np.array([0] * n_class1 + [1] * n_class2)
        shuffle_idx = np.random.permutation(len(y))
        return X[shuffle_idx], y[shuffle_idx]

    # Extract classes 1 and 2
    df = df[df[0].isin([1, 2])]
    y = df.iloc[:, 0].values
    y = np.where(y == 1, 0, 1)
    X = df.iloc[:, 1:].values

    return X, y


def standardize_features(X):
    # Standardize features to mean=0, std=1
    X_std = np.copy(X).astype(float)
    for i in range(X.shape[1]):
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X_std

"""
-------------------------------------------------------------------
RUN COMPARISON
-------------------------------------------------------------------
"""

def run_comparison():
    """Comparison between the 3"""

    X, y = load_wine_dataset()
    X_std = standardize_features(X)

    # Hyperparameters re the same for all methods
    eta = 0.01
    n_iter = 100
    batch_size = 32
    random_state = 1

    results = {}

    """
    -------------------------------------------------------------------
    1. FULL BATCH GRADIENT DESCENT
    -------------------------------------------------------------------
    """

    lr_gd = LogisticRegressionGD(eta=eta, n_iter=n_iter, random_state=random_state)
    start = time.time()
    lr_gd.fit(X_std, y)
    gd_time = time.time() - start

    results['Full Batch GD'] = {
        'model': lr_gd,
        'time': gd_time,
        'name': 'Full Batch GD',
        'updates_per_epoch': 1
    }

    """
    -------------------------------------------------------------------
    2. STOCHASTIC GRADIENT DESCENT
    -------------------------------------------------------------------
    """

    lr_sgd = LogisticRegressionGD(eta=eta, n_iter=n_iter, random_state=random_state)
    start = time.time()
    lr_sgd.fit_sgd(X_std, y, shuffle=True)
    sgd_time = time.time() - start

    results['SGD'] = {
        'model': lr_sgd,
        'time': sgd_time,
        'name': 'SGD',
        'updates_per_epoch': len(y)
    }

    """
    -------------------------------------------------------------------
    3. MINI-BATCH SGD
    -------------------------------------------------------------------
    """

    lr_mb = LogisticRegressionGD(eta=eta, n_iter=n_iter, random_state=random_state)
    start = time.time()
    lr_mb.fit_mini_batch_sgd(X_std, y, batch_size=batch_size, shuffle=True)
    mb_time = time.time() - start

    results['Mini-batch SGD'] = {
        'model': lr_mb,
        'time': mb_time,
        'name': f'Mini-batch SGD (batch={batch_size})',
        'updates_per_epoch': len(y) // batch_size
    }

    """
    -------------------------------------------------------------------
    PLOT RESULTS
    -------------------------------------------------------------------
    """

    plot_results(results)


def plot_results(results):
    """Create comparison plots"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'Full Batch GD': 'blue', 'SGD': 'orange', 'Mini-batch SGD': 'green'}

    # Plot 1: Loss Convergence
    ax = axes[0]
    for key, data in results.items():
        ax.plot(range(1, len(data['model'].losses_) + 1),
                data['model'].losses_,
                label=data['name'],
                color=colors[key],
                linewidth=2,
                alpha=0.7,
                marker='o',
                markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Loss', fontsize=12)
    ax.set_title('Loss Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Training Time
    ax = axes[1]
    names = [data['name'] for data in results.values()]
    times = [data['time'] for data in results.values()]
    bars = ax.bar(range(len(names)), times,
                  color=[colors[k] for k in results.keys()],
                  alpha=0.7,
                  edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{t:.4f}s', ha='center', va='bottom', fontsize=10)

    # Plot 3: Final Loss
    ax = axes[2]
    losses = [data['model'].losses_[-1] for data in results.values()]
    bars = ax.bar(range(len(names)), losses,
                  color=[colors[k] for k in results.keys()],
                  alpha=0.7,
                  edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{loss:.6f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    run_comparison()