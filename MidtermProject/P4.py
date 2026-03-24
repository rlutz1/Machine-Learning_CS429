"""
Task 4:

FOR BOTH MNIST and MNIST FASHION!

Write a program that trains a finite set of SVCs using bootstrap aggregating. 
Please use at least 8 models. The training dataset should be divided into multiple 
disjoint sets and each individual model is trained based on a subset. 
The final prediction for an image is obtained via voting. Please use the three kernels
along with their best hyperparameters you found in the previous task. 
Please do not use sklearn.ensemble.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

from MNISTReader import MNISTReader
from MNISTFashionReader import MNISTFashionReader


class BaggingSVC:
    def __init__(self, n_models=8, kernel="rbf", **svc_params):
        self.n_models = n_models
        self.kernel = kernel
        self.svc_params = svc_params
        self.models = []

    def fit(self, X, y):
        """
        Split the training data into n_models disjoint subsets,
        and train one SVC on each subset.
        """
        n_samples = X.shape[0]

        # Shuffle indices so the subsets are random
        indices = np.random.permutation(n_samples)

        # Split into disjoint stuff
        subsets = np.array_split(indices, self.n_models)

        self.models = []

        for subset_idx in subsets:
            X_sub = X[subset_idx]
            y_sub = y[subset_idx]

            model = SVC(kernel=self.kernel, **self.svc_params)
            model.fit(X_sub, y_sub)
            self.models.append(model)

    def predict(self, X):
        """
        Predict using majority voting across all models.
        """
        # (n_models, n_samples)
        all_preds = np.array([model.predict(X) for model in self.models])

        final_preds = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            vote_count = Counter(votes)
            final_preds.append(vote_count.most_common(1)[0][0])

        return np.array(final_preds)


def normalize_data(X):
    return X.astype(np.float32) / 255.0


def run_experiment(dataset_name, X_train, y_train, X_test, y_test, best_params):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    for kernel_name, params in best_params.items():
        print(f"\nTraining bagged SVC with kernel = {kernel_name}")

        ensemble = BaggingSVC(
            n_models=8,      # at least 8 models
            kernel=kernel_name,
            **params
        )

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy ({kernel_name}): {acc:.4f}")


def main():
    np.random.seed(42)

    # -------------------------
    # Load MNIST
    # -------------------------
    mnist = MNISTReader()
    X_train_mnist, y_train_mnist = mnist.load_train_data()
    X_test_mnist, y_test_mnist = mnist.load_test_data()

    X_train_mnist = normalize_data(X_train_mnist)
    X_test_mnist = normalize_data(X_test_mnist)

    # -------------------------
    # Load Fashion-MNIST
    # -------------------------
    fashion = MNISTFashionReader()
    X_train_fashion, y_train_fashion = fashion.load_train_data()
    X_test_fashion, y_test_fashion = fashion.load_test_data()

    X_train_fashion = normalize_data(X_train_fashion)
    X_test_fashion = normalize_data(X_test_fashion)

    # ---------------------------------------------------------
    # REPLACE THESE WITH THE BEST STUFF FROM QUESTION 3
    # ---------------------------------------------------------
    best_params_mnist = {
        "linear": {
            "C": 1.0
        },
        "poly": {
            "C": 1.0,
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0
        },
        "rbf": {
            "C": 5.0,
            "gamma": "scale"
        }
    }

    best_params_fashion = {
        "linear": {
            "C": 1.0
        },
        "poly": {
            "C": 1.0,
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0
        },
        "rbf": {
            "C": 5.0,
            "gamma": "scale"
        }
    }

    # -------------------------
    # Run MNIST
    # -------------------------
    run_experiment(
        "MNIST",
        X_train_mnist, y_train_mnist,
        X_test_mnist, y_test_mnist,
        best_params_mnist
    )

    # -------------------------
    # Run Fashion-MNIST
    # -------------------------
    run_experiment(
        "Fashion-MNIST",
        X_train_fashion, y_train_fashion,
        X_test_fashion, y_test_fashion,
        best_params_fashion
    )


if __name__ == "__main__":
    main()