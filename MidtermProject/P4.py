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
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        subsets = np.array_split(indices, self.n_models)

        self.models = []

        for subset_idx in subsets:
            X_sub = X[subset_idx]
            y_sub = y[subset_idx]

            model = SVC(kernel=self.kernel, **self.svc_params)
            model.fit(X_sub, y_sub)
            self.models.append(model)

    def predict(self, X):
        all_preds = np.array([model.predict(X) for model in self.models])

        final_preds = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            vote_counts = Counter(votes)
            final_preds.append(vote_counts.most_common(1)[0][0])

        return np.array(final_preds)


def normalize_data(X):
    return X.astype(np.float32) / 255.0


def evaluate_ensemble(dataset_name, X_train, y_train, X_test, y_test, best_params):
    print(f"\n{'=' * 60}")
    print(f"{dataset_name}")
    print(f"{'=' * 60}")

    for kernel_name, params in best_params.items():
        print(f"\nKernel: {kernel_name}")
        print(f"Parameters: {params}")

        ensemble = BaggingSVC(
            n_models=8,
            kernel=kernel_name,
            **params
        )

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Bagging Test Accuracy: {acc:.4f}")


def main():
    np.random.seed(42)

    mnist = MNISTReader()
    X_train_mnist = mnist.train_images
    y_train_mnist = mnist.train_labels
    X_test_mnist = mnist.test_images
    y_test_mnist = mnist.test_labels

    X_train_mnist = normalize_data(X_train_mnist)
    X_test_mnist = normalize_data(X_test_mnist)

    fashion = MNISTFashionReader()
    X_train_fashion = fashion.train_images
    y_train_fashion = fashion.train_labels
    X_test_fashion = fashion.test_images
    y_test_fashion = fashion.test_labels

    X_train_fashion = normalize_data(X_train_fashion)
    X_test_fashion = normalize_data(X_test_fashion)

    best_params_mnist = {
        "linear": {
            "C": 0.01
        },
        "rbf": {
            "C": 10,
            "gamma": 0.001
        },
        "poly": {
            "C": 100,
            "gamma": 0.001,
            "degree": 3
        }
    }

    best_params_fashion = {
        "linear": {
            "C": 0.01
        },
        "rbf": {
            "C": 10,
            "gamma": 0.001
        },
        "poly": {
            "C": 100,
            "gamma": 0.001,
            "degree": 3
        }
    }

    evaluate_ensemble(
        "MNIST",
        X_train_mnist, y_train_mnist,
        X_test_mnist, y_test_mnist,
        best_params_mnist
    )

    evaluate_ensemble(
        "Fashion-MNIST",
        X_train_fashion, y_train_fashion,
        X_test_fashion, y_test_fashion,
        best_params_fashion
    )


if __name__ == "__main__":
    main()