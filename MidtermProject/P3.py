"""
Task 3: SVC Tuning and data transformation.

FOR BOTH MNIST and MNIST FASHION!

Make a machine learning pipeline using scikit-learn to integrate the following steps:

3.1: 
Standardize the (flattened) samples. Notice that the preprocess mapping is computed based
on the training data, but the same transformation should also be applied to the test data

3.2:
Dimensionality reduction on the data. You are required to use two ways: Principal Component
Analysis (PCA) and Linear Discriminant Analysis (LDA), to reduce the number of features of the data.
The original dimensionality is 784, you are required to consider the reduced dimensionalities: 50, 100 and
200 for PCA. Similar to the previous task, the reduction mappings should be derived from the training
set but should also be used to compress the test data.

3.3:
Build a Support Vector Classifier (SVC) with a kernel for classifying the compressed data.
You should use the scikit-learn SVC class. You are required to consider the three kernels along with their
hyperparameters:
• 'linear' - Linear kernel, the only hyperparameter is C.
• 'rbf' - Radial basis function kernel, the hyperparameters are C and gamma.
• 'poly' - Polynomial kernel, the hyperparameters are C, gamma and degree.
You are required to tune the hyperparameters and choose the best setting for each kernel. When tuning
the parameters, you need to measure the prediction error on both training set and test set.
"""

import os
import numpy as np
import idx2numpy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from MNISTReader import MNISTReader
from MNISTFashionReader import MNISTFashionReader
import time
import warnings #lol
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# 3.1 - Flatten the training and test data

mnist = MNISTReader()
fashion = MNISTFashionReader()

mnist_train_labels   = idx2numpy.convert_from_file(os.path.join("datasets", "mnist", "train-labels-idx1-ubyte"))
mnist_test_labels    = idx2numpy.convert_from_file(os.path.join("datasets", "mnist", "t10k-labels-idx1-ubyte"))
fashion_train_labels = idx2numpy.convert_from_file(os.path.join("datasets", "mnist-fashion", "train-labels-idx1-ubyte"))
fashion_test_labels  = idx2numpy.convert_from_file(os.path.join("datasets", "mnist-fashion", "t10k-labels-idx1-ubyte"))

mnist_scaler = StandardScaler()
mnist_train_scaled = mnist_scaler.fit_transform(mnist.train_images)
mnist_test_scaled  = mnist_scaler.transform(mnist.test_images)

fashion_scaler = StandardScaler()
fashion_train_scaled = fashion_scaler.fit_transform(fashion.train_images)
fashion_test_scaled  = fashion_scaler.transform(fashion.test_images)

# print(f"\nMNIST scaled train shape:          {mnist_train_scaled.shape}")
# print(f"MNIST scaled test shape:           {mnist_test_scaled.shape}")
# print(f"Fashion-MNIST scaled train shape:  {fashion_train_scaled.shape}")
# print(f"Fashion-MNIST scaled test shape:   {fashion_test_scaled.shape}")


# 3.2 PCA and LDA fir on training data then applied to both sets

# PCA
PCA_DIMS = [50, 100, 200]

mnist_pca = {}
fashion_pca = {}

for n in PCA_DIMS:
    # MNIST
    pca = PCA(n_components = n, random_state = 42)
    # Fit on training and then apply to test
    mnist_pca[n] = (pca.fit_transform(mnist_train_scaled), pca.transform(mnist_test_scaled),)
    # Fashion
    pca = PCA(n_components = n, random_state = 42)
    fashion_pca[n] = (pca.fit_transform(fashion_train_scaled), pca.transform(fashion_test_scaled),)

    # print(f"PCA n={n}  | MNIST train: {mnist_pca[n][0].shape}  Fashion train: {fashion_pca[n][0].shape}")

# LDA
# Used ravel here because i was getting some error about the dimension of train_labels
lda = LDA(n_components=9)
mnist_lda = (lda.fit_transform(mnist_train_scaled, mnist_train_labels),lda.transform(mnist_test_scaled),)

lda = LDA(n_components=9)
fashion_lda = (lda.fit_transform(fashion_train_scaled, fashion_train_labels),lda.transform(fashion_test_scaled),)

# print(f"LDA n=9   | MNIST train: {mnist_lda[0].shape}  Fashion train: {fashion_lda[0].shape}")

# 3.3 SVC

warnings.filterwarnings("ignore")

PARAM_GRIDS = {
    "linear": {"C": [0.01, 0.1, 1, 10]},
    "rbf":    {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.001]},
    "poly":   {"C": [0.1, 1, 10], "gamma": ["scale"], "degree": [2, 3, 4]},
}

def tune_and_eval(X_train, X_test, y_train, y_test, kernel):
    # Use a small subset for CV tuning
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=10_000, random_state=42)
    tune_idx, _ = next(splitter.split(X_train, y_train))
    X_tune, y_tune = X_train[tune_idx], y_train[tune_idx]

    gs = GridSearchCV(SVC(kernel=kernel), param_grid=PARAM_GRIDS[kernel],
                      cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tune, y_tune)
    best_params = gs.best_params_

    # Refit on full training set with best params
    svc = SVC(kernel=kernel, **best_params)
    t0 = time.time()
    svc.fit(X_train, y_train)
    train_time = time.time() - t0

    train_error = 1.0 - accuracy_score(y_train, svc.predict(X_train))
    test_error = 1.0 - accuracy_score(y_test, svc.predict(X_test))

    return best_params, train_error, test_error, train_time

# Experiment for both datasets
datasets = {
    "MNIST": {
        "pca": mnist_pca, "lda": mnist_lda,"y_train": mnist_train_labels, "y_test": mnist_test_labels,
    },
    "Fashion-MNIST": {
        "pca": fashion_pca, "lda": fashion_lda,"y_train": fashion_train_labels, "y_test": fashion_test_labels,
    },
}

best_params_store = {}

for dataset_name, data in datasets.items():
    print(f"\n{'='*65}\n  {dataset_name}\n{'='*65}")
    best_params_store[dataset_name] = {}

    # PCA experiments
    for n in PCA_DIMS:
        X_train_red, X_test_red = data["pca"][n]
        for kernel in ["linear", "rbf", "poly"]:
            print(f"  PCA-{n} | {kernel} ...")
            params, tr_err, te_err, t = tune_and_eval(
                X_train_red, X_test_red,
                data["y_train"], data["y_test"], kernel
            )
            print(f"    best params: {params}")
            print(f"    train error: {tr_err:.4f}  test error: {te_err:.4f}  time: {t:.1f}s")

            # Save best params from PCA-200 for P4
            if n == 200:
                best_params_store[dataset_name][kernel] = {"kernel": kernel, **params}

    # LDA experiment
    X_train_red, X_test_red = data["lda"]
    for kernel in ["linear", "rbf", "poly"]:
        print(f"  LDA-9  | {kernel} ...")
        params, tr_err, te_err, t = tune_and_eval(
            X_train_red, X_test_red,
            data["y_train"], data["y_test"], kernel
        )
        print(f"    best params: {params}")
        print(f"    train error: {tr_err:.4f}  test error: {te_err:.4f}  time: {t:.1f}s")