"""
this file will be to visualize the data coming out of the CSVs.

we will primarily visualize:

=========================================================
PARAMETER(S) V TEST/TRAIN ACCURACY
=========================================================

rbf kernel ->
  x-axis: C
  y-axis: gamma
  color: test/train accuracy

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.layout_engine as le
import matplotlib.colors as color
import numpy as np

import os

# setting up paths for data read
MNIST_DATA_PATH = os.path.join(os.getcwd(), "data", "P3", "mnist") # directory to write data to for collection
MNIST_FASHION_DATA_PATH = os.path.join(os.getcwd(), "data", "P3", "mnist-fashion") # directory to write data to for collection
TEST_AND_TRAIN_ACC = "test-train-acc.csv" # for the accuracy of final SVC on test, then train data

# accuracy collection of each primary dataset run
acc_mnist = pd.read_csv(os.path.join(MNIST_DATA_PATH, TEST_AND_TRAIN_ACC))
acc_mnist_fashion = pd.read_csv(os.path.join(MNIST_FASHION_DATA_PATH, TEST_AND_TRAIN_ACC))

# mnist

fig, ax = plt.subplots(4, 2, figsize=(10, 8)) # 4 rows for dim reducers, 2 cols for 1000, 5000 iterations 

num_iterations = [1000, 5000]
kernel = "rbf"
dim_reducers = ["pca_50", "pca_100", "pca_200", "lda"]

for row, dim_reducer in zip(range(0, len(dim_reducers)), dim_reducers): # for each reduction (row of visual)
  for col, num_it in zip(range(0, len(num_iterations)), num_iterations): # for each baseline iteration count (column of visual)

    # pull out the necessary info from the accuracy csv
    relevent_entries = acc_mnist.loc[
      (acc_mnist["dim reducer"] == dim_reducer) &
      (acc_mnist["kernel"] == kernel) &
      (acc_mnist["num iterations"] == num_it)  
      ]

    test_scores = np.array(relevent_entries.iloc[0:, 6])
    test_scores = test_scores.reshape(7, 7) # C values stay same per row

    C_vals = np.unique(relevent_entries.iloc[0:, 2]) # grab the values of C
    gamma_vals = np.unique(relevent_entries.iloc[0:, 3]) # grab the values of gamma

    # plot a heat map to highlight hotspots of best accuracy
    im = ax[row, col].imshow(
        test_scores,
        interpolation="nearest",
        cmap=plt.cm.plasma,
        aspect="auto"
    )

    # subplot metadata
    fontsize = 9
    ax[row, col].set_xlabel("Gamma", fontsize=fontsize)
    ax[row, col].set_ylabel("C", fontsize=fontsize)
    ax[row, col].set_xticks(np.arange(len(gamma_vals)), gamma_vals, rotation=45, fontsize=fontsize)
    ax[row, col].set_yticks(np.arange(len(C_vals)), C_vals, fontsize=fontsize)
    ax[row, col].set_title(f"{num_it} iterations, {dim_reducer} reduction", fontsize=fontsize)

fig.colorbar(im, ax=ax, label="Test Accuracy")
fig.set_layout_engine(le.ConstrainedLayoutEngine(w_pad=0.2, compress=True))
fig.suptitle("RBF Test Accuracy, MNIST")

# mnist fashion

fig, ax = plt.subplots(4, 2, figsize=(10, 8)) # 4 rows for dim reducers, 2 cols for 1000, 5000 iterations 

for row, dim_reducer in zip(range(0, len(dim_reducers)), dim_reducers): # for each reduction (row of visual)
  for col, num_it in zip(range(0, len(num_iterations)), num_iterations): # for each baseline iteration count (column of visual)

    # pull out the necessary info from the accuracy csv
    relevent_entries = acc_mnist_fashion.loc[
      (acc_mnist_fashion["dim reducer"] == dim_reducer) &
      (acc_mnist_fashion["kernel"] == kernel) &
      (acc_mnist_fashion["num iterations"] == num_it)  
      ]

    test_scores = np.array(relevent_entries.iloc[0:, 6])
    test_scores = test_scores.reshape(7, 7) # C values stay same per row
    C_vals = np.unique(relevent_entries.iloc[0:, 2]) # grab the values of C
    gamma_vals = np.unique(relevent_entries.iloc[0:, 3]) # grab the values of gamma

    # plot a heat map to highlight hotspots of best accuracy
    im = ax[row, col].imshow(
        test_scores,
        interpolation="nearest",
        cmap=plt.cm.plasma,
        aspect="auto"
    )

    # subplot metadata
    fontsize = 9
    ax[row, col].set_xlabel("Gamma", fontsize=fontsize)
    ax[row, col].set_ylabel("C", fontsize=fontsize)
    ax[row, col].set_xticks(np.arange(len(gamma_vals)), gamma_vals, rotation=45, fontsize=fontsize)
    ax[row, col].set_yticks(np.arange(len(C_vals)), C_vals, fontsize=fontsize)
    ax[row, col].set_title(f"{num_it} iterations, {dim_reducer} reduction", fontsize=fontsize)

fig.colorbar(im, ax=ax, label="Test Accuracy")
fig.set_layout_engine(le.ConstrainedLayoutEngine(w_pad=0.2, compress=True))
fig.suptitle("RBF Test Accuracy, MNIST Fashion")

# show figs
plt.show()
