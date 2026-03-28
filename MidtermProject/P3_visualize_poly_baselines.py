"""
this file will be to visualize the data coming out of the CSVs.

we will primarily visualize:

=========================================================
PARAMETER(S) V TEST/TRAIN ACCURACY
=========================================================

poly kernel ->
  x-axis: C
  y-axis: gamma
  z-axis: degree
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

num_iterations = [1000, 5000]
kernel = "poly"
dim_reducers = ["pca_50", "pca_100", "pca_200", "lda"]

for row, dim_reducer in zip(range(0, len(dim_reducers)), dim_reducers): # for each reduction (row of visual)

  # make a new window for each -- very hard to see in one window
  i = 1
  fig = plt.figure(figsize=(10, 8)) # 4 rows for dim reducers, 2 cols for 1000, 5000 iterations 


  for col, num_it in zip(range(0, len(num_iterations)), num_iterations): # for each baseline iteration count (column of visual)

    # pull out the necessary info from the accuracy csv
    relevent_entries = acc_mnist.loc[
      (acc_mnist["dim reducer"] == dim_reducer) &
      (acc_mnist["kernel"] == kernel) &
      (acc_mnist["num iterations"] == num_it)  
      ]

    test_scores = np.array(relevent_entries.iloc[0:, 6])
    # test_scores = test_scores.reshape(7, 7) # C values stay same per row
    C_vals = np.log10(np.array(relevent_entries.iloc[0:, 2])) # grab the values of C
    gamma_vals = np.log10(np.array(relevent_entries.iloc[0:, 3])) # grab the values of gamma
    degree_vals = np.array(relevent_entries.iloc[0:, 4]) # grab the values of degree
    
    # add a 3d plot
    ax = fig.add_subplot(1, 2, i, projection='3d') 
    i += 1
    im = ax.scatter(C_vals, gamma_vals, degree_vals, c=test_scores, cmap=plt.cm.plasma, s=50) # 's' is the marker size
    
    # subplot metadata
    fontsize = 9
    ax.set_xlabel("C", fontsize=fontsize)
    ax.set_ylabel("gamma", fontsize=fontsize)
    ax.set_zlabel("degree", fontsize=fontsize)
    ax.set_title(f"{num_it} iterations, {dim_reducer} reduction", fontsize=fontsize)
  
  fig.suptitle("Poly Test Accuracy, MNIST")
  cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) # Position the colorbar axis
  fig.colorbar(im, cax=cbar_ax, ax=fig.get_axes(), label="Test Accuracy")
  fig.subplots_adjust(hspace=0.5, right=0.8)


# mnist fashion

for row, dim_reducer in zip(range(0, len(dim_reducers)), dim_reducers): # for each reduction (row of visual)

  # make a new window for each -- very hard to see in one window
  i = 1
  fig = plt.figure(figsize=(10, 8)) # 4 rows for dim reducers, 2 cols for 1000, 5000 iterations 


  for col, num_it in zip(range(0, len(num_iterations)), num_iterations): # for each baseline iteration count (column of visual)

    # pull out the necessary info from the accuracy csv
    relevent_entries = acc_mnist_fashion.loc[
      (acc_mnist_fashion["dim reducer"] == dim_reducer) &
      (acc_mnist_fashion["kernel"] == kernel) &
      (acc_mnist_fashion["num iterations"] == num_it)  
      ]

    test_scores = np.array(relevent_entries.iloc[0:, 6])
    # test_scores = test_scores.reshape(7, 7) # C values stay same per row
    C_vals = np.log10(np.array(relevent_entries.iloc[0:, 2])) # grab the values of C
    gamma_vals = np.log10(np.array(relevent_entries.iloc[0:, 3])) # grab the values of gamma
    degree_vals = np.array(relevent_entries.iloc[0:, 4]) # grab the values of degree
    
    # add a 3d plot
    ax = fig.add_subplot(1, 2, i, projection='3d') 
    i += 1
    im = ax.scatter(C_vals, gamma_vals, degree_vals, c=test_scores, cmap=plt.cm.plasma, s=50) # 's' is the marker size
    
    # subplot metadata
    fontsize = 9
    ax.set_xlabel("C", fontsize=fontsize)
    ax.set_ylabel("gamma", fontsize=fontsize)
    ax.set_zlabel("degree", fontsize=fontsize)
    ax.set_title(f"{num_it} iterations, {dim_reducer} reduction", fontsize=fontsize)

  fig.suptitle("Poly Test Accuracy, MNIST Fashion")
  cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) # Position the colorbar axis
  fig.colorbar(im, cax=cbar_ax, ax=fig.get_axes(), label="Test Accuracy")
  fig.subplots_adjust(hspace=0.5, right=0.8)

# show figs
plt.show()
