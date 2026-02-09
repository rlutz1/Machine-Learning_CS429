"""
Compare the performance of Adaline and logistic regression 
(bias absorbed versions) on the Iris
and Wine datasets that can be obtained from the UCI machine 
learning repository. You may use the Python
program given in our textbook (Page 117) to import the datasets.

• Iris dataset - You may consider the samples with the labels setosa, 
versicolor to form a training set for binary classification.

• Wine dataset - You may consider the samples with in the first 
two classes (1 and 2) to form a training set for binary classification.

The comparisons should be done based on the convergence of the loss. 

In order to make apple-to-apple comparisons, 
you should use the same hyperparameters 
and number of epochs for both learning algorithms.
"""
# from original_codes.adaline import AdalineGD
# from log_ada_absorbed_bias import AdalineGD, LogisticRegressionGD
# from plotters import plot_decision_regions
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt

# ===================================================================
# SCRIPTING
# ===================================================================

# grab the iris dataset
iris = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
# 'machine-learning-databases/wine/wine.data'
df = pd.read_csv(iris,
     header=None,
     encoding='utf-8')
# print(df)

# set up classes for setosa vs versi
y_iris = df.iloc[0:100, 4].values # values in the 4th column of csv -> names of iris
y_iris = np.where(y_iris == "Iris-setosa", 0, 1) # setosa -> 0, versi 1

# extract the other information defining the classes
X_iris = df.iloc[0:100, [0, 2]].values  

# grab the wine dataset
wine = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/wine/wine.data'
df = pd.read_csv(wine,
     header=None,
     encoding='utf-8')
# print(df)

y_wine = df.iloc[0:100, 0].values # values in the 4th column of csv -> type of grape
# print(y_wine)
y_wine = np.where(y_wine == 1, 0, 1) # classes 1 (0) and 2 of grapes (1)
# print(y_wine.size)
# extract the other information defining the classes
# alcohol(1), magnesium(5), color intensity(10), hue(11) 
# X_wine = df.iloc[0:100, [1, 5, 10, 11]].values # getting the vibe this is not linearly separable
# X_wine = df.iloc[0:100, [10, 11]].values # this one kinda works... with eta = 0.01, n = 10000
X_wine = df.iloc[0:100, [10, 11]].values # this one kinda works... with eta = 0.01, n = 10000
# print(X_wine)
# print(X_wine)


# ada = AdalineGD(eta=0.01, n_iter=1000) # note that eta needs to be small here!
# ada.fit(X_wine, y_wine) # hand off the iris data and correct labels to learning algorithm
# plotting of the linearly separable decision regions.
# plot_decision_regions(X_wine, y_wine, classifier=ada)

# log = LogisticRegressionGD(eta=0.01, n_iter=100)
# log.fit(X, y)
# plot_decision_regions(X, y, classifier=log)