"""
Modify the classes AdalineGD and LogisticRegressionGD 
in the textbook such that the bias
data field b is absorbed by the weight vector w . 
Your program is required to be compatible with the training
programs in the textbook.

REPORT:
Explain how the bias is transformed to an extra weight 
and why the translated model is equivalent
to the original one.
"""
# from log_ada_absorbed_bias import AdalineGD, LogisticRegressionGD
# from plotters import plot_decision_regions
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt


# ===================================================================
# SCRIPTING
# ===================================================================

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
print('From URL:', s)

df = pd.read_csv(s,
     header=None,
     encoding='utf-8')

# set up classes for setosa vs versi
y = df.iloc[0:100, 4].values # values in the 4th column of csv -> names of iris
y = np.where(y == "Iris-setosa", 0, 1) # setosa -> 0, versi 1

# extract the other information defining the classes
# specifically sepal and petal lengths
X = df.iloc[0:100, [0, 2]].values  

# ada = AdalineGD(eta=0.01, n_iter=10) # note that eta needs to be small here!
# ada.fit(X, y) # hand off the iris data and correct labels to learning algorithm
# # plotting of the linearly separable decision regions.
# plot_decision_regions(X, y, classifier=ada)

# log = LogisticRegressionGD(eta=0.01, n_iter=100)
# log.fit(X, y)
# plot_decision_regions(X, y, classifier=log)