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

from helper_code.Plotters import plot_2_params, plot_loss_ada_v_log
from helper_code.roxannes_abs_bias import AdalineGD, LogisticRegressionGD
import numpy as np
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
y_iris = np.where(y_iris == "Iris-setosa", 0, 1) # binary classification: setosa -> 0, versi 1

# extract the other information defining the classes
X_iris = df.iloc[0:100, [0, 2]].values  

# grab the wine dataset
wine = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/wine/wine.data'
df = pd.read_csv(wine,
     header=None,
     encoding='utf-8')

y_wine = df.iloc[0:100, 0].values # values in the 4th column of csv -> type of grape
y_wine = np.where(y_wine == 1, 0, 1) # classes 1 (0) and 2 (1) of grapes

# alcohol(1), magnesium(5), color intensity(10), hue(11) 
# X_wine = df.iloc[0:100, [1, 5, 10, 11]].values # getting the vibe this is not linearly separable or something
X_wine = df.iloc[0:100, [10, 11]].values # this one kinda works... with eta = 0.01, n = 10000

# -------------------------------------------------------------------
# IRIS:
# -------------------------------------------------------------------

ada = AdalineGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
ada.fit(X_iris, y_iris) # hand off the iris data and correct labels to learning algorithm

log = LogisticRegressionGD(eta=0.01, n_iter=10000) # same eta and iteration here
log.fit(X_iris, y_iris)

# two separate plots
plot_2_params(X_iris, y_iris, classifier=ada, title="Adaline with Iris", x_axis_title="Sepal Length", y_axis_title="Petal Length")
plot_2_params(X_iris, y_iris, classifier=log, title="Log Reg with Iris", x_axis_title="Sepal Length", y_axis_title="Petal Length")

# plot one on top of the other
plot_2_params(X_iris, y_iris, classifier=ada, show=False)
plot_2_params(X_iris, y_iris, classifier=log, show=False, title="Adaline and Log Reg with Iris", x_axis_title="Sepal Length", y_axis_title="Petal Length")
plt.show() # can see them on the same plot--easier to see difference a bit.

# plot loss 
plot_loss_ada_v_log(ada, log)

# -------------------------------------------------------------------
# WINE:
# -------------------------------------------------------------------

ada = AdalineGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
ada.fit(X_wine, y_wine) # hand off the iris data and correct labels to learning algorithm

log = LogisticRegressionGD(eta=0.01, n_iter=10000) # same eta and iteration here
log.fit(X_wine, y_wine)

# two separate plots
plot_2_params(X_wine, y_wine, classifier=ada, title="Adaline with Wine", x_axis_title="Color Intensity", y_axis_title="Hue")
plot_2_params(X_wine, y_wine, classifier=log, title="Log Reg with Wine", x_axis_title="Color Intensity", y_axis_title="Hue")

# plot one on top of the other
plot_2_params(X_wine, y_wine, classifier=ada, show=False)
plot_2_params(X_wine, y_wine, classifier=log, show=False, title="Adaline and Log Reg with Wine", x_axis_title="Color Intensity", y_axis_title="Hue")
plt.show() # can see them on the same plot--easier to see difference a bit.

# plot loss 
plot_loss_ada_v_log(ada, log)


