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

REPORT: 
For Task 2, explain the comparisons using figures
"""

from helper_code.Plotters import plot_2_params, plot_loss_ada_v_log
from helper_code.roxannes_abs_bias import AdalineGD, LogisticRegressionGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
===================================================================
SCRIPTING
===================================================================
"""

# grab the iris dataset
iris = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
# 'machine-learning-databases/wine/wine.data'
df = pd.read_csv(iris,
     header=None,
     encoding='utf-8')
# print(df)

# set up classes for setosa vs versi
y_iris = df.iloc[0:100, 4].values # values in the 5th column of csv -> names of iris
y_iris = np.where(y_iris == "Iris-setosa", 0, 1) # binary classification: setosa -> 0, versi 1

# extract the other information defining the classes (sepal and petal length)
X_iris = df.iloc[0:100, [0, 2]].values  

# grab the wine dataset
wine = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/wine/wine.data'
df = pd.read_csv(wine,
     header=None,
     encoding='utf-8')

y_wine = df.iloc[0:100, 0].values # values in the 1st column of csv -> type of grape
y_wine = np.where(y_wine == 1, 0, 1) # classes 1 (0) and 2 (1) of grapes

# extract the other information defining the classes (hue and color intensity)
# X_wine = df.iloc[0:100, [1, 2, 10, 11]].values 
X_wine = df.iloc[0:100, [10, 11]].values 
"""
-------------------------------------------------------------------
IRIS:
-------------------------------------------------------------------
"""

# use these to ensure models are running with the same parameters
e = 0.01 # learning rate for iris
i = 10000 # num iterations for iris

ada = AdalineGD(eta=e, n_iter=i) # note that eta needs to be small here!
ada.fit(X_iris, y_iris) # hand off the iris data and correct labels to learning algorithm

log = LogisticRegressionGD(eta=e, n_iter=i) # same eta and iteration here
log.fit(X_iris, y_iris)

# if X_iris.shape[1] == 2: # 2 params needed for that func, breaks with more
#     titles = [f"Adaline with Iris (learn rate={e}, epochs={i})", f"Log Reg with Iris (learn rate={e}, epochs={i})"]
#     x_axis_titles = ["Sepal Length"] * len(titles)
#     y_axis_titles = ["Petal Length"] * len(titles)
#     classifiers = [ada, log]
#     # plot 2 figures of the classifiers decision regions
#     plot_2_params(
#     X_iris, 
#     y_iris, 
#     classifiers=classifiers, 
#     titles=titles, 
#     x_axis_titles=x_axis_titles, 
#     y_axis_titles=y_axis_titles, 
#     num_plots=len(titles)
#     )
# # plot the loss comparison
# plot_loss_ada_v_log(ada, log)

"""
-------------------------------------------------------------------
WINE:
-------------------------------------------------------------------
"""

# use these to ensure models are running with the same parameters
e = 0.0001 # learning rate for wine
i = 10000 # num iterations for wine

ada = AdalineGD(eta=e, n_iter=i) # note that eta needs to be small here!
ada.fit(X_wine, y_wine) # hand off the iris data and correct labels to learning algorithm

log = LogisticRegressionGD(eta=e, n_iter=i) # same eta and iteration here
log.fit(X_wine, y_wine)

titles = [f"Adaline with Wine (learn rate={e}, epochs={i})", f"Log Reg with Wine (learn rate={e}, epochs={i})"]
x_axis_titles = ["Color Intensity"] * len(titles)
y_axis_titles = ["Hue"] * len(titles)
classifiers = [ada, log]

if X_wine.shape[1] == 2: # 2 params needed for that func, breaks with more
    # plot 2 figures of the classifiers decision regions
    plot_2_params(
    X_wine, 
    y_wine, 
    classifiers=classifiers, 
    titles=titles, 
    x_axis_titles=x_axis_titles, 
    y_axis_titles=y_axis_titles, 
    num_plots=len(titles)
    )

# plot the loss comparison
plot_loss_ada_v_log(ada, log)


