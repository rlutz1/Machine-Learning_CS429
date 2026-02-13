"""
PROBLEM 3 DRIVER

Adaline and perceptron learning can only be used for binary classification, however, the Iris dataset
has 3 classes: setosa, versicolor and virginica. If you are only allowed to use perceptrons but the number is
not limited, how would you like to perform a multiclass classification for the whole Iris data set? Please write
a program for this task
"""

from helper_code.Absorbed_Adaline import AdalineGD
from helper_code.Absorbed_LogReg import LogisticRegressionGD
from helper_code.unaltered_original_code.Perceptron import Perceptron
from helper_code.Plotters import plot_2_params
import numpy as np
import pandas as pd

"""
===================================================================
SCRIPTING
===================================================================
"""
# for clarity when plotting only. 
# corresponds the class to its class number.
SETOSA_CLASS = 0
VERSI_CLASS = 1
VIRGINIA_CLASS = 2


# grab the iris dataset
iris = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'

df = pd.read_csv(iris,
     header=None,
     encoding='utf-8')

# extract the other information defining the classes
X = df.iloc[0:, [0, 2]].values  # sepal, petal length
# X = df.iloc[0:, [2, 3]].values # petal length, width
y = df.iloc[0:, 4].values

# replace string names with numbers for plotting.
# setosa is class 0, versi class 1, virg class 2
conditions = [y == "Iris-setosa", y == "Iris-versicolor", y == "Iris-virginica"]
choices = [SETOSA_CLASS, VERSI_CLASS, VIRGINIA_CLASS]
y = np.select(conditions, choices)

# set up classes for setosa, versi, virg
y_setosa = df.iloc[0:, 4].values # values in the 4th column of csv -> names of iris
y_setosa = np.where(y_setosa == "Iris-setosa", 1, 0) # setosa -> 1, everything else 0 

y_versi = df.iloc[0:, 4].values # values in the 4th column of csv -> names of iris
y_versi = np.where(y_versi == "Iris-versicolor", 1, 0) # versicolor -> 1, everything else 0 

y_virg = df.iloc[0:, 4].values # values in the 4th column of csv -> names of iris
y_virg = np.where(y_virg == "Iris-virginica", 1, 0) # virginica -> 1, everything else 0 

# the following code is optional, but was used for a figure in report.
# we're gonna shift versi sepal lengths (0) over enough
# to enforce linear separability, to prove a point.
# this assumes you grab sepal and petal length ([0, 2]) above
# i = 0
# for f in y_versi:
#   if f == 1: # if class 1
#     X[i][0] += 5 # shove over sepal length 5 units
#   i += 1

# use these to ensure models are running with the same parameters
e = 0.01 # learning rate for iris
i = 10000 # num iterations for iris

"""
-------------------------------------------------------------------
TRAIN BASIC 0/1 PERCEPTRON:
-------------------------------------------------------------------
"""

P_setosa = Perceptron(eta=e, n_iter=i) 
P_setosa.fit(X, y_setosa) # hand off the iris data and correct labels to learning algorithm

P_versi = Perceptron(eta=e, n_iter=i) 
P_versi.fit(X, y_versi) # hand off the iris data and correct labels to learning algorithm

P_virg = Perceptron(eta=e, n_iter=i) 
P_virg.fit(X, y_virg) # hand off the iris data and correct labels to learning algorithm

"""
-------------------------------------------------------------------
TRAIN ADALINE:
-------------------------------------------------------------------
"""

A_setosa = AdalineGD(eta=e, n_iter=i) 
A_setosa.fit(X, y_setosa) # hand off the iris data and correct labels to learning algorithm

A_versi = AdalineGD(eta=e, n_iter=i) 
A_versi.fit(X, y_versi) # hand off the iris data and correct labels to learning algorithm

A_virg = AdalineGD(eta=e, n_iter=i) 
A_virg.fit(X, y_virg) # hand off the iris data and correct labels to learning algorithm

"""
-------------------------------------------------------------------
TRAIN LOGISTIC REGRESSION:
-------------------------------------------------------------------
"""

LR_setosa = LogisticRegressionGD(eta=e, n_iter=i) 
LR_setosa.fit(X, y_setosa) # hand off the iris data and correct labels to learning algorithm

LR_versi = LogisticRegressionGD(eta=e, n_iter=i) 
LR_versi.fit(X, y_versi) # hand off the iris data and correct labels to learning algorithm

LR_virg = LogisticRegressionGD(eta=e, n_iter=i) 
LR_virg.fit(X, y_virg) # hand off the iris data and correct labels to learning algorithm

"""
we will define a really basic 3 class predictor.
this takes the 3 classifiers on initialization.
the predict method is then simply to evaluate
the test samples given through all 3 classifiers.
whichever yields the maximum net_input value
will be chosen to represent the class that the sample 
must be.
"""

class TriClassPredictor():
  
  def __init__(self, P0, P1, P2):
    self.P0 = P0
    self.P1 = P1
    self.P2 = P2

  """
  for the iris example,
  P0 -> setosa classifier
  P1 -> versicolor classifier
  P2 -> virginia classifier
  whichever P# yields the larger net_input value
  corresponds with which iris it is more likely to be.
  ie, P0 yields max evaluation -> predict to be setosa. 
  """
  def predict(self, X):
    val0 = self.P0.net_input(X) # predictions of setosa
    val1 = self.P1.net_input(X) # predictions of versi
    val2 = self.P2.net_input(X) # predictions of virginia
    y = np.array([])
    for i in range(X.shape[0]): # for each of the test cases in X
      maximum = max(val0[i], val1[i], val2[i]) # get max of 3 for this sample

      # say the maximum is the class we will go with.
      if   maximum == val0[i]: y = np.append(y, SETOSA_CLASS) 
      elif maximum == val1[i]: y = np.append(y, VERSI_CLASS)
      elif maximum == val2[i]: y = np.append(y, VIRGINIA_CLASS)
      else: print("Something's off in the TriClassPerceptron ", maximum)

    return y


# simple accuracy rate calculator 
def acc_rate(pred_labels, true_labels):
  accurates = 0
  for (pred_label, true_label) in zip(pred_labels, true_labels):
    if true_label == pred_label:
      accurates += 1

  return accurates / true_labels.shape[0]

# accuracy values.
classifier_perc = TriClassPredictor(P_setosa, P_versi, P_virg)
print(f"accuracy rating of perceptron: {acc_rate(classifier_perc.predict(X), y)}" )

# accuracy values.
classifier_ada = TriClassPredictor(A_setosa, A_versi, A_virg)
print(f"accuracy rating of adaline: {acc_rate(classifier_ada.predict(X), y)}" )

# accuracy values.
classifier_log = TriClassPredictor(LR_setosa, LR_versi, LR_virg)
print(f"accuracy rating of logistic regression: {acc_rate(classifier_log.predict(X), y)}" )


# finally, generate a plot with visible decision regions
# using the tri class predictor

titles = ["Perceptron with 3 classes", "Adaline with 3 Classes", "Log Reg with with 3 Classes"]
x_axis_titles = ["Sepal Length"] * len(titles) # for when plotting [0, 2]
y_axis_titles = ["Petal Length"] * len(titles)
# x_axis_titles = ["Petal Length"] * len(titles) # for when plotting [2, 3]
# y_axis_titles = ["Petal Width"] * len(titles)
classifiers = [
  classifier_perc,
  classifier_ada,
  classifier_log
  ]

# plot 3 figures of the classifiers decision regions
plot_2_params(
  X, 
  y, 
  classifiers=classifiers, 
  titles=titles, 
  x_axis_titles=x_axis_titles, 
  y_axis_titles=y_axis_titles, 
  num_plots=len(titles)
  )