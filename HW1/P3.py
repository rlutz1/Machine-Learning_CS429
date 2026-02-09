"""
Adaline and perceptron learning can only be used for binary classification, however, the Iris dataset
has 3 classes: setosa, versicolor and virginica. If you are only allowed to use perceptrons but the number is
not limited, how would you like to perform a multiclass classification for the whole Iris data set? Please write
a program for this task
"""
from helper_code.plotters import plot_3_classes
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt

"""
textbook code for the vanilla perceptron.
"""
class Perceptron:

  """
  eta          -> the learning rate, best being small. being too big can cause 
                  instability
  n_iter       -> this many passes over the training set. why not tolerance?
  random_state -> randomizer seed such that you can get the same random
                  sequence each time.
  """
  def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  """
  this is the actual learning algorithm.
  """
  def fit(self, X, y):
    rgen = np.random.RandomState(self.random_state) # rand num generator, LEGACY
    self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1]) 
    self.b_ = np.float64(0.) # just 0 as a float value (float_ no longer works with current python)
    self.errors_ = [] # empty error count tracker
    for _ in range(self.n_iter): # iterate this many times
      errors = 0
      for xi, target in zip(X, y): 
        # update for the weight: w = w + eta * (predict - correct label) * xi
        update = self.eta * (target - self.predict(xi))
        self.w_ += update * xi
        # update for b: b = b + eta * (predict - correct label) 
        self.b_ += update
        errors += int(update != 0.0) # add to errors if error made, update is 0 means no misclass.
      self.errors_.append(errors) # how many misclassifications this epoch
    return self

  # use: [X (dot) w] + b
  def net_input(self, X):
    return np.dot(X, self.w_) + self.b_

  # the general prediction entry point
  def predict(self, X):
    return np.where(self.net_input(X) >= 0.0, 1, 0)


# ===================================================================
# SCRIPTING
# ===================================================================

# grab the iris dataset
iris = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
df = pd.read_csv(iris,
     header=None,
     encoding='utf-8')
# print(df)

# extract the other information defining the classes
X = df.iloc[0:150, [0, 2]].values  
y = df.iloc[0:150, 4].values
y[(y == "Iris-setosa")] = 0
y[(y == "Iris-versicolor")] = 1
y[(y == "Iris-virginica")] = 2



# set up classes for setosa, versi, virg
y_setosa = df.iloc[0:150, 4].values # values in the 4th column of csv -> names of iris
y_setosa = np.where(y_setosa == "Iris-setosa", 1, 0) # setosa -> 1, everything else 0 

y_versi = df.iloc[0:150, 4].values # values in the 4th column of csv -> names of iris
y_versi = np.where(y_versi == "Iris-versicolor", 1, 0) # versicolor -> 1, everything else 0 

y_virg = df.iloc[0:150, 4].values # values in the 4th column of csv -> names of iris
y_virg = np.where(y_virg == "Iris-virginica", 1, 0) # virginica -> 1, everything else 0 

P_setosa = Perceptron(eta=0.01, n_iter=1000) # note that eta needs to be small here!
P_setosa.fit(X, y_setosa) # hand off the iris data and correct labels to learning algorithm

P_versi = Perceptron(eta=0.01, n_iter=1000) # note that eta needs to be small here!
P_versi.fit(X, y_versi) # hand off the iris data and correct labels to learning algorithm

P_virg = Perceptron(eta=0.01, n_iter=1000) # note that eta needs to be small here!
P_virg.fit(X, y_virg) # hand off the iris data and correct labels to learning algorithm

test = [1, 2]
# test = [6, 5]
# test = [8, 7]
portion_setosa = P_setosa.net_input(test)
portion_versi =  P_versi.net_input(test)
portion_virg = P_virg.net_input(test)

class TriClassPerceptron():
  
  def __init__(self, P0, P1, P2):
    self.perceptrons = [P0, P1, P2]

  def predict(self, X):
    self.predictions = [self.P0.net_input(X), self.P1.net_input(X), self.P2.net_input(X)]
    max_prediction = max(self.predictions)
    if max_prediction == self.predictions[0]: return 0
    if max_prediction == self.predictions[1]: return 1
    if max_prediction == self.predictions[2]: return 2





plot_3_classes(X, y, classifier = TriClassPerceptron(P_setosa, P_versi, P_virg))

# testing input from X
# portion_setosa = P_setosa.net_input(X)
# portion_versi =  P_versi.net_input(X)
# portion_virg = P_virg.net_input(X)

# if portion_setosa > portion_versi and portion_setosa > portion_virg:
#   print("Setosa")
# elif  portion_versi > portion_setosa and portion_versi > portion_virg:
#   print("Versi")
# else:
#   print("Virg")

# # extract the other information defining the classes
# X_iris = df.iloc[0:100, [0, 2]].values  