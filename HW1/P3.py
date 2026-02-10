"""
Adaline and perceptron learning can only be used for binary classification, however, the Iris dataset
has 3 classes: setosa, versicolor and virginica. If you are only allowed to use perceptrons but the number is
not limited, how would you like to perform a multiclass classification for the whole Iris data set? Please write
a program for this task
"""
from helper_code.Perceptron import Perceptron
from helper_code.Plotters import plot_2_params
from helper_code.roxannes_abs_bias import AdalineGD, LogisticRegressionGD
import numpy as np
import pandas as pd

PERCEPTRON_MODE = "Perceptron"
ADALINE_MODE = "Adaline"
LOG_REG_MODE = "Logistic Regression"

mode = LOG_REG_MODE


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
X = df.iloc[0:, [0, 2]].values  
y = df.iloc[0:, 4].values

# replace string names with numbers.
# setosa is class 0, versi class 1, virg class 2
conditions = [y == "Iris-setosa", y == "Iris-versicolor", y == "Iris-virginica"]
choices = [0, 1, 2]
y = np.select(conditions, choices)

# set up classes for setosa, versi, virg
y_setosa = df.iloc[0:, 4].values # values in the 4th column of csv -> names of iris
y_setosa = np.where(y_setosa == "Iris-setosa", 1, 0) # setosa -> 1, everything else 0 

y_versi = df.iloc[0:, 4].values # values in the 4th column of csv -> names of iris
y_versi = np.where(y_versi == "Iris-versicolor", 1, 0) # versicolor -> 1, everything else 0 

y_virg = df.iloc[0:, 4].values # values in the 4th column of csv -> names of iris
y_virg = np.where(y_virg == "Iris-virginica", 1, 0) # virginica -> 1, everything else 0 

if mode == PERCEPTRON_MODE:
  P_setosa = Perceptron(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_setosa.fit(X, y_setosa) # hand off the iris data and correct labels to learning algorithm

  P_versi = Perceptron(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_versi.fit(X, y_versi) # hand off the iris data and correct labels to learning algorithm

  P_virg = Perceptron(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_virg.fit(X, y_virg) # hand off the iris data and correct labels to learning algorithm

elif mode == ADALINE_MODE:
  P_setosa = AdalineGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_setosa.fit(X, y_setosa) # hand off the iris data and correct labels to learning algorithm

  P_versi = AdalineGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_versi.fit(X, y_versi) # hand off the iris data and correct labels to learning algorithm

  P_virg = AdalineGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_virg.fit(X, y_virg) # hand off the iris data and correct labels to learning algorithm

else:
  P_setosa = LogisticRegressionGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_setosa.fit(X, y_setosa) # hand off the iris data and correct labels to learning algorithm

  P_versi = LogisticRegressionGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_versi.fit(X, y_versi) # hand off the iris data and correct labels to learning algorithm

  P_virg = LogisticRegressionGD(eta=0.01, n_iter=10000) # note that eta needs to be small here!
  P_virg.fit(X, y_virg) # hand off the iris data and correct labels to learning algorithm

# this is by no means the best way to do this, but the plotter 
# can use the predict method. so, this is 
class TriClassPerceptron():
  
  def __init__(self, P0, P1, P2):
    self.P0 = P0
    self.P1 = P1
    self.P2 = P2

  # essentially:
  # get all 3 perceptions net_input value (a real number)
  # and get the max of all 3. 
  # 0 -> setosa perceptron
  # 1 -> versicolor perceptron
  # 2 -> virginia perceptron
  # using net input for now since ALL models have that
  # activation is probably better practice, but doesn't have
  # discernable difference on decision regions
  def predict(self, X):
    val0 = self.P0.net_input(X)
    val1 = self.P1.net_input(X)
    val2 = self.P2.net_input(X)
    # made no discernable difference in the plot:
    # perceptron doesn't have activation func, to note
    # val0 = self.P0.activation(self.P0.net_input(X)) 
    # val1 = self.P1.activation(self.P1.net_input(X))
    # val2 = self.P2.activation(self.P2.net_input(X))
    y = np.array([])
    for i in range(X.shape[0]): # for each of the test cases
      maximum = max(val0[i], val1[i], val2[i]) # get max of 3
      # not the best way, but works for now.
      # say the maximum is the class we will go with.
      if maximum == val0[i]:   y = np.append(y, 0) 
      elif maximum == val1[i]: y = np.append(y, 1)
      elif maximum == val2[i]: y = np.append(y, 2)
      else: print("Something's off in the TriClassPerceptron ", maximum)

    return y

# finally, generate a plot with visible decision regions
plot_2_params(X, y, classifier = TriClassPerceptron(P_setosa, P_versi, P_virg), title=f"{mode} Model with 3 Classes", x_axis_title="Sepal Length", y_axis_title="Petal Length")