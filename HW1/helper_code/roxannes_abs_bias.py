import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt

"""
code from the text book for the adaline learning model.
below are modifications to absorb the bias into the vectors w and x.
"""
class AdalineGD:
  """ADAptive LInear NEuron classifier.
  Parameters
  ------------
  eta : float
  Learning rate (between 0.0 and 1.0)
  n_iter : int
  Passes over the training dataset.
  random_state : int
  Random number generator seed for random weight initialization.
  Attributes
  -----------
  w_ : 1d-array
  Weights after fitting.
  b_ : Scalar
  Bias unit after fitting.
  losses_ : list
  Mean squared error loss function values in each epoch.
  """
  def __init__(self, eta=0.01, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit training data.
    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
    Training vectors, where n_examples
    is the number of examples and
    n_features is the number of features.
    y : array-like, shape = [n_examples]
    Target values.
    Returns
    -------
    self : object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1] + 1) # add one to size for bias
    self.w_[-1] = np.float64(0.) # absorb b into w as the last element
    self.losses_ = []

    for i in range(self.n_iter): 
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output) 
      self.w_[0:(X.shape[1])] += self.eta * 2.0 * X.T.dot(errors) / X.shape[0] # update the weights
      self.w_[-1] += self.eta * 2.0 * errors.mean() # update b in w now
      loss = (errors**2).mean() 
      self.losses_.append(loss)
    # print("after train: ", self.w_)
    return self
  
  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[0:(X.shape[1])])  + self.w_[-1] # b is in w now
  
  def activation(self, X):
    """Compute linear activation"""
    return X
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
  
"""
code from the text book for the logistic regression learning model.
below are modifications to absorb the bias into the vectors w and x.
"""
class LogisticRegressionGD:
  """Gradient descent-based logistic regression classifier.
  Parameters
  ------------
  eta : float
  Learning rate (between 0.0 and 1.0)
  n_iter : int
  Passes over the training dataset.
  random_state : int
  Random number generator seed for random weight
  initialization.
  Attributes
  -----------
  w_ : 1d-array
  Weights after training.
  b_ : Scalar
  Bias unit after fitting.
  losses_ : list
  Mean squared error loss function values in each epoch.
  """
  def __init__(self, eta=0.01, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit training data.
    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
    Training vectors, where n_examples is the
    number of examples and n_features is the
    number of features.
    y : array-like, shape = [n_examples]
    Target values.
    Returns
    -------
    self : Instance of LogisticRegressionGD
    """
    rgen = np.random.RandomState(self.random_state)
    # self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
    # self.b_ = np.float_(0.)
    self.w_ = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1] + 1) # add one to size for bias
    self.w_[-1] = np.float64(0.) # absorb b into w as the last element
    self.losses_ = []

    for i in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output)
      self.w_[0:(X.shape[1])] += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
      self.w_[-1] += self.eta * 2.0 * errors.mean()
      loss = (-y.dot(np.log(output))
              - ((1 - y).dot(np.log(1 - output)))
              / X.shape[0])
      self.losses_.append(loss)
    print("after train ", self.w_)
    return self
  
  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[0:(X.shape[1])])  + self.w_[-1] # b is in w now
  
  def activation(self, z):
    """Compute logistic sigmoid activation"""
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
  