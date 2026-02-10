"""
textbook code for the vanilla perceptron.
"""
import numpy as np

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
