import numpy as np

"""
MODIFIED LOGISTIC REGERSSION FOR ABSORBED BIAS

baseline code is from the textbook for the logistic regression learning model.

below are modifications to absorb the bias into the vectors w and x,
comments mark the important changes.
"""
class LogisticRegressionGD:
  
  def __init__(self, eta=0.01, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  def fit(self, X, y):
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1] + 1) # add one to size for bias
    self.w_[-1] = np.float64(0.) # absorb b into w as the last element
    self.losses_ = []

    for i in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output)
      self.w_[0:(X.shape[1])] += self.eta * 2.0 * X.T.dot(errors) / X.shape[0] # update the weights
      self.w_[-1] += self.eta * 2.0 * errors.mean() # bias now in weight vector
      loss = ((-y.dot(np.log(output))
              - ((1 - y).dot(np.log(1 - output))))
              / X.shape[0])
      self.losses_.append(loss)
    return self
  
  # function to add ones to end of given samples
  # in order to make use of the absorbed bias during
  # net_input calculations.
  def extend_samples(self, X):
    ones = [[1]]* X.shape[0]
    X = np.hstack((X, ones))
    return X

  def net_input(self, X):
    """Calculate net input"""
    X = self.extend_samples(X) # extend X with ones
    return np.dot(X, self.w_) # w (dot) x, with bias absorbed.
  
  def activation(self, z):
    """Compute logistic sigmoid activation"""
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)