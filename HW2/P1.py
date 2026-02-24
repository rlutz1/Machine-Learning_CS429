"""
Implement a Python class named LinearSVC which learns a linear Support Vector Classifier (SVC)
from a set of training data. The class is required to have the following components:
• A constructor init which initialize an SVC using the given learning rate, number of epochs
and a random seed. (Similar to the perceptron class in our textbook.)
• A training function fit which trains the SVC based on a given labeled dataset. We consider
the soft-margin SVC using a hinge loss. You are required to use L2-regularization and expose the
regularization parameter as a hyperparameter.
• A function net input which computes the preactivation value for a given input sample.
• A function predict which generates the prediction for a given input sample.
"""
import numpy as np


class LinearSVC:
    """
    Linear Support Vector Classifier (soft-margin)
    trained with gradient descent using hinge loss
    and L2 regularization.
    """

    def __init__(self, eta=0.01, epochs=50, C=1.0, random_state=1):
        """
        Parameters
        ----------
        eta : float
            Learning rate
        epochs : int
            Number of passes over training data
        C : float
            Regularization strength (inverse of lambda)
        random_state : int
            Seed for reproducibility
        """
        self.eta = eta
        self.epochs = epochs
        self.C = C
        self.random_state = random_state

    def fit(self, X, y):
        """
        Train the SVC using hinge loss and L2 regularization.
        y must contain labels {-1, +1}
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.

        self.losses_ = []

        for _ in range(self.epochs):
            loss = 0
            for xi, target in zip(X, y):
                condition = target * self.net_input(xi)
                if condition >= 1:
                    # Only regularization term contributes
                    self.w_ -= self.eta * (2 * self.w_)
                else:
                    # Hinge loss + regularization
                    self.w_ -= self.eta * (2 * self.w_ - self.C * target * xi)
                    self.b_ += self.eta * self.C * target

                # Accumulate hinge loss
                loss += max(0, 1 - condition)
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Compute linear pre-activation."""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label {-1, +1}."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
