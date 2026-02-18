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