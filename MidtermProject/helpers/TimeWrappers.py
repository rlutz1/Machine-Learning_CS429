"""
this file holds a time wrapper class used to fit into a 
pipeline and hold the timing of fit and transform (if appropriate)
functions of a transformer or classifier model.

this is what is used when compiling time information during
training.
"""
import time
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# for a transformer, in which we expect at minimum a 
# fit() and transform() function available.
class TimedTransform(BaseEstimator,TransformerMixin):

  def __init__(self, transformer):
    self.transformer = transformer

    self.fit_time = 0
    self.transform_time = 0
    self.fit_and_transform_time = 0

  # override the fit by simply adding a timer,
  # then calling original fit.
  def fit(self, X, y=None):
    print("fitting")

    start = time.perf_counter()
    self.transformer.fit(X, y)
    end = time.perf_counter()
    self.fit_time = (end - start)

    return self
  
  # override the transform by simply adding a timer,
  # then calling original transform.
  def transform(self, X):
    print("transforming")

    start = time.perf_counter()
    X_transform = self.transformer.transform(X)
    end = time.perf_counter()
    self.transform_time = (end - start)

    return X_transform
  
  def fit_transform(self, X, y=None):
    print("fit-transforming")

    start = time.perf_counter()
    X_fit_and_trans = self.transformer.fit_transform(X, y)
    end = time.perf_counter()
    self.fit_and_transform_time = (end - start)

    return X_fit_and_trans
  
  # convenience method
  def total_time(self):
    return self.fit_and_transform_time + self.fit_time + self.transform_time
  
  # convenience method
  def clear_times(self):
    self.fit_time = 0
    self.transform_time = 0
    self.fit_and_transform_time = 0

  
# for a transformer, in which we expect at minimum a 
# fit() and transform() function available.
class TimedClassifier(BaseEstimator, ClassifierMixin):

  def __init__(self, classifier):
    self.classifier = classifier
    self.fit_time = 0
    self._is_fitted = False

  # override the fit by simply adding a timer,
  # then calling original fit.
  def fit(self, X, y=None):
    print("fitting")

    start = time.perf_counter()
    self.classifier.fit(X, y)
    end = time.perf_counter()
    self.fit_time = (end - start)

    self._is_fitted = True # set to true, for scikit learn check

    return self
  
  # simply for accessing the predict once fitting achieved
  def predict(self, X):
    print("predicting")
    check_is_fitted(self) # best practice validation check
    return self.classifier.predict(X)
  
  # simply for accessing the predict once fitting achieved
  def score(self, X, y):
    print("scoring")
    check_is_fitted(self) # best practice validation check
    return self.classifier.score(X, y)
  
  # convenience method
  def total_time(self):
    return self.fit_time # only care for fit time here
  
  # convenience method
  def clear_times(self):
    self.fit_time = 0

  # DO NOT REMOVE.
  # sklearn relies on this to check if an estimator
  # is fitted during the pipeline run.
  # https://scikit-learn.org/stable/auto_examples/developing_estimators/sklearn_is_fitted.html#sphx-glr-auto-examples-developing-estimators-sklearn-is-fitted-py
  def __sklearn_is_fitted__(self):
    return hasattr(self, "_is_fitted") and self._is_fitted