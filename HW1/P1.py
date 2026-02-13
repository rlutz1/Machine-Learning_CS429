"""
PROBLEM 1 DRIVER

Modify the classes AdalineGD and LogisticRegressionGD 
in the textbook such that the bias
data field b is absorbed by the weight vector w . 
Your program is required to be compatible with the training
programs in the textbook.
"""

from helper_code.Absorbed_Adaline import AdalineGD
from helper_code.Absorbed_LogReg import LogisticRegressionGD
from helper_code.unaltered_original_code.Adaline import Orig_AdalineGD
from helper_code.unaltered_original_code.LogReg import Orig_LogisticRegressionGD
import numpy as np
import pandas as pd

"""
===================================================================
SCRIPTING
===================================================================
"""

# for consistency in training models to ensure they have same params.
i = 1000 # number of iterations
e = 0.01 # learning rate

# use iris data set for testing
s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'

df = pd.read_csv(s,
     header=None,
     encoding='utf-8')

# set up classes for setosa vs versi
y = df.iloc[0:100, 4].values # values in the 4th column of csv -> names of iris
y = np.where(y == "Iris-setosa", 0, 1) # setosa -> 0, versi 1

# extract the other information defining the classes
# specifically sepal and petal lengths
X = df.iloc[0:100, [0, 2]].values  

# train absorbed ada
ada_abs = AdalineGD(eta=e, n_iter=i) 
ada_abs.fit(X, y) 

# train absorbed log
log_abs = LogisticRegressionGD(eta=e, n_iter=i)
log_abs.fit(X, y)

# train original ada
log_orig = Orig_LogisticRegressionGD(eta=e, n_iter=i)
log_orig.fit(X, y)

# train original log
ada_orig = Orig_AdalineGD(eta=e, n_iter=i)
ada_orig.fit(X, y) 

# checking to make sure the weights and bias are the same between them
print("ADA COMPARISON")
print(f"absorbed weights: {ada_abs.w_[0:(ada_abs.w_.size - 1)]}, orig weights: {ada_orig.w_}")
print(f"absorbed bias: {ada_abs.w_[-1]}, orig bias: {ada_orig.b_}")

print("LOG COMPARISON")
print(f"absorbed weights: {log_abs.w_[0:(log_abs.w_.size - 1)]}, orig weights: {log_orig.w_}")
print(f"absorbed bias: {log_abs.w_[-1]}, orig bias: {log_orig.b_}")