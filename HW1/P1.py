"""
Modify the classes AdalineGD and LogisticRegressionGD 
in the textbook such that the bias
data field b is absorbed by the weight vector w . 
Your program is required to be compatible with the training
programs in the textbook.

REPORT:
Explain how the bias is transformed to an extra weight 
and why the translated model is equivalent
to the original one.
"""
# from log_ada_absorbed_bias import AdalineGD, LogisticRegressionGD
# from plotters import plot_decision_regions
from helper_code.roxannes_abs_bias import AdalineGD, LogisticRegressionGD, Orig_AdalineGD, Orig_LogisticRegressionGD
import numpy as np
import pandas as pd



# ===================================================================
# SCRIPTING
# ===================================================================

# for consistency in training models so i don't lose my mind
i = 1000
e = 0.01

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
print('From URL:', s)

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

# i'm checking to make sure the weights and bias are the same between them
print("ADA COMPARISON")
print(f"absorbed weights (bias last): {ada_abs.w_}, orig weights: {ada_orig.w_}")
print(f"absorbed bias: {ada_abs.w_[-1]}, orig bias: {ada_orig.b_}")

print("LOG COMPARISON")
print(f"absorbed weights (bias last): {log_abs.w_}, orig weights: {log_orig.w_}")
print(f"absorbed bias: {log_abs.w_[-1]}, orig bias: {log_orig.b_}")