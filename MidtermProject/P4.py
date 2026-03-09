"""
Task 4:

FOR BOTH MNIST and MNIST FASHION!

Write a program that trains a finite set of SVCs using bootstrap aggregating. 
Please use at least 8 models. The training dataset should be divided into multiple 
disjoint sets and each individual model is trained based on a subset. 
The final prediction for an image is obtained via voting. Please use the three kernels
along with their best hyperparameters you found in the previous task. 
Please do not use sklearn.ensemble.
"""