"""
Task 3: SVC Tuning and data transformation.

FOR BOTH MNIST and MNIST FASHION!

Make a machine learning pipeline using scikit-learn to integrate the following steps:

3.1: 
Standardize the (flattened) samples. Notice that the preprocess mapping is computed based
on the training data, but the same transformation should also be applied to the test data

3.2:
Dimensionality reduction on the data. You are required to use two ways: Principal Component
Analysis (PCA) and Linear Discriminant Analysis (LDA), to reduce the number of features of the data.
The original dimensionality is 784, you are required to consider the reduced dimensionalities: 50, 100 and
200 for PCA. Similar to the previous task, the reduction mappings should be derived from the training
set but should also be used to compress the test data.

3.3:
Build a Support Vector Classifier (SVC) with a kernel for classifying the compressed data.
You should use the scikit-learn SVC class. You are required to consider the three kernels along with their
hyperparameters:
• 'linear' - Linear kernel, the only hyperparameter is C.
• 'rbf' - Radial basis function kernel, the hyperparameters are C and gamma.
• 'poly' - Polynomial kernel, the hyperparameters are C, gamma and degree.
You are required to tune the hyperparameters and choose the best setting for each kernel. When tuning
the parameters, you need to measure the prediction error on both training set and test set.
"""