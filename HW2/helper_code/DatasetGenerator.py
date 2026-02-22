"""
helper functions to write and read datasets from CSV
"""

import numpy as np
from sklearn.model_selection import train_test_split


"""
---------------------------------------------------------------
CONSTANTS
---------------------------------------------------------------
"""

TEST_APPEND = "TEST"
TRAIN_APPEND = "TRAIN"


"""
---------------------------------------------------------------
UTILITY FUNCTIONS
---------------------------------------------------------------
"""

"""
given a filename, samples with their labels, and a random seed,
split the training and testing samples 70/30
and write to an associated file for reproducability.

CSV file format: 
each row is a sample. last element in csv row is the class it belongs to (1/-1).
everything before class is associated features of the sample.
"""
def generate_data_sets_to_file(filename, samples, true_labels, rand_seed):

  X_train, X_test, y_train, y_test = train_test_split( # split with scikit learn, 70/30
      samples, true_labels, test_size=0.30, random_state=rand_seed)
  # print(X_test)
  # print(y_test)

  # write training data 
  with open(f'datasets/{filename}_{TRAIN_APPEND}.csv', 'w') as f:
      dataset_num = 0
      for sample in X_train:
        for s in sample:
          print(s, file=f, end=',')
        print(y_train[dataset_num], file=f, end='\n')  
        dataset_num += 1

  # write testing data
  with open(f'datasets/{filename}_{TEST_APPEND}.csv', 'w') as f:
      dataset_num = 0
      for sample in X_test:
        for s in sample:
          print(s, file=f, end=',')
        print(y_test[dataset_num], file=f, end='\n')  
        dataset_num += 1

"""
ease of use function to read back the data from the CSV
into samples and true labels.
X -> features
y -> true label (1/-1)
"""
def read_data_sets(filename):
  X = []
  y = [] 
  temp = []

  with open(f'datasets/{filename}.csv', 'r') as f:
    whole_file = f.read()
    samples = whole_file.split("\n")
    for s in samples:
      if s:
        features_and_class = s.split(",") 
        for feature in features_and_class[:-1]:
          temp.append(np.float64(feature))
        X.append(temp)
        y.append(int(features_and_class[-1]))
        temp = []
  
  return X, y
