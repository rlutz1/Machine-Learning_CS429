"""
Write a Python function make classification which generates a set of linearly separable data
based on a random separation hyperplane. We learned that an (d - 1)-dimensional hyperplane can be defined
as the set of points in R^d satisfying an equation `aT `x = b, i.e., {`x ∈ Rd | `aT `x = b}. For simplicity, we
assume that b = 0, then the hyperplane can be determined by a random vector `a. We use this idea to design
the following algorithm to generate random data which are linearly separable regarding to any number and
dimension:

• Step 1. Randomly generate a d-dimensional vector `a.
• Step 2. Randomly select n samples `x1, . . . , `xn in the range of [-u, u] in each dimension. You may use
a uniform or Gaussian distribution to do so.
• Step 3. Give each `xi a label yi such that if `aT `x < 0 then yi = -1, otherwise yi = 1.

Therefore, your function should have the following parameters that should given by the user: d, n, u, and a
random seed for reproducing the data. You need to additionally subdivide the dataset to a training dataset
(70%) and a test dataset (30%). You may use the scikit-learn function to do so, but make sure that you
specify the random seed such that the subdivision is reproducible.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""
function to generate linerally separable data

d         -> dimensions, or number features used for training; default 2
n         -> number of samples to generate with d features; default 100
u         -> defines range of [-u, u] in EACH dimension to generate samples in; default [-10, 10]
rand_seed -> random seed for random generation; default is 1
"""

def generate_line_sep_data(d=2, n=100, u=10, rand_seed=1):
  rgen = np.random.RandomState(rand_seed) # random generator
  
  # (1) generate a d dimensional hyperplane 
  a = rgen.uniform(-1, 1, d) # generate d random values for a, b = 0

  # (2) randomly select n samples from [-u, u] in all dimensions
  samples = [[] for _ in range(n)] # n empty samples
  for _ in range(d): # for each dimension
    new_samples = rgen.uniform(-u, u, n) # draw n features from -u to u from uniform distr
    
    for (i, sample) in zip(range(n), samples):
      sample.append(new_samples[i])

  # test: the dot prod zero regeneration
  # samples[0] = [0 for i in range(d)]
  # samples[1] = [0 for i in range(d)]

  # (3) give each `xi a label yi such that if `aT `x < 0 then yi = -1, otherwise yi = 1.
  true_labels = []

  for (i, s) in zip(range(n), samples):
    dot_prod = np.dot(a, s) # ax = ?
    
    if dot_prod < 0: true_labels.append(-1)
    
    elif dot_prod > 0: true_labels.append(1)
    
    else: # ax = 0, sample is on the line. regen the sample and test until a non-zero.
      print("Conducting a dot prod 0 swap.")
      
      while (dot_prod == 0):
        new_sample = rgen.uniform(-u, u, d) # make 1 new sample with d features
        dot_prod = np.dot(a, new_sample) # see what dot prod is now, break loop when not 0
      
      for j in range(d): samples[i][j] = new_sample[j] # replace these values with the new sample
      
      if dot_prod < 0: true_labels.append(-1) # update the true labels
      elif dot_prod > 0: true_labels.append(1)

  return (a, samples, true_labels)

"""
---------------------------------------------------------------
SCRIPT TO RUN
---------------------------------------------------------------
"""


TEST_APPEND = "TEST"
TRAIN_APPEND = "TRAIN"




def plot(d, u, samples, true_labels):
  if d == 2: # only if 2d, plot 2d demo
    x_hyperplane = np.linspace(-u, u, 100) # Creates 100 evenly spaced points from -u to u
    y_hyperplace = (-(a[0] / a[1])) * x_hyperplane  # y = (-a0/a1)x + 0 -> line equation
    plt.plot(x_hyperplane, y_hyperplace, 'g') # plot the line

    samples = np.array(samples) # so i can use some special syntax below
    for idx, cl in enumerate(np.unique(true_labels)):
        print(idx, cl)
        plt.scatter(
          x=samples[true_labels == cl, 0],
          y=samples[true_labels == cl, 1],
          alpha=0.8,
          c='red' if cl == -1 else 'blue',
          marker='o' if cl == -1 else '^',
          label= '-1' if cl == -1 else '1',
          edgecolor='black')
    plt.ylim((-u, u))
    plt.xlim((-u, u))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Generic Linearly Separable Data")
    plt.legend()

    plt.show()

def generate_data_sets_to_file(filename, samples, true_labels):

  # (a, samples, true_labels) = generate_line_sep_data(d=d, n=n, u=u, rand_seed=rand_seed) # generate test data
  X_train, X_test, y_train, y_test = train_test_split( # split with scikit learn, 70/30
      samples, true_labels, test_size=0.30, random_state=rand_seed)
  # print(X_test)
  # print(y_test)

  with open(f'datasets/{filename}_{TRAIN_APPEND}.csv', 'w') as f:
      dataset_num = 0
      for sample in X_train:
        for s in sample:
          print(s, file=f, end=',')
        print(y_train[dataset_num], file=f, end='\n')  
        dataset_num += 1

  with open(f'datasets/{filename}_{TEST_APPEND}.csv', 'w') as f:
      dataset_num = 0
      for sample in X_test:
        for s in sample:
          print(s, file=f, end=',')
        print(y_test[dataset_num], file=f, end='\n')  
        dataset_num += 1





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

# ease of change variables
d = 2
n = 10
u = 100
rand_seed = 42

(a, samples, true_labels) = generate_line_sep_data(d=d, n=n, u=u, rand_seed=rand_seed) # generate test data
plot(d, u, samples, true_labels) # plot data, only if d = 2

generate_data_sets_to_file("test", samples, true_labels)
(X, y) = read_data_sets(f"test_{TEST_APPEND}")
print(X, "\n", y)
(X, y) = read_data_sets(f"test_{TRAIN_APPEND}")
print(X, "\n", y)
