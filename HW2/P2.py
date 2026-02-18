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
"""
function to generate linerally separable data

d         -> dimensions, or number features used for training; default 2
n         -> number of samples to generate with d features; default 100
u         -> defines range of [-u, u] in EACH dimension to generate samples in; default [-10, 10]
rand_seed -> random seed for random generation; default is 1
"""

def generate_line_sep_data(d=2, n=100, u=10, rand_seed=1):
  # (1) generate a d dimensional hyperplane 
  a = np.random.uniform(-u, u, d)
  # print(a)

  # (2) randomly select n samples from [-u, u] in all dimensions
  samples = [[] for _ in range(n)] # n empty samples
  for _ in range(d): # for each dimension
    new_samples = np.random.uniform(-u, u, n) # draw n features from -u to u from uniform distr
    for (i, sample) in zip(range(n), samples):
      sample.append(new_samples[i])
  # print(samples)

  # (3) give each `xi a label yi such that if `aT `x < 0 then yi = -1, otherwise yi = 1.
  true_labels = []
  for s in samples:
    if np.dot(a, s) < 0: true_labels.append(-1)
    elif np.dot(a, s) > 0: true_labels.append(1)
    else: print("0!!!!!!!!!")

  print(true_labels)


generate_line_sep_data(d=2, n=3)