"""
Write a Python function make classification which generates a set of linearly separable data
based on a random separation hyperplane. We learned that an (d - 1)-dimensional hyperplane can be defined
as the set of points in Rd satisfying an equation `aT `x = b, i.e., {`x ∈ Rd | `aT `x = b}. For simplicity, we
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