import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.layout_engine as le
import matplotlib.colors as color
import numpy as np

import os

# setting up paths for data read
MNIST_DATA_PATH = os.path.join(os.getcwd(), "data", "P3", "mnist") # directory to write data to for collection
MNIST_FASHION_DATA_PATH = os.path.join(os.getcwd(), "data", "P3", "mnist-fashion") # directory to write data to for collection
INITIAL_FIT_TIMES = "initial-fit-times.csv" # for the initial fit times of everything in the pipeline
TEST_AND_TRAIN_ACC = "test-train-acc.csv" # for the accuracy of final SVC on test, then train data

# time collection of each primary dataset run
times_mnist = pd.read_csv(os.path.join(MNIST_DATA_PATH, INITIAL_FIT_TIMES))
times_mnist_fashion = pd.read_csv(os.path.join(MNIST_FASHION_DATA_PATH, INITIAL_FIT_TIMES))

# accuracy collection of each primary dataset run
acc_mnist = pd.read_csv(os.path.join(MNIST_DATA_PATH, TEST_AND_TRAIN_ACC))
acc_mnist_fashion = pd.read_csv(os.path.join(MNIST_FASHION_DATA_PATH, TEST_AND_TRAIN_ACC))


"""
=====================================================================================================
LINEAR KERNELS
=====================================================================================================
"""

# let's see what 5000 iteration with linear shows for time it took to train each reducion
# x axis will be parameter settings
# y axis will be the time it took to fit
# each bar is the dim reduction

time_mnist_entries = times_mnist.loc[
  (times_mnist["kernel"] == "linear") &
  (times_mnist["num iterations"] == 5000) 
  ]

time_fashion_entries = times_mnist_fashion.loc[
  (times_mnist_fashion["kernel"] == "linear") &
  (times_mnist_fashion["num iterations"] == 5000) 
  ]

# we will also pull the accuracy to compare through reductions
acc_mnist_entries = acc_mnist.loc[
  (acc_mnist["kernel"] == "linear") &
  (acc_mnist["num iterations"] == 5000) 
  ]

acc_fashion_entries = acc_mnist_fashion.loc[
  (acc_mnist_fashion["kernel"] == "linear") &
  (acc_mnist_fashion["num iterations"] == 5000) 
  ]

# we will also pull the accuracy to compare through reductions
acc_mnist_entries_10000 = acc_mnist.loc[
  (acc_mnist["kernel"] == "linear") &
  (acc_mnist["num iterations"] == 10000) 
  ]

acc_fashion_entries_10000 = acc_mnist_fashion.loc[
  (acc_mnist_fashion["kernel"] == "linear") &
  (acc_mnist_fashion["num iterations"] == 10000) 
  ]

# plot for times
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

# -------------------------------------
# time for linear, mnist
# -------------------------------------

parameters = np.unique(time_mnist_entries["C"]) # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
times = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for p in parameters:
  for (bar_name, csv_col) in dim_reducers:
    time = time_mnist_entries.loc[
      (time_mnist_entries["dim reducer"] == csv_col) &
      (time_mnist_entries["C"] == p)
    ]
    # print(time["fit time"].values[0])
    times[bar_name].append(np.round(time["fit time"].values[0], 2)) # round by 2, cleaner

# print(vals)

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in times.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Time to Fit (s)')
ax[0].set_xlabel('Parameter (C)')
ax[0].set_title('Time to Fit per Reduction Method, Linear Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 100)

# -------------------------------------
# time for linear, fashion
# -------------------------------------

parameters = np.unique(time_fashion_entries["C"]) # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
times = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for p in parameters:
  for (bar_name, csv_col) in dim_reducers:
    time = time_fashion_entries.loc[
      (time_fashion_entries["dim reducer"] == csv_col) &
      (time_fashion_entries["C"] == p)
    ]
    times[bar_name].append(np.round(time["fit time"].values[0], 2)) # round by 2, cleaner

# print(vals)

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in times.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Time to Fit (s)')
ax[1].set_xlabel('Parameter (C)')
ax[1].set_title('Time to Fit per Reduction Method, Linear Kernel, MNIST Fashion, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 100)


# plot for accuracy
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

# -------------------------------------
# test acc for linear, mnist
# -------------------------------------

parameters = np.unique(acc_mnist_entries["C"]) # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the test acc
for p in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_mnist_entries.loc[
      (acc_mnist_entries["dim reducer"] == csv_col) &
      (acc_mnist_entries["C"] == p)
    ]
    print(acc["test acc"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, acc in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, acc, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Parameter (C)')
ax[0].set_title('Accuracy per Reduction Method, Linear Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 1.5)

# -------------------------------------
# test acc for linear, fashion
# -------------------------------------

parameters = np.unique(acc_fashion_entries["C"]) # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the test acc
for p in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_fashion_entries.loc[
      (acc_fashion_entries["dim reducer"] == csv_col) &
      (acc_fashion_entries["C"] == p)
    ]
    print(acc["test acc"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, acc in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, acc, width, label=attribute)
    ax[1].bar_label(rects, padding=5)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Test Accuracy')
ax[1].set_xlabel('Parameter (C)')
ax[1].set_title('Accuracy per Reduction Method, Linear Kernel, MNIST Fashion, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 1.5)

"""
=====================================================================================================
RBF KERNELS
=====================================================================================================
"""

# let's see what 5000 iteration with rbf shows for time it took to train each reducion
# x axis will be parameter settings
# y axis will be the time it took to fit
# each bar is the dim reduction
# we will pick a few select combinations, all 49 is a bit much and not needed for this

time_mnist_entries = times_mnist.loc[
  (times_mnist["kernel"] == "rbf") &
  (times_mnist["num iterations"] == 5000) 
  ]

time_fashion_entries = times_mnist_fashion.loc[
  (times_mnist_fashion["kernel"] == "rbf") &
  (times_mnist_fashion["num iterations"] == 5000) 
  ]

# we will also pull the accuracy to compare through reductions
acc_mnist_entries = acc_mnist.loc[
  (acc_mnist["kernel"] == "rbf") &
  (acc_mnist["num iterations"] == 5000) 
  ]

acc_fashion_entries = acc_mnist_fashion.loc[
  (acc_mnist_fashion["kernel"] == "rbf") &
  (acc_mnist_fashion["num iterations"] == 5000) 
  ]

# plot for times
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

# -------------------------------------
# time for rbf, mnist
# -------------------------------------



# we're going to hand select a few combos from RBF because too many will be a mess.
# 7 groups worked nicely for linear, so we will choose:
parameters = [
 # (C, gamma)
   (0.1, 0.1),
   (0.1, 1),
   (1, 0.1),
   (1, 1),
   (10, 0.1),
   (10, 1),
   (10, 10)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
times = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma in parameters:
  for (bar_name, csv_col) in dim_reducers:
    time = time_mnist_entries.loc[
      (time_mnist_entries["dim reducer"] == csv_col) &
      (time_mnist_entries["C"] == c) &
      (time_mnist_entries["gamma"] == gamma) 
    ]
    # print(time["fit time"].values[0])
    times[bar_name].append(np.round(time["fit time"].values[0], 2)) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in times.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Time to Fit (s)')
ax[0].set_xlabel('Parameter (C, gamma)')
ax[0].set_title('Time to Fit per Reduction Method, RBF Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 1200)

# -------------------------------------
# time for rbf, fashion
# -------------------------------------

parameters = [
   # (C, gamma)
   (0.1, 0.1),
   (0.1, 1),
   (1, 0.1),
   (1, 1),
   (10, 0.1),
   (10, 1),
   (10, 10)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
times = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma in parameters:
  for (bar_name, csv_col) in dim_reducers:
    time = time_fashion_entries.loc[
      (time_fashion_entries["dim reducer"] == csv_col) &
      (time_fashion_entries["C"] == c) &
      (time_fashion_entries["gamma"] == gamma) 
    ]
    # print(time["fit time"].values[0])
    times[bar_name].append(np.round(time["fit time"].values[0], 2)) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in times.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Time to Fit (s)')
ax[1].set_xlabel('Parameter (C, gamma)')
ax[1].set_title('Time to Fit per Reduction Method, RBF Kernel, MNIST Fashion, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 1200)


# plot for accuracy
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

# -------------------------------------
# test acc for rbf, mnist
# -------------------------------------

# we're going to hand select a few combos from RBF because too many will be a mess.
# 7 groups worked nicely for linear, so we will choose:
parameters = [
 # (C, gamma)
   (1, 0.001),
   (1, 0.01),
   (1, 0.1),
   (1, 1),
   (1, 10),
   (1, 100),
   (1, 1000)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_mnist_entries.loc[
      (acc_mnist_entries["dim reducer"] == csv_col) &
      (acc_mnist_entries["C"] == c) &
      (acc_mnist_entries["gamma"] == gamma) 
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Parameter (C, gamma)')
ax[0].set_title('Test Accuracy per Reduction Method, RBF Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 1.5)

# -------------------------------------
# test acc for rbf, fashion
# -------------------------------------

# prepare bar graph information
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_fashion_entries.loc[
      (acc_fashion_entries["dim reducer"] == csv_col) &
      (acc_fashion_entries["C"] == c) &
      (acc_fashion_entries["gamma"] == gamma) 
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Test Accuracy')
ax[1].set_xlabel('Parameter (C, gamma)')
ax[1].set_title('Test Accuracy per Reduction Method, RBF Kernel, MNIST, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 1.5)

# trying ranges of gamma
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

# 7 groups worked nicely for linear, so we will choose:
parameters = [
 # (C, gamma)
   (0.001, 1),
   (0.01, 1),
   (0.1, 1),
   (1, 1),
   (10, 1),
   (100, 1),
   (1000, 1)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_mnist_entries.loc[
      (acc_mnist_entries["dim reducer"] == csv_col) &
      (acc_mnist_entries["C"] == c) &
      (acc_mnist_entries["gamma"] == gamma) 
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Parameter (C, gamma)')
ax[0].set_title('Test Accuracy per Reduction Method, RBF Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 1.5)

# -------------------------------------
# test acc for rbf, fashion
# -------------------------------------

# prepare bar graph information
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_fashion_entries.loc[
      (acc_fashion_entries["dim reducer"] == csv_col) &
      (acc_fashion_entries["C"] == c) &
      (acc_fashion_entries["gamma"] == gamma) 
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Test Accuracy')
ax[1].set_xlabel('Parameter (C, gamma)')
ax[1].set_title('Test Accuracy per Reduction Method, RBF Kernel, MNIST, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 1.5)

"""
=====================================================================================================
POLY KERNELS
=====================================================================================================
"""

# let's see what 5000 iteration with rbf shows for time it took to train each reducion
# x axis will be parameter settings
# y axis will be the time it took to fit
# each bar is the dim reduction
# we will pick a few select combinations, all 49 is a bit much and not needed for this

time_mnist_entries = times_mnist.loc[
  (times_mnist["kernel"] == "poly") &
  (times_mnist["num iterations"] == 5000) 
  ]

time_fashion_entries = times_mnist_fashion.loc[
  (times_mnist_fashion["kernel"] == "poly") &
  (times_mnist_fashion["num iterations"] == 5000) 
  ]


# we will also pull the accuracy to compare through reductions
acc_mnist_entries = acc_mnist.loc[
  (acc_mnist["kernel"] == "poly") &
  (acc_mnist["num iterations"] == 5000) 
  ]

acc_fashion_entries = acc_mnist_fashion.loc[
  (acc_mnist_fashion["kernel"] == "poly") &
  (acc_mnist_fashion["num iterations"] == 5000) 
  ]


# plot times
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

# -------------------------------------
# fit time for poly, mnist
# -------------------------------------

parameters = [
   # (C, gamma, degree)
   (0.1, 0.1, 3),
   (0.1, 1, 3),
   (1, 0.1, 3),
   (1, 1, 3),
   (1, 10, 3),
   (10, 0.1, 3),
   (10, 1, 3)
   ] # our x axis groups


# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
times = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    time = time_mnist_entries.loc[
      (time_mnist_entries["dim reducer"] == csv_col) &
      (time_mnist_entries["C"] == c) &
      (time_mnist_entries["gamma"] == gamma) &
      (time_mnist_entries["degree"] == degree)
    ]
    times[bar_name].append(np.round(time["fit time"].values[0], 2)) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


for attribute, time in times.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Time to Fit (s)')
ax[0].set_xlabel('Parameter (C)')
ax[0].set_title('Time to Fit per Reduction Method, Poly Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 100)

# add fashion set below

# -------------------------------------
# fit time for poly, fashion
# -------------------------------------

parameters = [
 # (C, gamma, degree)
   (0.1, 0.1, 3),
   (0.1, 1, 3),
   (1, 0.1, 3),
   (1, 1, 3),
   (1, 10, 3),
   (10, 0.1, 5),
   (100, 1, 3)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
times = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    time = time_fashion_entries.loc[
      (time_fashion_entries["dim reducer"] == csv_col) &
      (time_fashion_entries["C"] == c) &
      (time_fashion_entries["gamma"] == gamma) &
      (time_fashion_entries["degree"] == degree) 
    ]
    # print(time["fit time"].values[0])
    times[bar_name].append(np.round(time["fit time"].values[0], 2)) # round by 2, cleaner

# print(vals)

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in times.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Time to Fit (s)')
ax[1].set_xlabel('Parameter (C)')
ax[1].set_title('Time to Fit per Reduction Method, Poly Kernel, MNIST Fashion, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 100)


# plot acc
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')


# -------------------------------------
# test acc for poly, mnist
# -------------------------------------

# we're going to hand select a few combos from RBF because too many will be a mess.
# 7 groups worked nicely for linear, so we will choose:
parameters = [
 # (C, gamma, degree)
   (1, 1, 1),
   (1, 1, 2),
   (1, 1, 3),
   (1, 1, 4),
   (1, 1, 5)
  #  (10, 0.1, 3),
  #  (10, 1, 3)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_mnist_entries.loc[
      (acc_mnist_entries["dim reducer"] == csv_col) &
      (acc_mnist_entries["C"] == c) &
      (acc_mnist_entries["gamma"] == gamma) &
      (acc_mnist_entries["degree"] == degree)
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Parameter (C, gamma, degree)')
ax[0].set_title('Test Accuracy per Reduction Method, Poly Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 1.5)

# -------------------------------------
# test acc for poly, fashion
# -------------------------------------

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_fashion_entries.loc[
      (acc_fashion_entries["dim reducer"] == csv_col) &
      (acc_fashion_entries["C"] == c) &
      (acc_fashion_entries["gamma"] == gamma) &
      (acc_fashion_entries["degree"] == degree)
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Test Accuracy')
ax[1].set_xlabel('Parameter (C, gamma, degree)')
ax[1].set_title('Test Accuracy per Reduction Method, Poly Kernel, MNIST, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 1.5)


# plot acc
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')


# -------------------------------------
# test acc for poly, mnist
# -------------------------------------

# we're going to hand select a few combos from RBF because too many will be a mess.
# 7 groups worked nicely for linear, so we will choose:
parameters = [
 # (C, gamma, degree)
   (1, 0.001, 3),
   (1, 0.01, 3),
   (1, 0.1, 3),
   (1, 1, 3),
   (1, 10, 3),
   (1, 100, 3),
   (1, 1000, 3)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_mnist_entries.loc[
      (acc_mnist_entries["dim reducer"] == csv_col) &
      (acc_mnist_entries["C"] == c) &
      (acc_mnist_entries["gamma"] == gamma) &
      (acc_mnist_entries["degree"] == degree)
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Parameter (C, gamma, degree)')
ax[0].set_title('Test Accuracy per Reduction Method, Poly Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 1.5)

# -------------------------------------
# test acc for poly, fashion
# -------------------------------------

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_fashion_entries.loc[
      (acc_fashion_entries["dim reducer"] == csv_col) &
      (acc_fashion_entries["C"] == c) &
      (acc_fashion_entries["gamma"] == gamma) &
      (acc_fashion_entries["degree"] == degree)
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Test Accuracy')
ax[1].set_xlabel('Parameter (C, gamma, degree)')
ax[1].set_title('Test Accuracy per Reduction Method, Poly Kernel, MNIST, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 1.5)


# plot acc
fig, ax = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')


# -------------------------------------
# test acc for poly, mnist
# -------------------------------------

# we're going to hand select a few combos from RBF because too many will be a mess.
# 7 groups worked nicely for linear, so we will choose:
parameters = [
 # (C, gamma, degree)
   (0.001, 1, 3),
   (0.01, 1, 3),
   (0.1, 1, 3),
   (1, 1, 3),
   (10, 1, 3),
   (100, 1, 3),
   (1000, 1, 3)
   ] # our x axis groups

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_mnist_entries.loc[
      (acc_mnist_entries["dim reducer"] == csv_col) &
      (acc_mnist_entries["C"] == c) &
      (acc_mnist_entries["gamma"] == gamma) &
      (acc_mnist_entries["degree"] == degree)
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[0].bar(x + offset, time, width, label=attribute)
    ax[0].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Parameter (C, gamma, degree)')
ax[0].set_title('Test Accuracy per Reduction Method, Poly Kernel, MNIST, 5000 iterations')
ax[0].set_xticks(x + width, parameters)
ax[0].legend(loc='upper left', ncols=4)
ax[0].set_ylim(0, 1.5)

# -------------------------------------
# test acc for poly, fashion
# -------------------------------------

# prepare bar graph information
dim_reducers = [("PCA 50", "pca_50"), ("PCA 100", "pca_100"), ("PCA 200", "pca_200"), ("LDA", "lda")]
accs = {
  "PCA 200": [],
  "PCA 100": [],
  "PCA 50": [],
  "LDA": []
}

# for every unique param set on x axis, pull out the fit time
for c, gamma, degree in parameters:
  for (bar_name, csv_col) in dim_reducers:
    acc = acc_fashion_entries.loc[
      (acc_fashion_entries["dim reducer"] == csv_col) &
      (acc_fashion_entries["C"] == c) &
      (acc_fashion_entries["gamma"] == gamma) &
      (acc_fashion_entries["degree"] == degree)
    ]
    # print(time["fit time"].values[0])
    accs[bar_name].append(acc["test acc"].values[0]) # round by 2, cleaner

x = np.arange(len(parameters))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for attribute, time in accs.items():
    offset = width * multiplier
    print(x + offset)
    rects = ax[1].bar(x + offset, time, width, label=attribute)
    ax[1].bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Test Accuracy')
ax[1].set_xlabel('Parameter (C, gamma, degree)')
ax[1].set_title('Test Accuracy per Reduction Method, Poly Kernel, MNIST, 5000 iterations')
ax[1].set_xticks(x + width, parameters)
ax[1].legend(loc='upper left', ncols=4)
ax[1].set_ylim(0, 1.5)


"""
===============================================================
PLOT!
===============================================================
"""
# show the bar graphs!
plt.show()