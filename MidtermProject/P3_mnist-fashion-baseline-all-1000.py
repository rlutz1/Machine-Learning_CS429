"""
file to run many combos at 1000 iterations, linear, rbf, poly
"""

from MNISTFashionReader import MNISTFashionReader
from helpers.TimeWrappers import TimedTransform, TimedClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
import os
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

# setting up paths for data collection
MNIST_FASHION_DATA_PATH = os.path.join(os.getcwd(), "data", "P3", "mnist-fashion") # directory to write data to for collection
INITIAL_FIT_TIMES = "initial-fit-times.csv" # for the initial fit times of everything in the pipeline
TEST_AND_TRAIN_ACC = "test-train-acc.csv" # for the accuracy of final SVC on test, then train data
FINAL_TIMES = "final-times.csv" # for the final times, that includes final transforms of test data

# transformer names, need this to be effectively an enum to use in switch case
class DimReducers: 
  STD_SCALER = "std scaler"
  PCA_50 = "pca_50"
  PCA_100 = "pca_100"
  PCA_200 = "pca_200"
  LDA_N_CLASSES = "lda"


# set up and read in the data sets
reader = MNISTFashionReader()

# for ensuring no state leakage during script runs.
# get a new transformer object each time. 
# either this or cloning.
def get_transformer(name):
  match name:
    case DimReducers.STD_SCALER:
      return TimedTransform(StandardScaler()) # use standard scaling
    case DimReducers.PCA_50:
      return TimedTransform(PCA(n_components = 50)) # 50 features
    case DimReducers.PCA_100:
      return TimedTransform(PCA(n_components = 100)) # 100 features
    case DimReducers.PCA_200:
      return TimedTransform(PCA(n_components = 200)) # 200 features
    case DimReducers.LDA_N_CLASSES:
      return TimedTransform(LDA()) # capped at 9 features (n_classes - 1)

# for ease of testing all for different purposes
dim_reducers = [
  DimReducers.PCA_50,
  DimReducers.PCA_100,
  DimReducers.PCA_200,
  DimReducers.LDA_N_CLASSES
]


# =========================================================
# LINEAR KERNEL
# =========================================================

"""
For larger values of C, a smaller margin will be accepted if the decision function is 
better at classifying all training points correctly. 
A lower C will encourage a larger margin, therefore a simpler decision function, 
at the cost of training accuracy.
low -> underfit/less accurate/simpler
high -> overfit/more accurate/more complex
"""
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000] # some initial values to try

# staying the same for this script
kernel = "linear"
num_iterations = 1000

for dim_reducer_name in dim_reducers:

  # for recording data to csv throughout process
  csv_init_times_line = {
    "dim reducer": [], 
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "scale time": [],
    "dim reduce time": [],
    "fit time": [], 
    "converged": []
    }

  csv_test_train_acc_line = {
    "dim reducer": [],
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "test acc": [],
    "train acc": []
  }

  csv_final_times_line = {
    "dim reducer": [], 
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "scale time": [],
    "dim reduce time": [],
    "fit time": []
  }

  for c in C: # to create a base line for C params

    # gather some metadata for the csv entry
    csv_init_times_line["dim reducer"].append(dim_reducer_name)
    csv_init_times_line["kernel"].append(kernel)
  
    csv_test_train_acc_line["dim reducer"].append(dim_reducer_name)
    csv_test_train_acc_line["kernel"].append(kernel)

    csv_final_times_line["dim reducer"].append(dim_reducer_name)
    csv_final_times_line["kernel"].append(kernel)

    # add the parameter for records
    csv_init_times_line["C"].append(str(c))
    csv_init_times_line["gamma"].append("") # for linear only, empty
    csv_init_times_line["degree"].append("")
    csv_init_times_line["num iterations"].append(str(num_iterations))

    csv_test_train_acc_line["C"].append(str(c))
    csv_test_train_acc_line["gamma"].append("") # for linear only, empty
    csv_test_train_acc_line["degree"].append("")
    csv_test_train_acc_line["num iterations"].append(str(num_iterations))

    csv_final_times_line["C"].append(str(c))
    csv_final_times_line["gamma"].append("") # for linear only, empty
    csv_final_times_line["degree"].append("")
    csv_final_times_line["num iterations"].append(str(num_iterations))

    # initialize the transformers and svc
    sc = get_transformer(DimReducers.STD_SCALER)
    dim_reducer = get_transformer(dim_reducer_name)
    svc = TimedClassifier(SVC(kernel=kernel, C=c, max_iter=num_iterations)) 

    # construct a pipeline
    pipeline = Pipeline ([
      (DimReducers.STD_SCALER, sc), # use a standard scalar
      (dim_reducer_name, dim_reducer), # use the set dim reducer
      ("SVC", svc) # svc classifier
    ], verbose=True) # verbose true as a sanity check on times

    with warnings.catch_warnings(record=True) as w: # custom catch of convergence warnings
      warnings.simplefilter('always')
      
      # fit!
      pipeline.fit(reader.train_images, reader.train_labels)    
  
      # capture the time taken in all steps
      csv_init_times_line["scale time"].append(str(sc.total_time()))
      csv_init_times_line["dim reduce time"].append(str(dim_reducer.total_time()))
      csv_init_times_line["fit time"].append(str(svc.total_time()))

      if any(issubclass(warnings.category, ConvergenceWarning) for warnings in w):
          csv_init_times_line["converged"].append("DID NOT CONVERGE") # for records to know what did not converge
      else:
        csv_init_times_line["converged"].append("")

    # fitting done, now score the accuracy regardless of convergence
    score_test = pipeline.score(reader.test_images, reader.test_labels) # score on testers
    score_train = pipeline.score(reader.train_images, reader.train_labels) # score on trainers

    # capture the accuracy on both test and train data
    csv_test_train_acc_line["test acc"].append(str(score_test))
    csv_test_train_acc_line["train acc"].append(str(score_train))

    # capture the FINAL times taken in all steps, accounting for final transforms of data
    csv_final_times_line["scale time"].append(str(sc.total_time()))
    csv_final_times_line["dim reduce time"].append(str(dim_reducer.total_time()))
    csv_final_times_line["fit time"].append(str(svc.total_time()))

  # write this info to appropriate CSV
  df = pd.DataFrame(csv_init_times_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, INITIAL_FIT_TIMES), index=False, mode='a', header=False)

  df = pd.DataFrame(csv_test_train_acc_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, TEST_AND_TRAIN_ACC), index=False, mode='a', header=False)

  df = pd.DataFrame(csv_final_times_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, FINAL_TIMES), index=False, mode='a', header=False)

        
# =========================================================
# RADIAL BASIS FUNCTION (RBF) KERNEL
# =========================================================


# for ref:
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

"""
For larger values of C, a smaller margin will be accepted if the decision function is 
better at classifying all training points correctly. 
A lower C will encourage a larger margin, therefore a simpler decision function, 
at the cost of training accuracy.
low -> underfit/less accurate/simpler
high -> overfit/more accurate/more complex
"""
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000] # some initial values to try

"""
The gamma parameters can be seen as the inverse of the radius of influence of 
samples selected by the model as support vectors.
low -> underfit
high -> overfit
"""
gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# staying the same for this script
kernel = "rbf"
num_iterations = 1000

for dim_reducer_name in dim_reducers:

  
  # for recording data to csv throughout process
  csv_init_times_line = {
    "dim reducer": [], 
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "scale time": [],
    "dim reduce time": [],
    "fit time": [], 
    "converged": []
    }

  csv_test_train_acc_line = {
    "dim reducer": [],
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "test acc": [],
    "train acc": []
  }

  csv_final_times_line = {
    "dim reducer": [], 
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "scale time": [],
    "dim reduce time": [],
    "fit time": []
  }

  for c in C: # to create a base line for C params

    for g in gamma: # to create a base line for gamma params

      # gather some metadata for the csv entry
      csv_init_times_line["dim reducer"].append(dim_reducer_name)
      csv_init_times_line["kernel"].append(kernel)

      csv_test_train_acc_line["dim reducer"].append(dim_reducer_name)
      csv_test_train_acc_line["kernel"].append(kernel)

      csv_final_times_line["dim reducer"].append(dim_reducer_name)
      csv_final_times_line["kernel"].append(kernel)

      # add the parameter for records
      csv_init_times_line["C"].append(str(c))
      csv_init_times_line["gamma"].append(str(g)) 
      csv_init_times_line["degree"].append("")
      csv_init_times_line["num iterations"].append(str(num_iterations))

      csv_test_train_acc_line["C"].append(str(c))
      csv_test_train_acc_line["gamma"].append(str(g))
      csv_test_train_acc_line["degree"].append("")
      csv_test_train_acc_line["num iterations"].append(str(num_iterations))

      csv_final_times_line["C"].append(str(c))
      csv_final_times_line["gamma"].append(str(g)) 
      csv_final_times_line["degree"].append("")
      csv_final_times_line["num iterations"].append(str(num_iterations))

      # initialize the transformers and svc
      sc = get_transformer(DimReducers.STD_SCALER)
      dim_reducer = get_transformer(dim_reducer_name)
      svc = TimedClassifier(SVC(kernel=kernel, C=c, gamma=g, max_iter=num_iterations)) 

      # construct a pipeline
      pipeline = Pipeline ([
        (DimReducers.STD_SCALER, sc), # use a standard scalar
        (dim_reducer_name, dim_reducer), # use the set dim reducer
        ("SVC", svc) # svc classifier
      ], verbose=True) # verbose true as a sanity check on times

      with warnings.catch_warnings(record=True) as w: # custom catch of convergence warnings
        warnings.simplefilter('always')
        
        # fit!
        pipeline.fit(reader.train_images, reader.train_labels)    
    
        # capture the time taken in all steps
        csv_init_times_line["scale time"].append(str(sc.total_time()))
        csv_init_times_line["dim reduce time"].append(str(dim_reducer.total_time()))
        csv_init_times_line["fit time"].append(str(svc.total_time()))

        if any(issubclass(warnings.category, ConvergenceWarning) for warnings in w):
            csv_init_times_line["converged"].append("DID NOT CONVERGE") # for records to know what did not converge
        else:
          csv_init_times_line["converged"].append("")

      # fitting done, now score the accuracy regardless of convergence
      score_test = pipeline.score(reader.test_images, reader.test_labels) # score on testers
      score_train = pipeline.score(reader.train_images, reader.train_labels) # score on trainers

      # capture the accuracy on both test and train data
      csv_test_train_acc_line["test acc"].append(str(score_test))
      csv_test_train_acc_line["train acc"].append(str(score_train))

      # capture the FINAL times taken in all steps, accounting for final transforms of data
      csv_final_times_line["scale time"].append(str(sc.total_time()))
      csv_final_times_line["dim reduce time"].append(str(dim_reducer.total_time()))
      csv_final_times_line["fit time"].append(str(svc.total_time()))

  # write this info to appropriate CSV
  df = pd.DataFrame(csv_init_times_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, INITIAL_FIT_TIMES), index=False, mode='a', header=False)

  df = pd.DataFrame(csv_test_train_acc_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, TEST_AND_TRAIN_ACC), index=False, mode='a', header=False)

  df = pd.DataFrame(csv_final_times_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, FINAL_TIMES), index=False, mode='a', header=False)
 

# =========================================================
# POLYNOMIAL KERNEL
# =========================================================

"""
For larger values of C, a smaller margin will be accepted if the decision function is 
better at classifying all training points correctly. 
A lower C will encourage a larger margin, therefore a simpler decision function, 
at the cost of training accuracy.
low -> underfit/less accurate/simpler
high -> overfit/more accurate/more complex
"""
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000] # some initial values to try

"""
The gamma parameters can be seen as the inverse of the radius of influence of 
samples selected by the model as support vectors.
low -> underfit
high -> overfit
"""
gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

"""
The polynomial degree parameter starts at a default 3. so, 
we will test around that value, with slightly higher and lower values.
"""
degree = [1, 2, 3, 4, 5] # including 1, but suspect it behaves as a linear kernel

# staying the same for this script
kernel = "poly"
num_iterations = 1000

for dim_reducer_name in dim_reducers:

  # for recording data to csv throughout process
  csv_init_times_line = {
    "dim reducer": [], 
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "scale time": [],
    "dim reduce time": [],
    "fit time": [], 
    "converged": []
    }

  csv_test_train_acc_line = {
    "dim reducer": [],
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "test acc": [],
    "train acc": []
  }

  csv_final_times_line = {
    "dim reducer": [], 
    "kernel": [],
    "C": [],
    "gamma": [],
    "degree": [],
    "num iterations": [],
    "scale time": [],
    "dim reduce time": [],
    "fit time": []
  }

  for c in C: # to create a base line for C params

    for g in gamma: # to create a base line for gamma params

      for d in degree:

        # gather some metadata for the csv entry
        csv_init_times_line["dim reducer"].append(dim_reducer_name)
        csv_init_times_line["kernel"].append(kernel)

        csv_test_train_acc_line["dim reducer"].append(dim_reducer_name)
        csv_test_train_acc_line["kernel"].append(kernel)

        csv_final_times_line["dim reducer"].append(dim_reducer_name)
        csv_final_times_line["kernel"].append(kernel)

        # add the parameter for records
        csv_init_times_line["C"].append(str(c))
        csv_init_times_line["gamma"].append(str(g)) 
        csv_init_times_line["degree"].append(str(d))
        csv_init_times_line["num iterations"].append(str(num_iterations))

        csv_test_train_acc_line["C"].append(str(c))
        csv_test_train_acc_line["gamma"].append(str(g))
        csv_test_train_acc_line["degree"].append(str(d))
        csv_test_train_acc_line["num iterations"].append(str(num_iterations))

        csv_final_times_line["C"].append(str(c))
        csv_final_times_line["gamma"].append(str(g)) 
        csv_final_times_line["degree"].append(str(d))
        csv_final_times_line["num iterations"].append(str(num_iterations))

        # initialize the transformers and svc
        sc = get_transformer(DimReducers.STD_SCALER)
        dim_reducer = get_transformer(dim_reducer_name)
        svc = TimedClassifier(SVC(kernel=kernel, C=c, gamma=g, degree=d, max_iter=num_iterations)) 

        # construct a pipeline
        pipeline = Pipeline ([
          (DimReducers.STD_SCALER, sc), # use a standard scalar
          (dim_reducer_name, dim_reducer), # use the set dim reducer
          ("SVC", svc) # svc classifier
        ], verbose=True) # verbose true as a sanity check on times

        with warnings.catch_warnings(record=True) as w: # custom catch of convergence warnings
          warnings.simplefilter('always')
          
          # fit!
          pipeline.fit(reader.train_images, reader.train_labels)    
      
          # capture the time taken in all steps
          csv_init_times_line["scale time"].append(str(sc.total_time()))
          csv_init_times_line["dim reduce time"].append(str(dim_reducer.total_time()))
          csv_init_times_line["fit time"].append(str(svc.total_time()))

          if any(issubclass(warnings.category, ConvergenceWarning) for warnings in w):
              csv_init_times_line["converged"].append("DID NOT CONVERGE") # for records to know what did not converge
          else:
            csv_init_times_line["converged"].append("")

        # fitting done, now score the accuracy regardless of convergence
        score_test = pipeline.score(reader.test_images, reader.test_labels) # score on testers
        score_train = pipeline.score(reader.train_images, reader.train_labels) # score on trainers

        # capture the accuracy on both test and train data
        csv_test_train_acc_line["test acc"].append(str(score_test))
        csv_test_train_acc_line["train acc"].append(str(score_train))

        # capture the FINAL times taken in all steps, accounting for final transforms of data
        csv_final_times_line["scale time"].append(str(sc.total_time()))
        csv_final_times_line["dim reduce time"].append(str(dim_reducer.total_time()))
        csv_final_times_line["fit time"].append(str(svc.total_time()))

  # write this info to appropriate CSV
  df = pd.DataFrame(csv_init_times_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, INITIAL_FIT_TIMES), index=False, mode='a', header=False)

  df = pd.DataFrame(csv_test_train_acc_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, TEST_AND_TRAIN_ACC), index=False, mode='a', header=False)

  df = pd.DataFrame(csv_final_times_line)
  df.to_csv(os.path.join(MNIST_FASHION_DATA_PATH, FINAL_TIMES), index=False, mode='a', header=False)