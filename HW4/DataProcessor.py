"""
Task 1:
Find 4 stocks from Yahoo Finance and download their historical prices in one
or several years, e.g., from 02/01/2018 to 01/31/2021. 
You may use the same time window for all stocks.
You should appropriately preprocess your data using scikit-learn and keep the result as the PyTorch tensors.
Randomly split your data to a training set (80%) and a test set (20%)
"""
# imports
import numpy as np
import pandas as pd
import os
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from helper_code.TesterRNN import SimpleRNNModel
import torch.optim as optim # testing only
import torch
import torch.nn as nn
import random


"""
USAGE NOTES


"""



class DataProcessor:

  def __init__(
      self, 
      start_date="2020-01-01", # inclusive, yyyy-mm-dd
      end_date="2021-01-02", # exclusive, yyyy-mm-dd
      target="Close", # target prediction column, should ref a df column name; close price makes most sense
      window_size=50, # window size (M value, how many days to use for training)
      training_percent=0.8, # percent of windows to use for training
      data_grab=False, # grab data and download from yfinance, RAW data, uncleaned, run _get_raw_data()
      clean=False # clean the RAW data and re-write csv, run _clean()
      ):

    # initialize the training and testing arrays, all meta for training sets
    self.X_train = None
    self.y_train = None
    self.X_test = None
    self.y_test = None
    self.train_percent = training_percent 
    self.test_percent = 1 - self.train_percent # convenience only
    self.window_size = window_size 
    self.target = target 
    # for maleability of changing target without having to change the splitting function
    self.target_indeces = { # Close,High,Low,Open,Volume
      "Close": 0,
      "High": 1, 
      "Low": 2,
      "Open": 3, 
      "Volume": 4
    }

    # set up key pair of readable name to stock symbol
    # (legit just look up the symbols on the internet, ha)
    self.company_symbols = {
      "Microsoft": "MSFT",
      "Amazon": "AMZN",
      "NVIDIA": "NVDA",
      "Google": "GOOG" # GOOGL is also an option. not literate enough to know the distict diff yet
    }

    # select a time frame/period to shoot for
    self.start_date = start_date
    self.end_date = end_date

    # path for the raw data dump, NO cleaning
    self.raw_csv_dir = os.path.join("data", "raw")
    # path for CLEAN data dump
    self.clean_csv_dir = os.path.join("data", "clean")

    # encapsulate the data pull from yahoo finance, overwriting the current csv
    if data_grab:
      self._get_raw_data()

    # clean the data, overwriting the current csv
    if clean:
      self._clean()

  # method to pull the raw data initially
  def _get_raw_data(self):
    for symbol in self.company_symbols.values():
      print(f"pinging for {symbol} data over {self.start_date} - {self.end_date}")
      df = yf.download(symbol, start=self.start_date, end=self.end_date)
      print(df[:3]) # print first 3 things for confirmation
      df.to_csv(os.path.join(self.raw_csv_dir, f"{symbol}_raw.csv"), index=False, encoding="utf-8")# write ALL to a csv

  # clean the raw text using various methods
  # TODO: this should only be first two ops, move the other two to a new method: scale/standardize
  def _clean(self):
    for symbol in self.company_symbols.values():
      path = os.path.join(self.raw_csv_dir, f"{symbol}_raw.csv")
      df = pd.read_csv(path) # read in the raw data for this symbol
      df = self._remove_yfinance_header(df, symbol)# remove the weird row that fucking yfinance adds
      df = self._remove_missing(df) # drop missing values
      df.to_csv(os.path.join(self.clean_csv_dir, f"{symbol}_clean.csv"), index=False, encoding="utf-8") # write to the clean dir
  
  # there's a header row yfinance yields a header row for pulling multiple tickers
  # thats the second row: SYMBOL, SYMBOL, ... SYMBOL
  # want to remove that IF it is there.
  def _remove_yfinance_header(self, df, symbol):
    original_num_rows = df.shape[0] # for printing
    df = df[df["Close"] != symbol]
    df = df.reset_index(drop=True)
    print(f"head of df now for {symbol}, removed {original_num_rows - df.shape[0]} rows.")
    print(df[0:3])
    return df

  # remove all rows with missing values.
  # do not guess/back/forward fill values.
  def _remove_missing(self, df):
    original_num_samples = df.shape[0] # for printing
    df_drop = df.dropna(ignore_index=True)
    print(f"dropped {original_num_samples - df_drop.shape[0]} samples with missing data.")
    return df_drop

  # split into test and training sets.
  # this will create M length windows that overlap of all the data, 
  # and then setting the "true" label to the next closing price
  def split(self, symbol):
    path = os.path.join(self.clean_csv_dir, f"{symbol}_clean.csv")
    df = pd.read_csv(path) # read in the raw data for this symbol

    # uncomment below to remove outliers
    # outliers ==============================================
    df = self._remove_outliers(df, auto_drop=True) # remove outliers--scalers are very sensitive to these
    # outliers ==============================================

    df = df.to_numpy() # for ease of use, values only

    # uncomment below to scale
    # scaling ==============================================
    num_samples = df.shape[0]
    # testing
    samples_in_training_window = round((0.8 * num_samples) + (0.2 * self.window_size))
    scaler = MinMaxScaler()
    scaler = scaler.fit(df[:samples_in_training_window]) # fit_transform a scaler to the training set

    df = scaler.transform(df) # transform all samples
    # scaling ==============================================

    # create windows in both train/test of M size, with the "label" being the next day's closing cost
    X_windows, y_windows = self._create_windows(df)

    # find how many windows in our training set
    num_samples = X_windows.shape[0]
    num_train_samples = round(num_samples * self.train_percent) # get the training portion

    # scaling ==============================================
    # num_samples = df.shape[0]
    # # testing
    # samples_in_training_window = round((0.8 * num_samples) + (0.2 * self.window_size))
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(df[:samples_in_training_window]) # fit_transform a scaler to the training set

    # df = scaler.transform(df) # transform all samples
    # scaling ==============================================
    
    # actual splitting
    X_train = X_windows[:num_train_samples]
    y_train = y_windows[:num_train_samples]

    X_test = X_windows[num_train_samples:]
    y_test = y_windows[num_train_samples:]

    print(X_train.shape[0], X_train.shape[1], df.shape[1])
    # shape to: num samples * size of sequence * num features in a sample
    self.X_train = torch.tensor(X_train, dtype=torch.float32).view(X_train.shape[0], X_train.shape[1], df.shape[1])
    self.y_train = y_train

    self.X_test = torch.tensor(X_test, dtype=torch.float32).view(X_test.shape[0], X_test.shape[1], df.shape[1])
    self.y_test = y_test
    # print(X_train[:3])
    # print(y_train[:3])

  # helper method to create the sliding windows to training and testing
  def _create_windows(self, dataset):
    X = [] # containers for windows -> sequence of M days
    y = [] # containers for windows -> the next n days "label" of close price 

    # shift 1 day over for each window
    for i in range(len(dataset) - self.window_size):
      X.append(dataset[i:i + self.window_size]) # grab the next window_size rows
      y.append(dataset[i + self.window_size, self.target_indeces[self.target]]) # grab the NEXT ROW'S target value

    return np.array(X), np.array(y) # return the sliding windows.

  # method to remove outliers using Z score dropping qualification
  def _remove_outliers(self, df, auto_drop=True):

    # helper function to be able to id potential
    # outlier samples per column
    poss_outlier_rows = self._find_outlier_samples(
      df=df,
      col_labels=df.columns.values, 
      num_samples=df.shape[0]
      )
    
    # sanity checking
    print("outlier rows per column:")
    print(poss_outlier_rows)
    
    # for controlling this in case you want to look at the rows FIRST
    # you should likely do this before scaling, since scaling 
    # is sensitive to these outliers.
    # however, it is possible that an outlier is important info, so, leaving room
    if auto_drop: 

      # get ALL outlier rows
      all_poss_outlier_rows = []
      for row_indeces in poss_outlier_rows.values():
        all_poss_outlier_rows += row_indeces

      # remove duplicates
      all_unique_probs = np.unique(all_poss_outlier_rows) 

      # sanity checking
      print(f"removing outlier samples: {all_unique_probs}")

      # actual dropping
      df = df.drop(df.index[all_unique_probs])
      df = df.reset_index(drop=True)

    return df

  #  helper function to identify all outlier samples as qualified by each column
  def _find_outlier_samples(self, df, col_labels, num_samples, sd_threshold=3):
    df = df.astype(float)
    problems_per_col = {} # for holding issues per column for clarity/debugging

    for label in col_labels: # for each column
      z = np.abs(stats.zscore(df[label])) # get the z score of each sample for this column
      outlier_rows = []
      for i in range(num_samples): # for each sample
        # for each corresponding sample, see if within 3 sd's in col
        if not np.isnan(z[i]) and z[i] > sd_threshold: outlier_rows.append(i)
      
      problems_per_col[label] = outlier_rows # add to the problem children

    return problems_per_col

  # convenience wrapper method to convert the split
  # sets into pytorch tensors after splitting
  def _to_tensor(self):
    pass
    

# ===========================================
# TESTING

dp = DataProcessor(
  data_grab=False, # TRUE: grab the raw data from yahoo finance and overwrite the existing CSVs
  clean=False # TRUE: clean the raw data and overwrite the existing CSVs
  ) # TODO set to false before any usage so they don't have to repull crap 

dp.split(dp.company_symbols["NVIDIA"]) # get ready for training

# enforce reproducability
torch.manual_seed(42)
np.random.seed(42)

# Define RNN model
input_size = 5 # 5 features--cols 
hidden_size = 10
output_size = 1
model = SimpleRNNModel(input_size, hidden_size, output_size)

model.fit(dp.X_train, dp.y_train) # quick fit

print(f"MAPE score on train: {model.score(dp.X_train, dp.y_train)}%")
print(f"MAPE score on test: {model.score(dp.X_test, dp.y_test)}%")
# ===========================================