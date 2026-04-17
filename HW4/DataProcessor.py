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
import torch



"""
USAGE NOTES

the defaults on initialization are as follows

dp = DataProcessor(
    start_date="2020-01-01", # inclusive, yyyy-mm-dd, start date of data
    end_date="2021-01-02", # exclusive, yyyy-mm-dd, end date of data
    target="Close", # target prediction column, should ref a df column name; close price makes most sense
    window_size=50, # window size (M value, how many days to use for training)
    training_percent=0.8, # percent of raw data to use for training
    scaler=MinMaxScaler(), # the default scaling tactic
    data_grab=False, # grab data and download from yfinance, RAW data, uncleaned, run _get_raw_data() on init
    clean=False # clean the RAW data and re-write csv, run _clean() on init
  )  

getting the data prepped for the model can be done by: 

dp.split(dp.company_symbols["Microsoft"]) # get microsoft ready for training
model.fit(dp.X_train, dp.y_train) # quick fit, for example

dp.split(dp.company_symbols["Google"]) # get google ready for training
model.fit(dp.X_train, dp.y_train) # quick fit, for example

dp.split(dp.company_symbols["NVIDIA"]) # get nvidia ready for training
model.fit(dp.X_train, dp.y_train) # quick fit, for example

dp.split(dp.company_symbols["Amazon"]) # get amazon ready for training
model.fit(dp.X_train, dp.y_train) # quick fit, for example

you may change the above companies in __init__, see dictionary company_symbols. this
can be changed in order to pull new data from different companies. in that case--you'll want to 
set data_grab && clean to True.
"""



class DataProcessor:

  def __init__(
      self, 
      start_date="2020-01-01", # inclusive, yyyy-mm-dd
      end_date="2021-01-02", # exclusive, yyyy-mm-dd
      target="Close", # target prediction column, should ref a df column name; close price makes most sense
      window_size=50, # window size (M value, how many days to use for training)
      training_percent=0.8, # percent of raw data to use for training
      scaler=MinMaxScaler(), # the default scaling tactic
      data_grab=False, # grab data and download from yfinance, RAW data, uncleaned, run _get_raw_data() on init
      clean=False # clean the RAW data and re-write csv, run _clean() on init
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
    self.scaler = scaler # (0, 1) default range scaling
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
    df = df[df["Close"] != symbol] # Close is just the first col, arbitrary choice
    df = df.reset_index(drop=True) # reset the index to 0
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

  # trying something else because this scaling is making me very nervous
  # unless xin tells me no, it is a LOT safer to split the raw data FIRST
  # and then scale, then windows. trying to make windows, split, then scale
  # is a messy mess that i could do, but it needs to be done VERY carefully.
  # in the name of time, let us have more data and split this way for now.
  def split(self, symbol):
    path = os.path.join(self.clean_csv_dir, f"{symbol}_clean.csv")
    df = pd.read_csv(path) # read in the raw data for this symbol

    df = df.to_numpy() # for ease of use, values only

    # find number of training samples
    num_samples = df.shape[0]
    # num_train_samples = round(num_samples * self.train_percent) # get the training portion
    # simple algebra to see what the percentage is of the RAW samples we need to 
    # dedication to training set in order to ensure the trainin percentage specified is
    # of WINDOWS, which are the real testing set.
    percent_needed_to_split_windows_correctly = self.train_percent - ((self.window_size * ((2 * self.train_percent) - 1)) / num_samples)
    num_train_samples = round(num_samples * percent_needed_to_split_windows_correctly) # get the training portion
    
    # splitting
    train_set = df[:num_train_samples]
    test_set = df[num_train_samples:]

    print(f"splitting {num_samples} raw data samples by {percent_needed_to_split_windows_correctly * 100} to ensure good percentage of windows.")
    print(f"total num of windows: {num_samples - (2 * self.window_size)}")
    print(f"{self.train_percent * 100}% of windows: {self.train_percent * (num_samples - (2 * self.window_size))}")
    print(f"training set windows: {train_set.shape[0] - self.window_size}")

    # scale
    train_set_scaled = self.scaler.fit_transform(train_set) # fit_transform a scaler to the training set
    test_set_scaled = self.scaler.transform(test_set) # transform tester

    # create windows
    X_train, y_train = self._create_windows(train_set_scaled)
    X_test, y_test = self._create_windows(test_set_scaled)

    # TODO: removing outlier detection for a moment due to reordering
    
    print(f"shape we're changing to for training for pytorch: {X_train.shape[0]}, {X_train.shape[1]}, {df.shape[1]}")
    # shape to: num samples * size of sequence * num features in a sample (columns)
    self.X_train = torch.tensor(X_train, dtype=torch.float32).view(X_train.shape[0], X_train.shape[1], df.shape[1])
    self.y_train = torch.tensor(y_train, dtype=torch.float32)

    print(f"shape we're changing to for training for pytorch: {X_test.shape[0]}, {X_test.shape[1]}, {df.shape[1]}")
    self.X_test = torch.tensor(X_test, dtype=torch.float32).view(X_test.shape[0], X_test.shape[1], df.shape[1])
    self.y_test = torch.tensor(y_test, dtype=torch.float32)

  # helper method to create the sliding windows to training and testing
  def _create_windows(self, dataset):
    X = [] # containers for windows -> sequence of M days
    y = [] # containers for windows -> the next n days "label" of close price 

    # shift 1 day over for each window
    for i in range(len(dataset) - self.window_size):
      X.append(dataset[i:i + self.window_size]) # grab the next window_size rows
      y.append(dataset[i + self.window_size, self.target_indeces[self.target]]) # grab the NEXT ROW'S target value
    
    # ensuring we're grabbing the right thing lmfao
    # print("first window")
    # print(dataset[0:60])
    # print("target val")
    # print(y[0])
    # print(dataset[60][0])

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

  # inversion for mape statistics.
  # essentialy allows to get the original (or close enough) target value 
  # by inverting the scaling. 
  def inverse_target(self, y_scaled):
    dummy = np.zeros((len(y_scaled), 5))  # y size * 5 features 2d array with 0s
    # print(dummy)
    dummy[:, self.target_indeces[self.target]] = y_scaled # put y scaled into correct col
    # print(dummy)
    original_targets = self.scaler.inverse_transform(dummy)[:, self.target_indeces[self.target]] # return only the rescaled target
    # print(original_targets)
    return original_targets

# ===========================================
# TESTING

dp = DataProcessor(
  data_grab=True, # TRUE: grab the raw data from yahoo finance and overwrite the existing CSVs
  clean=True, # TRUE: clean the raw data and overwrite the existing CSVs
  target="Close",
  start_date="2020-01-01",
  end_date="2024-01-02",
  scaler=MinMaxScaler(),
  training_percent=0.8,
  window_size=50
  ) # TODO set to false before any usage so they don't have to repull crap 


dp.split(dp.company_symbols["Microsoft"]) # get ready for training
# dp.split(dp.company_symbols["Google"]) # get ready for training
# dp.split(dp.company_symbols["Amazon"]) # get ready for training
# dp.split(dp.company_symbols["NVIDIA"]) # get ready for training

# enforce reproducability
torch.manual_seed(42)
np.random.seed(42)

# Define RNN model
input_size = 5 # 5 features--cols 
hidden_size = 10
output_size = 1
model = SimpleRNNModel(input_size, hidden_size, output_size)

model.fit(dp.X_train, dp.y_train) # quick fit

print(f"MAPE score on train: {model.mape(dp.X_train, dp.y_train, dp)}%")
print(f"MAPE score on test: {model.mape(dp.X_test, dp.y_test, dp)}%")
# ===========================================