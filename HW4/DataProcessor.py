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


"""
USAGE NOTES


"""



class DataProcessor:

  def __init__(self, data_grab=False, clean=False):

    # initialize the training and testing arrays, all meta for training sets
    self.X_train = np.array([])
    self.y_train = np.array([])
    self.X_test = np.array([])
    self.y_test = np.array([])
    self.train_percent = 0.8 # convenience only
    self.test_percent = 1 - self.train_percent
    self.window_size = 50 # start with 50 "timesteps"; our M; ie: 50 == one training window is 50 days long
    self.overlap_step = 5 # allowable overlap of windows
    # self.prediction_timesteps = 1 # start with 1 "timestep"; our n; ie: 1 == predict next 1 day target
    self.target = "Close" # target predication is the closing price
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
      "OpenAI": "OPAI.PVT",
      "Anthropic": "ANTH.PVT",
      "NVIDIA": "NVDA",
      "Google": "GOOG" # GOOGL is also an option. not literate enough to know the distict diff yet
    }

    # select a time frame/period to shoot for
    self.time_frame = "1y" # 1 year to start
    # self.time_frame = "1mo" # testing

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

    # split the data into test and train
    self._split() # TODO: should be splitting first, and cleaning the training data, transforming test with THAT

    # convenience wrapper to remember to convert np to pytorch tensor
    self._to_tensor()

  # method to pull the raw data initially
  def _get_raw_data(self):
    for symbol in self.company_symbols.values():
      print(f"pinging for {symbol} data over {self.time_frame}")
      df = yf.download(symbol, period=self.time_frame)
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
      # TODO uncomment
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
  def _split(self):
    for symbol in self.company_symbols.values():
      path = os.path.join(self.clean_csv_dir, f"{symbol}_clean.csv")
      df = pd.read_csv(path) # read in the raw data for this symbol

      df = self._remove_outliers(df, auto_drop=True) # remove outliers--scalers are very sensitive to these

      df = df.to_numpy()

      # print(df[:3])

      
      # X_testing_windows, y_testing_windows = self._create_windows(testing_set_scaled)

      # print(X_windows[:3])
      # print(y_windows[:3])

      # extremely convoluted, but hold on bucko
      # scaling
      num_samples = df.shape[0]
      # testing
      samples_in_training_window = round((0.8 * num_samples) + (0.2 * self.window_size))
      scaler = MinMaxScaler()
      scaler = scaler.fit(df[:samples_in_training_window]) # fit_transform a scaler to the training set

      df = scaler.transform(df) # transform all samples


      # create windows in both train/test of M size, with the "label" being the next day's closing cost
      X_windows, y_windows = self._create_windows(df)

      # next steps
      # SPLIT the test and train set
      num_samples = X_windows.shape[0]
      num_train_samples = round(num_samples * self.train_percent) # get the training portion
      
      # actual splitting
      X_train = X_windows[:num_train_samples]
      y_train = y_windows[:num_train_samples]

      X_test = X_windows[num_train_samples:]
      y_test = y_windows[num_train_samples:]

      print(X_train[:3])
      print(y_train[:3])

      

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
    probs = self._find_outlier_samples(
      df=df,
      col_labels=df.columns.values, 
      num_samples=df.shape[0]
      )
    
    # sanity checking
    print("outlier rows per column:")
    print(probs)
    
    # for controlling this in case you want to look at the rows FIRST
    # you should likely do this before scaling, since scaling 
    # is sensitive to these outliers.
    # however, it is possible that an outlier is important info, so, leaving room
    if auto_drop: 

      # get ALL outlier rows
      all_probs = []
      for row_indeces in probs.values():
        all_probs += row_indeces

      # remove duplicates
      all_unique_probs = np.unique(all_probs) 

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

# ===========================================