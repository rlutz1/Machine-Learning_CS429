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


"""
USAGE NOTES


"""



class DataProcessor:

  def __init__(self, data_grab=False, clean=False):

    # initialize the training and testing arrays
    self.X_train = np.array([])
    self.y_train = np.array([])
    self.X_test = np.array([])
    self.y_test = np.array([])

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
    self._split()
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
  def _clean(self):
    for symbol in self.company_symbols.values():
      path = os.path.join(self.raw_csv_dir, f"{symbol}_raw.csv")
      df = pd.read_csv(path) # read in the raw data for this symbol
      df = self._remove_missing(df) # drop missing values
      df = self._remove_outliers(df) # remove outliers--scalers are very sensitive to these
      df = self._normalize(df) # standardize the set with stdscaler or minmax
      df.to_csv(os.path.join(self.clean_csv_dir, f"{symbol}_clean.csv")) # write to the clean dir
  

  # remove all rows with missing values.
  # do not guess/back/forward fill values.
  def _remove_missing(self, df):
    return df.dropna()

  def _remove_outliers(self, df):
    pass

  def _normalize(self, df):
    # fit on training, transform it (fit_transform)
    # transform test data only
    # this needs to be done carefully--the paper mentioned only normalizing the 
    # closing cost (and sentiment in their case.) 
    pass

  # split into test and training sets.
  # this will create M length windows that overlap of all the data, 
  # and then setting the "true" label to the next closing price
  def _split(self):
    pass

  # convenience wrapper method to convert the split
  # sets into pytorch tensors after splitting
  def _to_tensor(self):
    pass
    

# ===========================================
# TESTING

dp = DataProcessor(
  data_grab=True, # TRUE: grab the raw data from yahoo finance and overwrite the existing CSVs
  clean=True # TRUE: clean the raw data and overwrite the existing CSVs
  ) # TODO set to false before any usage so they don't have to repull crap 

# ===========================================