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

"""
USAGE NOTES


"""



class DataProcessor:

  def __init__(self):

    # set up key pair of readable name to stock symbol
    self.company_symbols = {
      "OpenAI": "OPAI.PVT",
      "Anthropic": "ANTH.PVT",
      "NVIDIA": "NVDA",
      "Google": "GOOG" # GOOGL is also an option. not literate enough to know the distict diff yet
    }

    # testing
    # print(" ".join(list(self.company_symbols.values())))

    # select a time frame/period to shoot for
    # self.time_frame = "1y" # 1 year to start
    self.time_frame = "1mo" # testing

    # path for the raw data dump, NO cleaning
    self.raw_csv_dir = os.path.join("data", "raw")

    # encapsulate the data pull from yahoo finance
    self._get_init_data()


  # method to pull the raw data initially
  def _get_init_data(self):
    for symbol in self.company_symbols.values():
      print(f"pinging for {symbol} data over {self.time_frame}")
      df = yf.download(symbol, period=self.time_frame)
      print(df[:3]) # print first 3 things
      df.to_csv(os.path.join(self.raw_csv_dir, f"{symbol}_raw.csv"), index=False, encoding="utf-8")# write ALL to a csv
    

# ===========================================
# TESTING

dp = DataProcessor()

# ===========================================