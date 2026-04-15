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
    