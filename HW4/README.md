# CS429 Assignment 4
Assembled by: Roxanne Krause, Kurukulasuriya Leitan, Marnina Willard \
Submission Date: April 24, 2026

## Main Assignment Files

### DataProcessor.py

This file contains the DataProcessor class that achieves all data pre-processing for use in model training and evaluation. Usage notes are at the top of the file.

This class accomplishes all needs of Task 1 in the assignment, making the data available for all other tasks to easily use.

### RNN.py

Baseline recurrent neural network model. Uses a single RNN layer to predict the next day's closing price from the past 60 days of stock data. Trained with Adam optimizer and mini-batch gradient descent. Run this file to train and evaluate the baseline RNN on all four stocks.

### GRU.py

Improved model using Gated Recurrent Units. Same input/output setup as the RNN, but replaces the recurrent layer with two stacked GRU layers. The gating mechanism helps the model better capture long-term patterns in the price data. Outperforms the baseline RNN on all four stocks. Run this file to train and evaluate the GRU on all four stocks.

## data Directory

### raw Directory

Contains the RAW, uncleaned data pull as CSV from `yfinance.download()` with the following targets and parameters:
+ Data Start Date: January 1, 2020
+ Data End Date: January 1, 2024
+ Companies (as signified in file name with their ticker symbol): Microsoft, Amazon, NVIDIA, Google

### clean Directory

Contains the CLEAN data pull as CSV from `yfinance.download()` with the following targets and parameters:
+ Data Start Date: January 1, 2020
+ Data End Date: January 1, 2024
+ Companies (as signified in file name with their ticker symbol): Microsoft, Amazon, NVIDIA, Google

See the report section Task 1 for details on the cleaning techniques used.
