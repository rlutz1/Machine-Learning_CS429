# CS429 Assignment 4
Assembled by: Roxanne Krause, Kurukulasuriya Leitan, Marnina Willard \
Submission Date: April 24, 2026

## Main Assignment Files 

### DataProcessor.py

This file contains the DataProcessor class that achieves all data pre-processing for use in model training and evaluation. Usage notes are at the top of the file.

This class accomplishes all needs of Task 1 in the assignment, making the data available for all other tasks to easily use.

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
