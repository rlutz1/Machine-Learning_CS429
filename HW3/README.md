# CS429 Assignment 3
Assembled by: Roxanne Krause, Kurukulasuriya Leitan, Marnina Willard \
Submission Date: April 10, 2026

## Main Assignment Files 

The following files were used to gather all data needed to accomplish the necessary tasks for Assignment 3. 

### DataProcessor.py

This file was used to accomplish all needs of Task 1: preparing the reviews as tf-idf format. This was used to extract all the reviews and labels, transform to csv, and convert to tf-idf format with Scikit Learn. Usage details can be seen at the top of this file, and this usage was used throughout the following files. 

### FeedForwardNN.py

This file was used to accomplish all needs of primarily Tasks 2 and 3, while allowing for tuning in 5a. This shows our implementation of a Feed Forward Neural Network baseline model. When run, this will run our best accuracy obtained parameters (as detailed in the report) without dropout and compare against a LogisticRegression model as requested.

### KFold.py

This file was used to accomplish all needs of Task 4, the k-fold technique being implemented. When run, this will run a single split and 5-fold model and print a comparison of the lines.

### Bagger.py

This file was used to accomplish all needs to Task 5b, the bagging technique being implemented. We used a general majority vote approach, with all models trained on disjoint subsets of the larger training set. When run, this will run a 5 model bag with our best implementation of the baseline model and no dropout.

## `data` Directory 

This contains
+ **aclImdb_v1.tar.gz** - The TAR file of reviews that was used for extraction.
+ **csv_data.csv** - The uncleaned reviews and labels, shuffled.
+ **csv_clean_data.csv** - The cleaned reviews and labels, shuffled. Used for training.
+ **csv_more_clean_data.csv** - The reviews with the extra tactics of stemming and removing stop words. This ended up yielding consistently lower accuracy when used, so is not used in training. The noise proved to be beneficial in our case.