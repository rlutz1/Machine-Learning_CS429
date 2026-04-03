import tarfile
import os
import pyprind 
import pandas as pd
import sys
import numpy as np


class DataProcessor:

  def __init__(self):
    self._tar_file_path = os.path.join("data", "aclImdb_v1.tar.gz") # relative path name
    self._extraction_path = os.path.join("data", ".extract") # extraction folder, to be ignored by git
    self._csv_path = os.path.join("data", ".csv") # path to write all reviews to as a csv

  def extract(self):

    if not os.path.isdir(self._extraction_path):
      print("Extraction path not made. Creating")

      # extraction script alone, better in a class method for use
      with tarfile.open(self._tar_file_path, 'r:gz') as tar:
        total_files = len(tar.getmembers()) # all files in folder
        for m in tar.getmembers(): 
          print("extracting", m.name, ", ", total_files, " left.") # for knowing it is still running instead of dead hang
          tar.extract(m, path=self._extraction_path) # extract to folder
          total_files -= 1 # ease of reading only

      print("Extraction complete.")
    else:
      print("Extraction path already created. If want to re-extract, delete /data/.extract and re-run this extraction.")

  def process(self, shuffle_seed=42):

    labels = {'pos': 1, 'neg': 0} # positive vibes == 1, negative is 0
    pbar = pyprind.ProgBar(50000, stream=sys.stdout) # set up a progress bar for sanity
    df = pd.DataFrame() # data fram setup

    for s in ('test', 'train'): # for both test and training extraction folders

      for l in ('pos', 'neg'): # for each positive, negative in extraction folders
        path = os.path.join(self._extraction_path, s, l) # build path to reviews
        
        for file in sorted(path): # for each sorted file in path?
          with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
            txt = infile.read() # read in this review file
          df = df.append([[txt, labels[l]]], ignore_index=True) # append row with review and pos/neg label
          pbar.update() # update the progress bar
    df.columns = ['review', 'sentiment'] # label the columns

    # shuffle the data so it's not in complete order
    np.random.seed(shuffle_seed)
    df = df.reindex(np.random.permutation(df.index)) # shuffle up the rows
    df.to_csv(self._csv_path, index=False, encoding="utf-8")# write to CSV


    # sanity checking
    df.head(3)

dp = DataProcessor()
dp.extract()
dp.process()