import tarfile
import os
import pyprind 
import pandas as pd
import sys
import numpy as np
import re

"""
class to encapsulate all pre-processing that occurs
"""

class DataProcessor:

  def __init__(self):
    self._tar_file_path = os.path.join("data", "aclImdb_v1.tar.gz") # relative path name
    self._extraction_path = os.path.join("data", "extract") # extraction folder, to be ignored by git
    self._csv_path = os.path.join("data", "csv_data.csv") # path to write all reviews to as a csv
    self._clean_csv_path =  os.path.join("data", "csv_clean_data.csv") # path to write the cleaned reviews to
    self.train_reviews = []
    self.train_sentiments = []
    self.test_reviews = []
    self.test_sentiments = []

  # method to extract the files from the tar file
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
      print("Extraction path already created. If want to re-extract, delete /data/extract and re-run this extraction.")

  # method to process the extracted text files into single, shuffled csv
  def to_csv(self, shuffle_seed=42):

    if not os.path.exists(self._csv_path):
      print("Extraction path not made. Creating")

      labels = {'pos': 1, 'neg': 0} # positive vibes == 1, negative is 0
      pbar = pyprind.ProgBar(50000, stream=sys.stdout) # set up a progress bar for sanity
      df = pd.DataFrame({'review': [], 'sentiment': []}) # the dataframe start

      for s in ('test', 'train'): # for both test and training extraction folders

        for l in ('pos', 'neg'): # for each positive, negative in extraction folders

          path = os.path.join(os.getcwd(), self._extraction_path, "aclImdb", s, l) # build path to reviews
        
          with os.scandir(path) as files:
            for file in files: # for each sorted file in path?
              if file.is_file():
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                  txt = infile.read() # read in this review file

                row = pd.DataFrame({"review": [txt], "sentiment": [labels[l]]})
                df = pd.concat([df, row], ignore_index=True) # append row with review and pos/neg label
                pbar.update() # update the progress bar

      df.columns = ['review', 'sentiment'] # label the columns

      # shuffle the data so it's not in complete order
      np.random.seed(shuffle_seed)
      df = df.reindex(np.random.permutation(df.index)) # shuffle up the rows
      df.to_csv(self._csv_path, index=False, encoding="utf-8")# write to CSV


      # sanity checking
      df.head(3)
    else:
      print("CSV data already created. If want to recreate, delete data/csv_data.csv and rerun this process.")

  # read in the uncleaned csv and apply cleaner to all reviews
  def clean(self):
    if not os.path.exists(self._clean_csv_path): 
      print("Creating a cleaned data CSV.")
      df = pd.read_csv(self._csv_path)
      df['review'] = df['review'].apply(self.clean_line)
      df.to_csv(self._clean_csv_path, index=False, encoding="utf-8")# write to CSV
    else:
      print("Clean CSV data already created. If want to recreate, delete data/csv_clean_data.csv and rerun this process.")


  # method to read and then clean specific line of text from html and emoticons
  def clean_line(self, text):
    text = re.sub('<[^>]*>', '', text) # remove the html markup
    # find emoticons and put them at the end, removing noses
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) 
    text = (re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

  def split(self):
    # TODO to be the 70/30 splitter
    # train_review, sent, test_review, sent
    pass

  # conduct all steps needed in order to process the reviews
  def process(self):
    dp.extract()
    dp.to_csv()
    dp.clean()
    dp.split()




dp = DataProcessor()
dp.process() # process the data into usable.
