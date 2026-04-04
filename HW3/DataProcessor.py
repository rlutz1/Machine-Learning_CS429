import tarfile
import os
import pyprind 
import pandas as pd
import sys
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""
class to encapsulate all pre-processing that occurs.

USAGE DETAILS 

i will make the cleaned csv available primarily and that is the main truth source for 
csv data to train on. therefore, shouldn't need to run the full
extraction or anything but split.

if the cleaned csv is in the expected location (should be), the
constructor will simply call split automatically using that cleaned csv and you
will have the following:
===============================================================
dp = DataProcessor()

dp.train_reviews_str  # string version of reviews
dp.train_reviews      # tf-idf version for training with the model
dp.train_sentiments   # training "labels", pos == 1, neg == 0

dp.test_reviews_str   # string version of reviews
dp.test_reviews       # tf-idf version for testing with the model
dp.test_sentiments    # testing "labels", pos == 1, neg == 0
===============================================================

if you for whatever reason want to regenerate anything/start fresh, you may call:
===============================================================
dp = DataProcessor()
dp.process() 
===============================================================

process() will 
  1. extract() -> extract all 100000 files from the tar file holding reviews
               -> NOTE: the extraction folder created is listed in gitignore!!! it's like 250 mb, and not worth pushing
                        to git! if you want it on your local machine for some reason, just call extract().
               -> this will not run if /extract folder exists. delete manually if you *really* want to reextract 
  2. to_csv()  -> take all the reviews and their sentiment and put together into an UNCLEAN csv
               -> this will take some time! but, progress bar for sanity. usually my laptop takes ~15 mins max
               -> this will not run of data/csv_data.csv exists. delete manually if you *really* want to recreate
  3. clean()   -> clean the gunk out of the unclean csv and write to a new, clean csv that has the reviews shuffled
               -> right now, primarily cleaning out html and editing emojis. may be where to add stopword clearing if done
               -> this will not run of data/csv_clean_data.csv exists. delete manually if you *really* want to recreate
  4. split()   -> split the CLEANED csv into a 70/30 split. will also create the tf-idf versions that
                  will need to be used for training.
               -> will not run if the cleaned csv is not found. ensure data/csv_clean_data.csv exists for this.

"""

class DataProcessor:

  def __init__(self):
    self._tar_file_path = os.path.join("data", "aclImdb_v1.tar.gz") # relative path name
    self._extraction_path = os.path.join("data", "extract") # extraction folder, to be ignored by git
    self._csv_path = os.path.join("data", "csv_data.csv") # path to write all reviews to as a csv
    self._clean_csv_path =  os.path.join("data", "csv_clean_data.csv") # path to write the cleaned reviews to
    
    self.train_reviews_str = None # string version if needed for training
    self.train_reviews = None # tf-idf version for training
    self.train_sentiments = None # training "labels", pos == 1, neg == 0
    self.test_reviews_str = None # string version if needed for testing
    self.test_reviews = None # tf-idf version for testing
    self.test_sentiments = None # testing "labels", pos == 1, neg == 0

    # now that we have the cleaned CSV, we don't need anyone else to extract really
    if os.path.exists(self._clean_csv_path): 
      self.split() # split up the data

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
      print("Unclean CSV path not made. Creating")

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

    else:
      print("CSV data already created. If want to recreate, delete data/csv_data.csv and rerun this process.")

  # read in the uncleaned csv and apply cleaner to all reviews
  def clean(self):
    if not os.path.exists(self._clean_csv_path): 
      print("Creating a cleaned data CSV.")
      df = pd.read_csv(self._csv_path)
      df['review'] = df['review'].apply(self.clean_line) # remove html and clean emoticons
      # any other cleaning here if desired (stopwords?)
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

  # split into 70/30 train/test sets
  def split(self):
    if os.path.exists(self._clean_csv_path): 
      df = pd.read_csv(self._clean_csv_path) # read in the cleaned data
      
      # 35000 -> 70% of 50000
      # note: loc is end-inclusive with slice notation! fun fact!
      self.train_reviews_str = df.loc[:34999, 'review'].values
      self.train_reviews = self.tf_idf_transform(self.train_reviews_str)
      self.train_sentiments  = df.loc[:34999, 'sentiment'].values
      
      # 15000 -> 30% of 50000
      self.test_reviews_str = df.loc[35000:, 'review'].values
      self.test_reviews = self.tf_idf_transform(self.test_reviews_str)
      self.test_sentiments  = df.loc[35000:, 'sentiment'].values
     
      # sanity checking
      # print(self.train_reviews_str.size)
      # print(self.test_reviews_str.size)
    else: 
      print(f"Cleaned version of CSV reviews not found. Need {self._clean_csv_path}")

  # method to use for (1) producing a bag of words and then (2) transforming to a tf-idf 
  def tf_idf_transform(self, text_reviews):
    count = CountVectorizer()
    bag_of_words = count.fit_transform(text_reviews) # get the bag of words
    tfidf = TfidfTransformer()
    return tfidf.fit_transform(bag_of_words) # return the tf idf breakdown of words

  # method to conduct ALL steps in order. note this may take some time, 
  # do not call unless you need to, likely if you want a fresh start
  def process(self):
    self.extract()
    self.to_csv()
    self.clean()
    self.split()

# testing as coding:
dp = DataProcessor()
# dp.process() # process the data into usable.

# np.set_printoptions(threshold=sys.maxsize) # for seeing more of that array

print(dp.train_reviews_str[:3])
print(dp.train_reviews.toarray()[:3])
print(dp.train_sentiments[:3])

print("=========")

print(dp.test_reviews_str[:3])
print(dp.test_reviews.toarray()[:3])
print(dp.test_sentiments[:3])
