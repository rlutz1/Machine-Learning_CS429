import tarfile
import os
import pyprind 
import pandas as pd
import sys


class DataProcessor:

  def __init__(self):
    self._tar_file_path = os.path.join("data", "aclImdb_v1.tar.gz") # relative path name

  def extract(self):

    extraction_path = os.path.join("data", ".extract") # extraction folder, to be ignored by git

    if not os.path.isdir(extraction_path):
      print("Extraction path not made. Creating")

      # extraction script alone, better in a class method for use
      with tarfile.open(self._tar_file_path, 'r:gz') as tar:
        total_files = len(tar.getmembers()) # all files in folder
        for m in tar.getmembers(): 
          print("extracting", m.name, ", ", total_files, " left.") # for knowing it is still running instead of dead hang
          tar.extract(m, path=extraction_path) # extract to folder
          total_files -= 1 # ease of reading only
    
    print("Extraction path already created. If want to re-extract, delete /data/.extract and re-run this extraction.")

  def process(self):
    extraction_path = os.path.join("data", ".extract") # extraction folder, to be ignored by git

    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000, stream=sys.stdout)
    df = pd.DataFrame()

    for s in ('test', 'train'):

      for l in ('pos', 'neg'):
        path = os.path.join(extraction_path, s, l)
        for file in sorted(os.listdir(path)):
          with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
            txt = infile.read()
          df = df.append([[txt, labels[l]]], ignore_index=True)
          pbar.update()
    df.columns = ['review', 'sentiment']

dp = DataProcessor()
dp.extract()
dp.process()