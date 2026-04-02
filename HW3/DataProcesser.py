import tarfile
import os

TAR_FILE = os.path.join(os.getcwd(), "data", "aclImdb_v1.tar.gz") # the tar file, TO BE stored by git
EXTRACTION_PATH = os.path.join(os.getcwd(), "data", ".extract") # extraction folder, to be ignored by git

# extraction script alone, better in a class method for use
with tarfile.open(TAR_FILE, 'r:gz') as tar:
  total_files = len(tar.getmembers()) # all files in folder
  for m in tar.getmembers(): 
    print("extracting...", m.name, ", ", total_files, " left.") # for knowing it is still running instead of dead hang
    tar.extract(m, path=EXTRACTION_PATH) # extract to folder
    total_files -= 1 # ease of reading only