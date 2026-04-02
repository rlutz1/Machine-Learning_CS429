import tarfile
import os

TAR_FILE = os.path.join(os.getcwd(), "data", "aclImdb_v1.tar.gz")
EXTRACTION_PATH = os.path.join(os.getcwd(), "data", ".extract")


with tarfile.open(TAR_FILE, 'r:gz') as tar:
  total_files = len(tar.getmembers())
  for m in tar.getmembers():
    print("extracting...", m.name, ", ", total_files, " left.")
    tar.extract(m, path=EXTRACTION_PATH)
    total_files -= 1