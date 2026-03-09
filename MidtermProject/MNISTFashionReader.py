"""
this class is for use in accomplishing tasks 1 and 2 for the MNIST Fashion dataset.
this class should be used for tasks 3 and 4 for ease of training.

Task 1:

Download and import the training and test images from MNIST and Fashion MNIST. The imported
data are required to be kept in NumPy array format. Complete the following tasks on both of the datasets.

Task 2:

Perform a data format transformation by flattening each image to a 1-D NumPy array. You may
use the NumPy function reshape
"""
import os
import matplotlib.pyplot as plt
import idx2numpy
import numpy as np

# print(path)

class MNISTFashionReader:

  # the path to the data sets -- assuming you're running from MidtermProject directory
  # this is shared by ALL instances of this reader
  path = os.path.join(os.getcwd(), "datasets", "mnist-fashion")

  def __init__(self):
    # read in the training images from idx files, convert to np
    train_images = idx2numpy.convert_from_file(os.path.join(MNISTFashionReader.path, 'train-images-idx3-ubyte'))
    # flatten to 1D np array
    self.train_images = train_images.reshape(train_images.shape[0], 28 * 28) 

    # read in the test images from idx files, convert to np
    test_images = idx2numpy.convert_from_file(os.path.join(MNISTFashionReader.path, 't10k-images-idx3-ubyte'))
    # flatten to 1D np array
    self.test_images = test_images.reshape(test_images.shape[0], 28 * 28) 
    
    # grab training labels
    self.train_labels = idx2numpy.convert_from_file(os.path.join(MNISTFashionReader.path, 'train-labels-idx1-ubyte'))
    # grab testing labels
    self.test_labels = idx2numpy.convert_from_file(os.path.join(MNISTFashionReader.path, 't10k-labels-idx1-ubyte'))
    

  def display_train_image(self, image_num):
    if (image_num >= 0 and image_num < self.train_images.shape[0]):
      print(self.train_images.size)
      plt.imshow(self.train_images.reshape(self.train_images.shape[0], 28, 28)[image_num], cmap='gray')
      plt.show()
    else:
      print(f"Image number must be on interval [0, {self.train_images.shape[0]})")


    
    
   



reader = MNISTFashionReader()

reader.display_train_image(0)