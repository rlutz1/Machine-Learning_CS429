"""
this class is for use in accomplishing tasks 1 and 2 for the MNIST dataset.
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

"""
EXAMPLE USEAGE:

# creating a new reader will automatically read in the data set and flatten
reader = MNISTReader() # create a new reader

# this is accessible to simply get a printable string name for a specfic image
# from a specific dataset. image numbers are ZERO indexed.
# example: print label from image 1678 from test dataset:
print(reader.label_to_readable_name(1678, MNISTReader.TEST_DATASET))
# do the same with image 1678 from training dataset
print(reader.label_to_readable_name(1678, MNISTReader.TRAIN_DATASET))

# this is another accessible way to simply view the image. this will
# create a black and white plot of the image as a form of sanity checking/as needed.
reader.display_image(1678, MNISTReader.TEST_DATASET)
# do the same with the training set:
reader.display_image(1678, MNISTReader.TRAIN_DATASET)

# to grab the 1D flattened training data and corresponding labels:
reader.train_images
reader.train_labels

# to grab the 1D flattened testing data and corresponding labels:
reader.test_images
reader.test_labels
"""

class MNISTReader:

  # the path to the data sets -- assuming you're running from MidtermProject directory
  # this is shared by ALL instances of this reader
  path = os.path.join(os.getcwd(), "datasets", "mnist")
  # constants for readability
  TRAIN_DATASET = "train" 
  TEST_DATASET = "test"

  # class constructor
  def __init__(self):
    # read in the training images from idx files, convert to np
    train_images = idx2numpy.convert_from_file(os.path.join(MNISTReader.path, 'train-images-idx3-ubyte'))
    # flatten to 1D np array
    self.train_images = train_images.reshape(train_images.shape[0], 28 * 28) 

    # read in the test images from idx files, convert to np
    test_images = idx2numpy.convert_from_file(os.path.join(MNISTReader.path, 't10k-images-idx3-ubyte'))
    # flatten to 1D np array
    self.test_images = test_images.reshape(test_images.shape[0], 28 * 28) 
    
    # grab training labels
    self.train_labels = idx2numpy.convert_from_file(os.path.join(MNISTReader.path, 'train-labels-idx1-ubyte'))
    # grab testing labels
    self.test_labels = idx2numpy.convert_from_file(os.path.join(MNISTReader.path, 't10k-labels-idx1-ubyte'))
    
  # convenience method to display the image very basically as a sanity check
  def display_image(self, image_num, dataset):
    if (dataset == MNISTReader.TRAIN_DATASET): set_to_grab = self.train_images
    elif (dataset == MNISTReader.TEST_DATASET): set_to_grab = self.test_images
    else: 
      print(f"Received an unknown dataset: {dataset}. Try {MNISTReader.TRAIN_DATASET} or {MNISTReader.TEST_DATASET}")
      return
    
    if (image_num < set_to_grab.shape[0]):
      # print(set_to_grab.size)
      plt.title(self.label_to_readable_name(image_num, dataset))
      plt.imshow(set_to_grab.reshape(set_to_grab.shape[0], 28, 28)[image_num], cmap='gray')
      plt.show()
    else:
      print(f"Image number must be no larger than {set_to_grab.shape[0]}.")

  # convenience method to grab the readable name of a specific image number
  # in a specific dataset.
  def label_to_readable_name(self, image_num, dataset):

    # distinguish between datasets
    if (dataset == MNISTReader.TRAIN_DATASET): set_to_grab = self.train_labels
    elif (dataset == MNISTReader.TEST_DATASET): set_to_grab = self.test_labels
    else: 
      print(f"Received an unknown dataset: {dataset}. Try {MNISTReader.TRAIN_DATASET} or {MNISTReader.TEST_DATASET}")
      return ""

    # following readable names as per documentation: https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file
    if (image_num < set_to_grab.size):
   
      match set_to_grab[image_num]:
        case 0:
          return "0"
        case 1:
          return "1"
        case 2:
          return "2"
        case 3:
          return "3"
        case 4:
          return "4"
        case 5:
          return "5"
        case 6:
          return "6"
        case 7:
          return "7"
        case 8:
          return "8"
        case 9:
          return "9"
        
    else:
      print(f"Image number must be no larger than {set_to_grab.shape[0]}.")
      return ""
