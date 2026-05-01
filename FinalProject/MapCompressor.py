"""
============================================================================================================
ORIGINAL PROBLEM STATEMENT
============================================================================================================
MAP ABSTRACTION

The task requires to abstract a given BMP map to a data structure which can
be easily processed by an RL procedure. We suggest to use a matrix. A BMP file can be loaded in Python
to a NumPy array. We only consider gray scale images, therefore the array will be 2-dimensional and each
element keeps the value of a pixel in the image. As an example, an image of the resolution 300 x 200 is kept
as a 2-dimensional array of the size 300 x 200. In our implementation, we only consider two colors, white
and black, such that a white pixel indicates a tiny region that is accessible, while the a black pixel indicates
a forbidden region. Hence, the union of the black pixels defines the obstacles in the map.

It is possible to load the image of a map and then give it to an RL procedure. However, it is often time costly
to do so due to the size of the matrix. To improve the performance and still obtain a valid result, the matrix
is often “over-approximated” by a smaller one. For example, Figure 1a shows a map whose size is 528 x 532.
It has 280896 positions which may require an RL procedure to investigate millions of episodes. To avoid such
a high computational cost, we abstract this map by the one in Figure 1b, its size is only 40 x 40. To ensure
the over-approximation property of the abstraction, we may use an approach similar to pooling. For example,
if the given image is of the size 600 x 600 and we want to compute an abstraction of the size 60 x 60, then
we may uniformly subdivide the given map to 60 x 60 many 10 x 10 blocks, each of them is white if there
is no black pixel included, otherwise black. If the size of the original map is not divisible by the abstraction
size, the tail pieces are treated in a similar way
============================================================================================================
"""

"""
============================================================================================================
AUTHOR NOTES
============================================================================================================
the following class is responsible for taking a raw bmp file and compressing (or abstracting it) 
through over approximation into a platable matrix.

the resultant matrix will be 40x40 or less.

USAGE NOTES

this should be used as follows:

-------------------------------------------------------------------------

mc = MapCompressor()

<numpy array> map = mc.compress("/path/to/image.bmp")

-------------------------------------------------------------------------

also offered are some set paths if you just want to use those.
the set paths are to the provided maps 1 - 4.

-------------------------------------------------------------------------

mc = MapCompressor()

<numpy array> map1 = mc.compress(mc.MAP_1_PATH)
<numpy array> map2 = mc.compress(mc.MAP_2_PATH)
<numpy array> map3 = mc.compress(mc.MAP_3_PATH)
<numpy array> map4 = mc.compress(mc.MAP_4_PATH)

-------------------------------------------------------------------------

============================================================================================================
"""

# imports
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

class MapCompressor:

  # initializer
  def __init__(self):
    self.MAP_1_PATH = os.path.join("data", "original_images", "map1.bmp")
    self.MAP_2_PATH = os.path.join("data", "original_images", "map2.bmp")
    self.MAP_3_PATH = os.path.join("data", "original_images", "map3.bmp")
    self.MAP_4_PATH = os.path.join("data", "original_images", "map4.bmp")
    self.BASE_SIZE = 40

  # method to hold the overestimation of this map 
  def compress(self, path_to_bmp=None):

    if (path_to_bmp is None): path_to_bmp = self.MAP_2_PATH # get around defualt issue wtih self

    im = Image.open(path_to_bmp)
    # im.show() # testing

    im_array = np.array(im)
    # print(im_array.shape) # 400x400 for bmp 2 -> correct
    # print(im_array[100:120]) # printing to 255 -> white, 0 -> black

    im_extended = self.reshape(im_array)

    im_compressed = self.overestimate(im_extended)
  

    return im_compressed

  # to make life simpler, we will extend the end of rows
  # and bottom of columns with their existing value.
  def reshape(self, im):
    print(f"original shape {im.shape}")
    num_rows = im.shape[0]
    num_cols = im.shape[1]

    # don't do anything if it's already small
    if num_rows <= self.BASE_SIZE and num_cols <= self.BASE_SIZE: return im

    # extend all the rows of im with edge values thus divisible by base size
    # print((num_rows % self.BASE_SIZE))
    row_extension = (((num_rows // self.BASE_SIZE) + 1) * self.BASE_SIZE) - num_rows
    col_extension = (((num_cols // self.BASE_SIZE) + 1) * self.BASE_SIZE) - num_cols
    extension = np.pad(im, ((0, row_extension), (0, 0)), mode='edge') # extend columns
    extension = np.pad(extension, ((0, 0), (0, col_extension)), mode='edge') # extend rows
    print(f"extending with edge vals {extension.shape}")

    # testing padding 
    # extension = np.pad([[1, 2], [3, 4]], ((0, (num_rows % self.BASE_SIZE)), (0, 0)), mode='edge')
    # print(extension)
    # extension = np.pad(extension, ((0, 0), (0, (num_cols % self.BASE_SIZE))), mode='edge')
    # print(extension)

    return extension # return the extension
  
  # use overestimation method to compress the image
  def overestimate(self, im):
    num_rows = im.shape[0]
    num_cols = im.shape[1]

    # don't do anything if it's already small
    if num_rows <= self.BASE_SIZE and num_cols <= self.BASE_SIZE: return im

    # just in case called directly
    if num_rows % self.BASE_SIZE != 0 or num_cols % self.BASE_SIZE != 0: 
      # print(num_rows )
      # print(num_cols )
      im = self.reshape(im)
    
    # the small kernel sizes to split to
    num_neighborhood_rows = num_rows // self.BASE_SIZE
    num_neighborhood_cols = num_cols // self.BASE_SIZE

    # matrix for compressing values
    compression = np.ones((self.BASE_SIZE, self.BASE_SIZE))
    # kernel to use
    kernel = np.ones((num_neighborhood_rows, num_neighborhood_cols))
    # boundary = False

    # outer iteration of larger array
    for (c_r, R) in zip(range(0, self.BASE_SIZE), range(0, num_rows, num_neighborhood_rows)):
      for (c_c, C) in zip(range(0, self.BASE_SIZE), range(0, num_cols, num_neighborhood_cols)):
        compression[c_r][c_c] = 255
        # kernel iteration
        boundary = False
        for r in range(R, R + num_neighborhood_rows):
          if not boundary:
            for c in range(C, C + num_neighborhood_cols):
              # if ANY pixel is black in the neighborhood
              if im[r][c] == 0: 
                compression[c_r][c_c] = 0
                boundary = True
                break
          else:
            break

    print(compression)
   
    return compression


# ==========================================================================================================
# TESTING
# ==========================================================================================================

mc = MapCompressor()
im = mc.compress(mc.MAP_3_PATH)
plt.imshow(im)
plt.colorbar()
plt.show()