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
"""

class MapCompressor:

  # initializer
  def __init__():
    pass