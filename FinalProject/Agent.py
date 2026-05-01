"""
BUILDING THE AGENT

You are required to write a Python class for the agent which keeps a
Q-table. At least a function for the interaction with an environment should be implemented. It performs an
interaction with an environment and updates the Q-table. As it is explained in the class, the Q-table should
be a 3-D matrix. S1 can be a naive strategy in which a very high reward is returned when the target position
is reached and a very negative reward is returned when an obstacle is reached
"""
# imports
import numpy as np
from MapCompressor import MapCompressor

class Agent:

  

  def __init__(self, environment, map_width=40, map_height=40):
    # make a 3d matrix: 
    # dimension 1: x coords (map_width)
    # dimension 2: y coords (map_height)
    # dimension 3: actions (U, D, L, R) (4)
    self.Q_TABLE = np.zeros((map_width, map_height, 4))
    print(self.Q_TABLE.shape)
    self.environment = environment

# ==========================================================================================================
# TESTING
# ==========================================================================================================

mc = MapCompressor()
im = mc.compress(mc.MAP_1_PATH)

agent = Agent(im.shape[0], im.shape[1])
# plt.imshow(im)
# plt.colorbar()
# plt.show()