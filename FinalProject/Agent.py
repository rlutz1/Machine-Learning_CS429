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
from Environment import Environment

class Agent:

  

  def __init__(self, environment, map_width=40, map_height=40):
    # make a 3d matrix: 
    # dimension 1: x coords (map_width)
    # dimension 2: y coords (map_height)
    # dimension 3: actions (U, D, L, R) (4)
    self.Q_TABLE = np.zeros((map_width, map_height, 4))
    # print(self.Q_TABLE.shape)
    self.environment = environment

  # interaction with environment, update Qtable
  def interact(self, 
               curr_state: tuple,        # next coordinate (x, y)
               next_action: int,         # next action {0, 1, 2, 3} -> (LEFT / RIGHT / UP / DOWN)
               strategy: str = "S1" # strategy to use
               ):
    # probe the environment
    next_state, reward, done = self.environment.step(state=curr_state, action=next_action, strategy=strategy)
    if not done:
      self.update_Q_TABLE(state=next_state, action=action, reward=reward)

  # is this where policy eval/update goes? unclear... hold on
  def update_Q_TABLE(self,  state: tuple, action: int, reward:float):
    x = state[0]
    y = state[1]
    self.Q_TABLE[x][y][action] = self.algo()

  # redefine this with sarsa update or QLearn?
  def algo(self):
    pass

# ==========================================================================================================
# TESTING
# ==========================================================================================================

mc = MapCompressor()
im = mc.compress(mc.MAP_1_PATH)

environment = Environment(im, (0, 0)) # testing only

agent = Agent(environment, im.shape[0], im.shape[1]) # agent 
# plt.imshow(im)
# plt.colorbar()
# plt.show()