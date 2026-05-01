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

  def __init__(self, 
               environment, 
               map_width=40, 
               map_height=40,
               strategy_to_use="S1"):
    # make a 3d matrix: 
    # dimension 1: x coords (map_width)
    # dimension 2: y coords (map_height)
    # dimension 3: actions (U, D, L, R) (4)
    self.Q_TABLE = np.zeros((map_width, map_height, 4))
    self.environment = environment # environment to probe
    self.strategy = strategy_to_use # tie to the agent
    # for not hardcoding every little thing.
    self.actions = {
      "L": 0, "R": 1, "U": 2, "D": 3
    }

  # interaction with environment simply a wrapper
  def interact(self, 
               curr_state: tuple,        # next coordinate (x, y)
               next_action: int,         # next action {0, 1, 2, 3} -> (LEFT / RIGHT / UP / DOWN)
               strategy: str = "S1" # strategy to use
               ):
    # probe the environment with your current state (S) and desired next action (A).
    # take action (A), observe next state (S') and reward (R)
    return self.environment.step(state=curr_state, action=next_action, strategy=strategy)

  # override with SARSA or QLearn?
  def develop_policy(self, episodes=1000):
    pass


class QLearnAgent(Agent):

  def __init__(self, environment, learn_rate=0.5, discount=0.5,  map_width=40, map_height=40):
    super().__init__(environment, map_width, map_height) # init the agent, with the Q table
    self.learn_rate = learn_rate
    self.discount = discount

  # initiate QLearn algorithm?
  def develop_policy(self, episodes=1000):
    # next_state, reward, done = self.interact()
    pass
 

# ==========================================================================================================
# TESTING
# ==========================================================================================================

mc = MapCompressor()
im = mc.compress(mc.MAP_1_PATH)

environment = Environment(im, (0, 0)) # testing only

# agent = Agent(environment, im.shape[0], im.shape[1]) # agent 
agent = QLearnAgent(environment, im.shape[0], im.shape[1]) # agent 
# plt.imshow(im)
# plt.colorbar()
# plt.show()